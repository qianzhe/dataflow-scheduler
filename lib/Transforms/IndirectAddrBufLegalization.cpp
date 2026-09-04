//===----------------------------------------------------------------------===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// IndirectAddrBufLegalization: materialize ind_addr_buf window and per-entry
// loops required by the hardware indirect address buffer constraint.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "dataflow-scheduler/Dialect/KTDF/KTDFAttributes.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFEnums.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

#define PASS_NAME "indirect-addr-buf-legalization"
#define DEBUG_TYPE PASS_NAME

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_INDIRECTADDRBUFLEGALIZATIONPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

/// Drop dimension `drop_dim` from `set` (which has `orig_num_dims` dimensions).
/// Constraints that reference `drop_dim` are removed; the remaining dimensions
/// above `drop_dim` are shifted down by one.
static mlir::IntegerSet dropDimFromIntegerSet(mlir::IntegerSet set,
                                              unsigned drop_dim) {
  mlir::MLIRContext* ctx = set.getContext();
  unsigned orig_num_dims = set.getNumDims();
  unsigned new_num_dims = orig_num_dims - 1;

  // Build replacement array: d_i → d_i for i < drop_dim,
  //                          d_i → d_{i-1} for i > drop_dim,
  //                          d_drop_dim → constant 0 (placeholder; constraints
  //                          referencing it will be removed below).
  llvm::SmallVector<mlir::AffineExpr> dim_replacements(orig_num_dims);
  for (unsigned i = 0; i < orig_num_dims; ++i) {
    if (i < drop_dim)
      dim_replacements[i] = mlir::getAffineDimExpr(i, ctx);
    else if (i == drop_dim)
      dim_replacements[i] = mlir::getAffineConstantExpr(0, ctx);
    else  // i > drop_dim
      dim_replacements[i] = mlir::getAffineDimExpr(i - 1, ctx);
  }

  llvm::SmallVector<mlir::AffineExpr> new_constraints;
  llvm::SmallVector<bool> new_eq_flags;
  auto constraints = set.getConstraints();
  auto eq_flags = set.getEqFlags();
  for (unsigned i = 0; i < constraints.size(); ++i) {
    // Skip any constraint that involves the dimension being absorbed.
    if (constraints[i].isFunctionOfDim(drop_dim)) continue;
    // Remap remaining dimensions.
    auto remapped = constraints[i].replaceDimsAndSymbols(
        dim_replacements, /*symReplacements=*/{});
    new_constraints.push_back(remapped);
    new_eq_flags.push_back(eq_flags[i]);
  }

  return mlir::IntegerSet::get(new_num_dims, set.getNumSymbols(),
                               new_constraints, new_eq_flags);
}

/// Drop dimension `drop_dim` from `map` (which maps `orig_num_dims` dims).
/// The result expression at position `drop_dim` is removed; remaining
/// expressions with dim indices above `drop_dim` are shifted down.
static mlir::AffineMap dropDimFromAffineMap(mlir::AffineMap map,
                                            unsigned drop_dim) {
  mlir::MLIRContext* ctx = map.getContext();
  unsigned orig_num_dims = map.getNumDims();
  unsigned new_num_dims = orig_num_dims - 1;

  // Build dim replacements: shift dims above drop_dim down by one.
  llvm::SmallVector<mlir::AffineExpr> dim_replacements(orig_num_dims);
  for (unsigned i = 0; i < orig_num_dims; ++i) {
    if (i < drop_dim)
      dim_replacements[i] = mlir::getAffineDimExpr(i, ctx);
    else if (i == drop_dim)
      dim_replacements[i] = mlir::getAffineConstantExpr(0, ctx);
    else
      dim_replacements[i] = mlir::getAffineDimExpr(i - 1, ctx);
  }

  llvm::SmallVector<mlir::AffineExpr> new_results;
  for (unsigned i = 0; i < map.getNumResults(); ++i) {
    if (i == drop_dim) continue;
    new_results.push_back(
        map.getResult(i).replaceDimsAndSymbols(dim_replacements, {}));
  }
  return mlir::AffineMap::get(new_num_dims, map.getNumSymbols(), new_results,
                              ctx);
}

/// Narrow an access_tile_order map by dropping both the result at
/// `tile_dim` and the corresponding input dimension.  access_tile_order is
/// always a square permutation map (same number of inputs and outputs), so
/// we must drop the input dim that maps to output position `tile_dim`.
/// For a plain identity map that is just `dropDimFromAffineMap(map, tile_dim)`.
/// For a permuted map, the input dim to drop is the AffineDimExpr position
/// found at result `tile_dim`.
static mlir::AffineMap dropTileDimFromOrderMap(mlir::AffineMap map,
                                               unsigned tile_dim) {
  // The result at `tile_dim` must be a plain AffineDimExpr identifying the
  // input dimension that drives this tile output.
  auto dim_expr = mlir::dyn_cast<mlir::AffineDimExpr>(map.getResult(tile_dim));
  unsigned input_dim_to_drop = dim_expr ? dim_expr.getPosition() : tile_dim;
  return dropDimFromAffineMap(map, input_dim_to_drop);
}

/// Return a copy of `shape` with the element at position `drop_dim` removed.
static llvm::SmallVector<int64_t> dropShapeDim(llvm::ArrayRef<int64_t> shape,
                                               unsigned drop_dim) {
  llvm::SmallVector<int64_t> result;
  result.reserve(shape.size() - 1);
  for (unsigned i = 0; i < shape.size(); ++i) {
    if (i != drop_dim) result.push_back(shape[i]);
  }
  return result;
}

/// Extract a constant trip count for dimension `dim_idx` from `set`.
/// Looks for a constraint of the form `-d_{dim_idx} + (N-1) >= 0` and returns
/// N.  Returns std::nullopt if no such constraint is found.
static std::optional<int64_t> getTripCount(mlir::IntegerSet set,
                                           unsigned dim_idx) {
  for (unsigned i = 0; i < set.getNumConstraints(); ++i) {
    if (set.isEq(i)) continue;
    mlir::AffineExpr expr = set.getConstraint(i);
    // We expect a constraint that is purely a function of dim_idx and has the
    // form `-d + C >= 0` (i.e. d <= C).  Concretely: constant + (-1)*d >= 0.
    if (!expr.isFunctionOfDim(dim_idx)) continue;

    // Walk to find the coefficient of dim_idx and the constant part.
    // We accept only simple two-term sums: (constant_expr + dim_expr*coeff).
    auto bin = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expr);
    if (!bin) continue;
    if (bin.getKind() != mlir::AffineExprKind::Add) continue;

    // One side should be the constant, the other a scaled dim.
    mlir::AffineExpr lhs = bin.getLHS(), rhs = bin.getRHS();
    mlir::AffineConstantExpr const_part;
    mlir::AffineExpr dim_part;
    if (mlir::isa<mlir::AffineConstantExpr>(lhs)) {
      const_part = mlir::cast<mlir::AffineConstantExpr>(lhs);
      dim_part = rhs;
    } else if (mlir::isa<mlir::AffineConstantExpr>(rhs)) {
      const_part = mlir::cast<mlir::AffineConstantExpr>(rhs);
      dim_part = lhs;
    } else {
      continue;
    }

    // dim_part should be -1 * d_{dim_idx}  (i.e. MulExpr with coeff = -1).
    auto mul = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(dim_part);
    if (!mul || mul.getKind() != mlir::AffineExprKind::Mul) continue;
    auto dim_expr = mlir::dyn_cast<mlir::AffineDimExpr>(mul.getLHS());
    auto coeff_expr =
        mlir::dyn_cast<mlir::AffineConstantExpr>(mul.getRHS());
    if (!dim_expr || !coeff_expr) continue;
    if (dim_expr.getPosition() != dim_idx) continue;
    if (coeff_expr.getValue() != -1) continue;

    // constraint: const_part + (-1)*d >= 0  ⇒  d <= const_part  ⇒  N = const_part + 1
    return const_part.getValue() + 1;
  }
  return std::nullopt;
}

/// Return true if `type` carries dimension info that the narrowing propagation
/// must follow.  MemRefs are excluded: base descriptor memrefs (desc_1, desc_2,
/// addr_buf) are out-of-scope, and the IAB memref is handled by steps 4/5
/// before propagateNarrowing is called.
static bool isNarrowable(mlir::Type type) {
  return mlir::isa<mlir::ktdp::AccessTileType, mlir::RankedTensorType>(type);
}

enum class PropDir { DOWN, UP };

/// Narrowing dispatch: mutate `op` in-place so that all shaped operands and
/// results reflect the dimension drop already applied to upstream values via
/// RAUW.  Must NOT propagate further — that is the caller's job.
///
/// If the op was rebuilt (RAUW + erase), `*rebuilt_out` is set to the new op.
/// Otherwise `*rebuilt_out` is left unchanged (caller should initialize to
/// nullptr).  Pass nullptr if the caller does not need this information.
///
/// Returns failure() if `op` is unknown or internally inconsistent.
static mlir::LogicalResult narrowOp(
    mlir::Operation* op, unsigned tile_dim_to_drop, mlir::Value for_iv,
    int64_t iab_rank, mlir::Location loc, mlir::MLIRContext* ctx,
    mlir::Operation** rebuilt_out = nullptr) {
  // ── ConstructIndirectAccessTileOp ────────────────────────────────────────
  // Pre-marked in narrowed_ops; must never be reached here.
  if (mlir::isa<mlir::ktdp_lowering::ConstructIndirectAccessTileOp>(op))
    return op->emitError(
        "ConstructIndirectAccessTileOp unexpectedly reached narrowOp");

  // ── ktdp.load ─────────────────────────────────────────────────────────────
  // Called only from the DOWN direction (AT already narrowed by RAUW).
  // Rebuild so the result tensor type matches the narrowed AT shape.
  // The result element type is the data type (e.g. f16), NOT the AT element
  // type (which is always 'index' for indirect access tiles).
  if (auto load = mlir::dyn_cast<mlir::ktdp::LoadOp>(op)) {
    auto res_type =
        mlir::cast<mlir::RankedTensorType>(load.getResult().getType());
    mlir::OpBuilder builder(load);
    auto new_load = mlir::ktdp::LoadOp::create(
        builder, load.getLoc(), load.getAccessTile(),
        res_type.getElementType());
    load.getResult().replaceAllUsesWith(new_load.getResult());
    load.erase();
    if (rebuilt_out) *rebuilt_out = new_load.getOperation();
    return mlir::success();
  }

  // ── ktdp.construct_access_tile ────────────────────────────────────────────
  // Narrow shape, access_tile_set, access_tile_order, and indices.
  // Loop IV is placed at tile_dim_to_drop; c0 fills the remaining slots.
  if (auto at = mlir::dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(op)) {
    auto old_type =
        mlir::cast<mlir::ktdp::AccessTileType>(at.getResult().getType());
    auto new_type = mlir::ktdp::AccessTileType::get(
        dropShapeDim(old_type.getShape(), tile_dim_to_drop),
        old_type.getElementType());
    mlir::IntegerSet new_set =
        dropDimFromIntegerSet(at.getAccessTileSet().getValue(), tile_dim_to_drop);
    mlir::AffineMap new_order =
        dropTileDimFromOrderMap(at.getAccessTileOrder(), tile_dim_to_drop);
    unsigned new_rank = new_set.getNumDims();
    mlir::AffineMap new_base_map =
        mlir::AffineMap::getMultiDimIdentityMap(new_rank, ctx);

    mlir::OpBuilder ib(at);
    mlir::Value c0 =
        mlir::arith::ConstantIndexOp::create(ib, loc, 0).getResult();
    llvm::SmallVector<mlir::Value> new_indices(new_rank, c0);
    if (tile_dim_to_drop < new_rank) new_indices[tile_dim_to_drop] = for_iv;

    mlir::OpBuilder ab(at);
    auto new_at = mlir::ktdp::ConstructAccessTilesOp::create(
        ab, at.getLoc(), new_type, at.getBase(), new_base_map, new_indices,
        new_set, new_order);
    at.getResult().replaceAllUsesWith(new_at.getResult());
    at.erase();
    if (rebuilt_out) *rebuilt_out = new_at.getOperation();
    return mlir::success();
  }

  // ── ktdp_lowering.ConstructMemoryViewOp ──────────────────────────────────
  // Only narrow the IAB memref (rank == iab_rank before this absorption).
  // Always drops the leading dimension (dim 0) — IAB memref space is
  // independent of tile_dim_to_drop in the data-tile space.
  if (auto mv = mlir::dyn_cast<mlir::ktdp_lowering::ConstructMemoryViewOp>(op)) {
    auto mv_type = mlir::cast<mlir::MemRefType>(mv.getResult().getType());
    if (mv_type.getRank() != iab_rank) return mlir::success();  // not the IAB mv
    llvm::ArrayRef<int64_t> old_shape = mv_type.getShape();
    llvm::SmallVector<int64_t> new_shape =
        dropShapeDim(old_shape, /*drop_dim=*/0);
    int64_t new_rank = static_cast<int64_t>(new_shape.size());
    llvm::SmallVector<int64_t> new_strides(new_rank);
    int64_t stride = 1;
    for (int64_t i = new_rank - 1; i >= 0; --i) {
      new_strides[i] = stride;
      stride *= new_shape[i];
    }
    mlir::IntegerSet new_coord =
        dropDimFromIntegerSet(mv.getCoordinateSet().getValue(), /*drop_dim=*/0);
    mlir::Attribute memory_space = mv.getMemorySpace();
    auto new_mv_type = mlir::MemRefType::get(new_shape, mlir::IndexType::get(ctx),
                                              mlir::MemRefLayoutAttrInterface{},
                                              memory_space);
    mlir::OpBuilder builder(mv);
    auto new_mv = mlir::ktdp_lowering::ConstructMemoryViewOp::create(
        builder, mv.getLoc(), new_mv_type, mv.getOffset(),
        /*sizes=*/mlir::ValueRange{}, /*strides=*/mlir::ValueRange{},
        mlir::DenseI64ArrayAttr::get(ctx, new_shape),
        mlir::DenseI64ArrayAttr::get(ctx, new_strides), memory_space,
        mlir::IntegerSetAttr::get(new_coord));
    mv.getResult().replaceAllUsesWith(new_mv.getResult());
    mv.erase();
    if (rebuilt_out) *rebuilt_out = new_mv.getOperation();
    return mlir::success();
  }

  // ── linalg.generic ───────────────────────────────────────────────────────
  // Narrow indexing_maps, iterator_types, and result type.
  // tensor.empty outs operands are NOT handled here — the walk will visit
  // them via UP from the outs operand and call narrowOp(tensor.empty) there.
  if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(op)) {
    llvm::SmallVector<mlir::AffineMap> new_maps;
    for (mlir::AffineMap m : generic.getIndexingMapsArray())
      new_maps.push_back(dropDimFromAffineMap(m, tile_dim_to_drop));
    generic.setIndexingMapsAttr(mlir::ArrayAttr::get(
        ctx,
        llvm::to_vector(llvm::map_range(new_maps, [](mlir::AffineMap m) {
          return mlir::cast<mlir::Attribute>(mlir::AffineMapAttr::get(m));
        }))));

    auto old_iters = generic.getIteratorTypesArray();
    llvm::SmallVector<mlir::Attribute> new_iters;
    for (size_t i = 0; i < old_iters.size(); ++i) {
      if (i != tile_dim_to_drop)
        new_iters.push_back(
            mlir::linalg::IteratorTypeAttr::get(ctx, old_iters[i]));
    }
    generic.setIteratorTypesAttr(mlir::ArrayAttr::get(ctx, new_iters));

    auto old_res =
        mlir::cast<mlir::RankedTensorType>(generic.getResult(0).getType());
    generic.getResult(0).setType(mlir::RankedTensorType::get(
        dropShapeDim(old_res.getShape(), tile_dim_to_drop),
        old_res.getElementType()));
    return mlir::success();
  }

  // ── tensor.empty ──────────────────────────────────────────────────────────
  // Rebuild with one fewer dimension; RAUW and erase.
  if (auto empty = mlir::dyn_cast<mlir::tensor::EmptyOp>(op)) {
    auto old_type =
        mlir::cast<mlir::RankedTensorType>(empty.getResult().getType());
    auto new_type = mlir::RankedTensorType::get(
        dropShapeDim(old_type.getShape(), tile_dim_to_drop),
        old_type.getElementType());
    mlir::OpBuilder builder(empty);
    auto new_empty =
        mlir::tensor::EmptyOp::create(builder, empty.getLoc(), new_type, {});
    empty.getResult().replaceAllUsesWith(new_empty.getResult());
    empty.erase();
    if (rebuilt_out) *rebuilt_out = new_empty.getOperation();
    return mlir::success();
  }

  // ── ktdp.store ────────────────────────────────────────────────────────────
  // Sink: no results, no shape-encoding attributes.  Both operands are already
  // updated by RAUW from upstream rebuilds.
  if (mlir::isa<mlir::ktdp::StoreOp>(op)) return mlir::success();

  // ── Unknown ───────────────────────────────────────────────────────────────
  return op->emitError("unknown op in narrowing propagation");
}

/// Bidirectional def-use narrowing propagation.
///
/// Starting from `new_op` (already rebuilt with a narrowed result type),
/// propagate the shape change through the shaped-value graph **within the
/// single basic block that contains `new_op`** (the just-materialised scf.for
/// body).  Ops whose defining/owning block is different are silently skipped —
/// this prevents the walk from crossing into outer-scope ops (base memref
/// views, output desc memrefs, etc.) that must not be narrowed.
///
/// Parameters:
///   new_op           — Already-rebuilt ConstructIndirectAccessTileOp.
///   tile_dim_to_drop — Access-tile / tensor dimension to remove.
///   for_iv           — Induction variable of the just-emitted scf.for.
///   iab_rank         — Rank of the IAB memref *before* this absorption.
///   pre_narrowed     — Ops already correctly rebuilt by the caller (step 4/5
///                      of materializeWindowLoops): pre-inserted into
///                      narrowed_ops so the walk skips re-mutating them.
///   loc / ctx        — Needed when building new ops.
static mlir::LogicalResult propagateNarrowing(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp new_op,
    unsigned tile_dim_to_drop, mlir::Value for_iv, int64_t iab_rank,
    llvm::ArrayRef<mlir::Operation*> pre_narrowed, mlir::Location loc,
    mlir::MLIRContext* ctx) {
  // The propagation is bounded to this block — the scf.for body.
  mlir::Block* const scope_block = new_op->getBlock();

  // ── State ─────────────────────────────────────────────────────────────────
  llvm::SmallVector<std::pair<mlir::Value, PropDir>> worklist;
  llvm::DenseSet<mlir::OpOperand*> visited_uses;
  // narrowed_ops: ops that have already been mutated (or are pre-narrowed).
  // The walk calls narrowOp exactly once per op.
  llvm::DenseSet<mlir::Operation*> narrowed_ops;
  // opaque_ops: ops that are correctly built and must not be propagated
  // through — new_op itself and all pre_narrowed step-4/5 ops.
  llvm::DenseSet<mlir::Operation*> opaque_ops;

  // ── Seed ──────────────────────────────────────────────────────────────────
  narrowed_ops.insert(new_op.getOperation());
  opaque_ops.insert(new_op.getOperation());
  // Pre-mark ops already rebuilt by step 4/5 (IAB mv, IAB fill AT, addr_buf
  // AT, fill load) so the walk does not attempt to narrow or propagate through
  // them again.
  for (mlir::Operation* op : pre_narrowed) {
    if (op) {
      narrowed_ops.insert(op);
      opaque_ops.insert(op);
    }
  }

  // Follow every user of the narrowed AT result (DOWN).
  worklist.push_back({new_op.getResult(), PropDir::DOWN});

  // ── Worklist loop ─────────────────────────────────────────────────────────
  while (!worklist.empty()) {
    auto [val, dir] = worklist.pop_back_val();

    if (dir == PropDir::DOWN) {
      // Fan out to every user of val within scope_block.
      // Snapshot uses first: narrowOp may RAUW, invalidating the iterator.
      llvm::SmallVector<mlir::OpOperand*> uses;
      for (mlir::OpOperand& use : val.getUses()) uses.push_back(&use);

      for (mlir::OpOperand* use_ptr : uses) {
        if (visited_uses.count(use_ptr)) continue;
        visited_uses.insert(use_ptr);

        mlir::Operation* user = use_ptr->getOwner();

        // Skip ops outside the scf.for body — they must not be narrowed.
        if (user->getBlock() != scope_block) continue;

        if (!narrowed_ops.count(user)) {
          if (mlir::failed(narrowOp(user, tile_dim_to_drop, for_iv, iab_rank,
                                    loc, ctx)))
            return mlir::failure();
          narrowed_ops.insert(user);
        }

        // Opaque ops (new_op + pre_narrowed) are already complete; do not
        // propagate through their operands or results.
        if (opaque_ops.count(user)) continue;

        // If narrowOp rebuilt `user` (RAUW + erase), `user` is now dead.
        // The rebuilt op uses `val` as an operand too; find it among the
        // live (non-snapshotted) uses of `val` and continue propagation.
        if (!user->getBlock()) {
          for (mlir::OpOperand& live_use : val.getUses()) {
            if (visited_uses.count(&live_use)) continue;
            mlir::Operation* rebuilt = live_use.getOwner();
            if (rebuilt->getBlock() != scope_block) continue;
            // Mark the rebuilt op as already narrowed so subsequent UP visits
            // of its results do not call narrowOp on it again.
            narrowed_ops.insert(rebuilt);
            for (mlir::Value r : rebuilt->getResults())
              if (isNarrowable(r.getType()))
                worklist.push_back({r, PropDir::DOWN});
            for (mlir::Value o : rebuilt->getOperands())
              if (isNarrowable(o.getType()))
                worklist.push_back({o, PropDir::UP});
          }
          continue;
        }

        for (mlir::Value operand : user->getOperands()) {
          if (isNarrowable(operand.getType()))
            worklist.push_back({operand, PropDir::UP});
        }
        for (mlir::Value result : user->getResults()) {
          if (isNarrowable(result.getType()))
            worklist.push_back({result, PropDir::DOWN});
        }
      }
    } else {
      // UP: visit the single defining op of val.
      mlir::Operation* def_op = val.getDefiningOp();
      // Skip block args, and ops defined outside the scf.for body.
      if (!def_op || def_op->getBlock() != scope_block) continue;

      if (!narrowed_ops.count(def_op)) {
        // ktdp.load is only rebuilt when its AT operand has already been
        // narrowed.  In the UP direction the AT may not be narrowed yet, so
        // we skip narrowOp here and let the generic propagation push UP from
        // the AT operand.  narrowOp(load) will be called correctly via DOWN
        // once the AT is narrowed and its result is pushed DOWN.
        if (!mlir::isa<mlir::ktdp::LoadOp>(def_op)) {
          mlir::Operation* rebuilt = nullptr;
          if (mlir::failed(narrowOp(def_op, tile_dim_to_drop, for_iv, iab_rank,
                                    loc, ctx, &rebuilt)))
            return mlir::failure();
          narrowed_ops.insert(def_op);
          if (rebuilt) {
            // Mark rebuilt op as already narrowed.
            narrowed_ops.insert(rebuilt);
            // Push rebuilt op's narrowable results DOWN so their users (e.g. a
            // ktdp.load using a just-narrowed construct_access_tile) can be
            // narrowed in turn.  Necessary for scatter where the source AT is
            // narrowed via UP and its downstream load must be rebuilt.
            for (mlir::Value r : rebuilt->getResults())
              if (isNarrowable(r.getType()))
                worklist.push_back({r, PropDir::DOWN});
          }
        }
        // For ktdp.load in the UP direction: fall through to generic
        // propagation below (push AT UP, result DOWN) without calling narrowOp.
      }

      // Opaque ops: do not propagate through their operands or results.
      if (opaque_ops.count(def_op)) continue;

      // Guard: def_op may have been erased by narrowOp.
      if (!def_op->getBlock()) continue;

      for (mlir::Value operand : def_op->getOperands()) {
        if (isNarrowable(operand.getType()))
          worklist.push_back({operand, PropDir::UP});
      }
      for (mlir::Value result : def_op->getResults()) {
        if (isNarrowable(result.getType()))
          worklist.push_back({result, PropDir::DOWN});
      }
    }
  }
  return mlir::success();
}

/// Collect all ops in `block` that must move into the window loop body.
///
/// The reachability set is seeded from `indirect_op` and grown by:
///   UP:   follow def-use edges through operands whose type is narrowable
///         (AccessTileType or RankedTensorType).  When an IAB memref operand
///         is encountered, also include every other user of that memref
///         (the IAB fill chain) and walk their operands upward.
///   DOWN: follow def-use edges through users of `indirect_op`'s result,
///         then UP from each user's non-indirect operands (pulls in the
///         scatter source-data chain that is only reachable downstream).
///
/// Returns {splice_begin_op, splice_end_op}: the earliest
/// ktdp.construct_access_tile and the latest ktdp.store in block order
/// among the collected set.  Both are guaranteed to be non-null because
/// the input MLIR is in the canonical "indirect_load + compute + store"
/// form.
static std::pair<mlir::Operation*, mlir::Operation*> findSpliceBoundary(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp indirect_op) {
  mlir::Block* block = indirect_op->getBlock();
  llvm::DenseSet<mlir::Operation*> visited;
  llvm::SmallVector<mlir::Operation*> worklist;

  auto enqueue = [&](mlir::Operation* op) {
    if (op && op->getBlock() == block && visited.insert(op).second)
      worklist.push_back(op);
  };

  // Seed: the indirect op itself.
  enqueue(indirect_op.getOperation());

  // Walk the worklist; each item is processed for both UP and DOWN edges.
  // We do a single unified worklist: for each visited op we push its
  // defining-op (UP) and its users (DOWN) that live in the same block.
  //
  // Special case: when we encounter the IAB memref operand of indirect_op,
  // we also enqueue all other users of that memref (the fill chain).
  mlir::Value iab_memref = indirect_op.getIndAddrBufMemref();

  while (!worklist.empty()) {
    mlir::Operation* cur = worklist.pop_back_val();

    // UP: walk operands.
    for (mlir::Value operand : cur->getOperands()) {
      // If this operand IS the IAB memref, include all its users (fill chain).
      if (operand == iab_memref) {
        for (mlir::OpOperand& use : iab_memref.getUses())
          enqueue(use.getOwner());
        continue;
      }
      // Otherwise follow narrowable operands (AccessTile / tensor) upward.
      if (mlir::isa<mlir::ktdp::AccessTileType, mlir::RankedTensorType>(
              operand.getType())) {
        if (mlir::Operation* def = operand.getDefiningOp())
          enqueue(def);
      }
    }

    // DOWN: walk users of each result.
    for (mlir::Value result : cur->getResults()) {
      if (mlir::isa<mlir::ktdp::AccessTileType, mlir::RankedTensorType>(
              result.getType())) {
        for (mlir::OpOperand& use : result.getUses())
          enqueue(use.getOwner());
      }
    }
  }

  // Scan the block in order to find the boundary ops.
  mlir::Operation* splice_begin = nullptr;  // earliest construct_access_tile
  mlir::Operation* splice_end = nullptr;    // latest ktdp.store
  for (mlir::Operation& blk_op : *block) {
    if (!visited.count(&blk_op)) continue;
    if (mlir::isa<mlir::ktdp::ConstructAccessTilesOp>(&blk_op) &&
        !splice_begin)
      splice_begin = &blk_op;
    if (mlir::isa<mlir::ktdp::StoreOp>(&blk_op))
      splice_end = &blk_op;
  }
  return {splice_begin, splice_end};
}

/// Sub-step 2a: for one ConstructIndirectAccessTileOp, materialise all window
/// scf.for loops (all IAB subscript dimensions except the innermost per-entry
/// one), narrowing the IAB memref and updating the op on each iteration.
/// The access tile dimension to drop each iteration is derived from
/// variables_space_order.getResult(0) and is not assumed to be dim 0.
static mlir::LogicalResult materializeWindowLoops(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp op,
    int64_t iab_size) {
  mlir::MLIRContext* ctx = op.getContext();
  mlir::Location loc = op.getLoc();

  auto iab_type =
      mlir::cast<mlir::MemRefType>(op.getIndAddrBufMemref().getType());
  int64_t iab_rank = iab_type.getRank();

  // Validate: innermost IAB dimension must equal the hardware IAB size.
  int64_t innermost_dim = iab_type.getShape()[iab_rank - 1];
  if (innermost_dim != iab_size) {
    op.emitError() << "innermost IAB dimension (" << innermost_dim
                   << ") does not equal hardware IAB size (" << iab_size
                   << ")";
    return mlir::failure();
  }

  // W = number of window dimensions (all IAB dims except the innermost).
  int64_t W = iab_rank - 1;

  // If W == 0 the IAB is already 1-D; nothing to do for sub-step 2a.
  if (W == 0) return mlir::success();

  // The current op may be mutated (replaced) during the loop; keep a mutable
  // handle that always points to the live op.
  mlir::ktdp_lowering::ConstructIndirectAccessTileOp current_op = op;

  for (int64_t k = 0; k < W; ++k) {
    // ── Step 1: identify window variable w_k and trip count ──────────────
    // The window variable being absorbed is the one that drives the outermost
    // IAB subscript, i.e. ind_addr_buf_dim_positions[0].  That unified-space
    // position minus the number of captured variables gives the index within
    // the intermediate variable list (variables_space_set dim space).
    unsigned num_captured_k = current_op.getCapturedVariables().size();
    llvm::ArrayRef<int32_t> iab_positions_k =
        current_op.getIndAddrBufDimPositions();
    // iab_positions_k[0] is the unified-space position of the outermost IAB
    // subscript; subtract captured vars to get the intermediate variable index.
    unsigned absorbed_interm_idx =
        static_cast<unsigned>(iab_positions_k[0]) - num_captured_k;

    mlir::IntegerSet vars_set = current_op.getVariablesSpaceSet().getValue();
    auto trip_count = getTripCount(vars_set, absorbed_interm_idx);
    if (!trip_count) {
      current_op.emitError()
          << "could not extract constant trip count for window variable " << k
          << " (intermediate var index " << absorbed_interm_idx
          << ") from variables_space_set";
      return mlir::failure();
    }
    int64_t N = *trip_count;

    // ── Step 2: determine splice boundary ────────────────────────────────
    // Derive the boundary from the def-use graph of current_op:
    //   splice_begin_op — earliest ktdp.construct_access_tile reachable from
    //                     current_op (via upstream operands, IAB memref
    //                     use-list, and downstream users).
    //   splice_end_op   — latest ktdp.store reachable by the same walk.
    // For k > 0 current_op is already inside the k-1 loop body, so we splice
    // from the very first op in that body.
    mlir::Block* src_block = current_op->getBlock();
    mlir::Operation* splice_begin_op = nullptr;
    mlir::Operation* splice_end_op = nullptr;
    if (k == 0) {
      auto [begin, end] = findSpliceBoundary(current_op);
      splice_begin_op = begin;
      splice_end_op = end;
    } else {
      splice_begin_op = &src_block->front();
      // For k > 0 splice the entire body up to (and including) the last op
      // before the terminator.
      splice_end_op = &*std::prev(src_block->without_terminator().end());
    }
    if (!splice_begin_op || !splice_end_op) {
      current_op.emitError()
          << "could not determine splice boundary for window loop " << k;
      return mlir::failure();
    }

    // ── Step 3: emit scf.for BEFORE splice_begin_op ──────────────────────
    // Insert c0/cN/c1/for_op immediately before splice_begin_op so they land
    // in src_block.  Then splice [splice_begin_op, without-terminator-end)
    // into the for body.  for_op is before splice_begin_op so it won't be
    // included in the splice range.
    mlir::OpBuilder builder(splice_begin_op);
    mlir::Value c0 =
        mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult();
    mlir::Value cN =
        mlir::arith::ConstantIndexOp::create(builder, loc, N).getResult();
    mlir::Value c1 =
        mlir::arith::ConstantIndexOp::create(builder, loc, 1).getResult();
    auto for_op = mlir::scf::ForOp::create(builder, loc, c0, cN, c1);

    // Annotate with loop_type = parallel_loop.
    auto parallel_attr = mlir::ktdf::LoopTypeAttr::get(
        ctx, mlir::ktdf::LoopType::ParallelLoop);
    for_op->setAttr("loop_type", parallel_attr);

    mlir::Block* dst_block = for_op.getBody();

    // Splice [splice_begin_op, splice_end_op] (inclusive) from src_block into
    // the for body.  for_op was inserted before splice_begin_op so it is
    // not in the range.  For k > 0 splice_end_op points to the last op before
    // the terminator so std::next reaches the terminator — same as before.
    mlir::Block::iterator splice_begin = splice_begin_op->getIterator();
    mlir::Block::iterator splice_end = std::next(splice_end_op->getIterator());
    dst_block->getOperations().splice(dst_block->begin(),
                                      src_block->getOperations(),
                                      splice_begin, splice_end);

    // ── Step 4: replace IAB construct_memory_view with narrowed version ──
    // Drop the leading dimension from the IAB memref shape/strides/coord_set.
    auto cur_iab_mv = mlir::cast<mlir::ktdp_lowering::ConstructMemoryViewOp>(
        current_op.getIndAddrBufMemref().getDefiningOp());
    mlir::MemRefType cur_iab_memref_type =
        mlir::cast<mlir::MemRefType>(cur_iab_mv.getResult().getType());
    llvm::ArrayRef<int64_t> cur_iab_shape = cur_iab_memref_type.getShape();
    llvm::SmallVector<int64_t> new_iab_shape(cur_iab_shape.begin() + 1,
                                             cur_iab_shape.end());
    int64_t new_iab_rank = static_cast<int64_t>(new_iab_shape.size());

    llvm::SmallVector<int64_t> new_iab_strides(new_iab_rank);
    int64_t stride = 1;
    for (int64_t i = new_iab_rank - 1; i >= 0; --i) {
      new_iab_strides[i] = stride;
      stride *= new_iab_shape[i];
    }

    mlir::IntegerSet new_iab_coord_set = dropDimFromIntegerSet(
        cur_iab_mv.getCoordinateSet().getValue(), /*drop_dim=*/0);
    mlir::Attribute iab_memory_space = cur_iab_mv.getMemorySpace();
    auto new_iab_memref_type = mlir::MemRefType::get(
        new_iab_shape, mlir::IndexType::get(ctx),
        mlir::MemRefLayoutAttrInterface{}, iab_memory_space);

    // ── Steps 6 & 7: rebuild construct_indirect_access_tile ──────────────
    // The window variable being absorbed is the one that drives IAB dim 0
    // (the outermost IAB subscript).  Its index in the intermediate variable
    // list was already computed above as absorbed_interm_idx.  Its position in
    // the unified (captured..., intermediate...) dimension space is:
    unsigned num_captured = current_op.getCapturedVariables().size();
    unsigned absorbed_unified_dim = num_captured + absorbed_interm_idx;

    // Determine which access tile dimension corresponds to the absorbed window
    // variable.  variables_space_order.getResult(absorbed_interm_idx) is the
    // expression that maps that intermediate variable to its position in the
    // access tile; it must be a plain AffineDimExpr.
    mlir::AffineMap vars_order_map = current_op.getVariablesSpaceOrder();
    auto absorbed_dim_expr = mlir::dyn_cast<mlir::AffineDimExpr>(
        vars_order_map.getResult(absorbed_interm_idx));
    if (!absorbed_dim_expr) {
      current_op.emitError()
          << "variables_space_order result " << absorbed_interm_idx
          << " is not a plain dim expression; "
             "cannot determine access tile dimension to drop";
      return mlir::failure();
    }
    unsigned tile_dim_to_drop = absorbed_dim_expr.getPosition();

    // Narrow variables_space_set and variables_space_order: drop the dim
    // corresponding to the absorbed intermediate variable.
    mlir::IntegerSet new_vars_set = dropDimFromIntegerSet(
        current_op.getVariablesSpaceSet().getValue(), absorbed_interm_idx);
    mlir::AffineMap new_vars_order = dropDimFromAffineMap(
        current_op.getVariablesSpaceOrder(), absorbed_interm_idx);

    // New numIntermediateVariables: one fewer than before.
    unsigned new_num_interm =
        static_cast<unsigned>(current_op.getIntermediateVariables().size()) - 1;

    // Narrow ind_addr_buf_dim_positions: drop position 0 (the outermost IAB
    // dim, which is always absorbed first).  The remaining positions are
    // shifted down by one because absorbed_unified_dim is removed from the
    // unified dimension space.
    llvm::ArrayRef<int32_t> old_iab_positions =
        current_op.getIndAddrBufDimPositions();
    llvm::SmallVector<int32_t> new_iab_positions;
    // old_iab_positions[0] is the absorbed outermost IAB dim; skip it.
    for (size_t i = 1; i < old_iab_positions.size(); ++i) {
      int32_t pos = old_iab_positions[i];
      // Positions above absorbed_unified_dim shift down by 1.
      if (static_cast<unsigned>(pos) > absorbed_unified_dim) --pos;
      new_iab_positions.push_back(pos);
    }

    // Narrow per_dim_subscript_maps: drop the absorbed unified dim from each
    // map.  The absorbed dim index in the unified space is absorbed_unified_dim.
    llvm::SmallVector<mlir::Attribute> new_subscript_maps;
    for (mlir::Attribute attr : current_op.getPerDimSubscriptMaps()) {
      mlir::AffineMap old_map = mlir::cast<mlir::AffineMapAttr>(attr).getValue();
      // dropDimFromAffineMap removes result at drop_dim; here we need to drop
      // dim absorbed_unified_dim from the *input* space of the map (not a
      // result).  Build explicit dim replacements and compress.
      unsigned old_num_dims = old_map.getNumDims();
      llvm::SmallVector<mlir::AffineExpr> dim_repls(old_num_dims);
      for (unsigned d = 0; d < old_num_dims; ++d) {
        if (d < absorbed_unified_dim)
          dim_repls[d] = mlir::getAffineDimExpr(d, ctx);
        else if (d == absorbed_unified_dim)
          dim_repls[d] = mlir::getAffineConstantExpr(0, ctx);
        else
          dim_repls[d] = mlir::getAffineDimExpr(d - 1, ctx);
      }
      // Rebuild all results of the map with the shifted dims.
      llvm::SmallVector<mlir::AffineExpr> new_results;
      for (unsigned r = 0; r < old_map.getNumResults(); ++r) {
        new_results.push_back(old_map.getResult(r).replaceDimsAndSymbols(
            dim_repls, /*symReplacements=*/{}));
      }
      unsigned new_num_dims = old_num_dims - 1;
      new_subscript_maps.push_back(mlir::AffineMapAttr::get(mlir::AffineMap::get(
          new_num_dims, old_map.getNumSymbols(), new_results, ctx)));
    }

    // captured_variables are unchanged (the absorbed var was an intermediate
    // variable, not a captured one).
    llvm::SmallVector<mlir::Value> captured_vars(
        current_op.getCapturedVariables().begin(),
        current_op.getCapturedVariables().end());
    mlir::Value base = current_op.getBase();

    // Result type: drop the access tile dimension corresponding to the absorbed
    // window variable (tile_dim_to_drop), not necessarily the leading one.
    auto cur_result_type =
        mlir::cast<mlir::ktdp::AccessTileType>(current_op.getResult().getType());
    llvm::SmallVector<int64_t> new_result_shape =
        dropShapeDim(cur_result_type.getShape(), tile_dim_to_drop);
    auto new_result_type = mlir::ktdp::AccessTileType::get(
        new_result_shape, cur_result_type.getElementType());

    // Emit the narrowed IAB construct_memory_view before cur_iab_mv.
    mlir::OpBuilder body_builder(cur_iab_mv);
    auto new_iab_mv = mlir::ktdp_lowering::ConstructMemoryViewOp::create(
        body_builder, loc, new_iab_memref_type, cur_iab_mv.getOffset(),
        /*sizes=*/mlir::ValueRange{}, /*strides=*/mlir::ValueRange{},
        mlir::DenseI64ArrayAttr::get(ctx, new_iab_shape),
        mlir::DenseI64ArrayAttr::get(ctx, new_iab_strides), iab_memory_space,
        mlir::IntegerSetAttr::get(new_iab_coord_set));

    // ── Step 5: narrow the IAB fill chain ────────────────────────────────
    // cur_iab_mv may also be used by the IAB fill access tile
    // (ktdp.construct_access_tile %iab_mv[...]).  Narrow each such user to
    // use new_iab_mv, dropping the window dim from subscripts/set/order.
    // Also narrow the corresponding addr_buf access tile (which feeds the
    // store's data_tile) so the loaded tensor shape matches.
    //
    // All ops built here are collected into pre_narrowed and passed to
    // propagateNarrowing so it does not attempt to re-narrow them.
    llvm::SmallVector<mlir::Operation*> pre_narrowed;
    pre_narrowed.push_back(new_iab_mv.getOperation());

    llvm::SmallVector<mlir::ktdp::ConstructAccessTilesOp> iab_fill_ats;
    for (mlir::OpOperand& use : cur_iab_mv.getResult().getUses()) {
      if (use.getOwner() == current_op.getOperation()) continue;
      if (auto at = mlir::dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(
              use.getOwner()))
        iab_fill_ats.push_back(at);
    }

    for (mlir::ktdp::ConstructAccessTilesOp iab_at : iab_fill_ats) {
      // Narrow the IAB fill access tile: drop the window dim from set/order.
      // The IAB fill tile always covers the entire remaining row, so the
      // absorbed subscript is replaced with c0.
      // The IAB access tile's shape mirrors the IAB memref rank, which is
      // always reduced from the leading dimension, so drop dim 0 here.
      auto old_at_type =
          mlir::cast<mlir::ktdp::AccessTileType>(iab_at.getResult().getType());
      llvm::SmallVector<int64_t> new_at_shape =
          dropShapeDim(old_at_type.getShape(), 0);
      auto new_at_type = mlir::ktdp::AccessTileType::get(
          new_at_shape, old_at_type.getElementType());

      mlir::IntegerSet new_at_set =
          dropDimFromIntegerSet(iab_at.getAccessTileSet().getValue(), 0);
      mlir::AffineMap new_at_order =
          dropDimFromAffineMap(iab_at.getAccessTileOrder(), 0);

      unsigned new_rank = new_at_set.getNumDims();
      mlir::AffineMap new_base_map =
          mlir::AffineMap::getMultiDimIdentityMap(new_rank, ctx);

      // All subscripts become c0 (the fill covers the entire 1D row).
      mlir::OpBuilder idx_b(iab_at);
      mlir::Value c0_iab =
          mlir::arith::ConstantIndexOp::create(idx_b, loc, 0).getResult();
      llvm::SmallVector<mlir::Value> new_at_indices(new_rank, c0_iab);

      mlir::OpBuilder at_b(iab_at);
      auto new_iab_at = mlir::ktdp::ConstructAccessTilesOp::create(
          at_b, iab_at.getLoc(), new_at_type, new_iab_mv.getResult(),
          new_base_map, new_at_indices, new_at_set, new_at_order);
      pre_narrowed.push_back(new_iab_at.getOperation());

      // For each store that uses iab_at, also narrow the data-source chain:
      // store.data_tile → ktdp.load → ktdp.construct_access_tile (addr_buf_at).
      // The addr_buf access tile is narrowed from [c0, c0] to [for_iv, c0].
      for (mlir::OpOperand& at_use : iab_at.getResult().getUses()) {
        auto fill_store =
            mlir::dyn_cast<mlir::ktdp::StoreOp>(at_use.getOwner());
        if (!fill_store) continue;
        pre_narrowed.push_back(fill_store.getOperation());

        mlir::Value data_val = fill_store.getDataTile();
        auto fill_load = mlir::dyn_cast_if_present<mlir::ktdp::LoadOp>(
            data_val.getDefiningOp());
        if (!fill_load) continue;

        auto addr_buf_at =
            mlir::dyn_cast_if_present<mlir::ktdp::ConstructAccessTilesOp>(
                fill_load.getAccessTile().getDefiningOp());
        if (!addr_buf_at) continue;

        // Narrow addr_buf access tile: the absorbed window dimension is the
        // leading dimension of the addr_buf memref (always dim 0 in the
        // addr_buf tile), so we drop dim 0 from set/order/shape.
        // subscript[0] becomes the loop IV (selects the current window row).
        auto old_ab_type = mlir::cast<mlir::ktdp::AccessTileType>(
            addr_buf_at.getResult().getType());
        llvm::SmallVector<int64_t> new_ab_shape =
            dropShapeDim(old_ab_type.getShape(), 0);
        auto new_ab_type = mlir::ktdp::AccessTileType::get(
            new_ab_shape, old_ab_type.getElementType());

        mlir::IntegerSet new_ab_set = dropDimFromIntegerSet(
            addr_buf_at.getAccessTileSet().getValue(), 0);
        mlir::AffineMap new_ab_order =
            dropDimFromAffineMap(addr_buf_at.getAccessTileOrder(), 0);
        unsigned new_ab_rank = new_ab_set.getNumDims();
        mlir::AffineMap new_ab_base_map =
            mlir::AffineMap::getMultiDimIdentityMap(new_ab_rank, ctx);

        // Indices: [for_iv, c0, ...]
        mlir::OpBuilder ab_idx_b(addr_buf_at);
        mlir::Value c0_ab =
            mlir::arith::ConstantIndexOp::create(ab_idx_b, loc, 0).getResult();
        llvm::SmallVector<mlir::Value> new_ab_indices;
        new_ab_indices.push_back(for_op.getInductionVar());
        for (unsigned i = 1; i < new_ab_rank; ++i)
          new_ab_indices.push_back(c0_ab);

        mlir::OpBuilder ab_b(addr_buf_at);
        auto new_addr_buf_at = mlir::ktdp::ConstructAccessTilesOp::create(
            ab_b, addr_buf_at.getLoc(), new_ab_type, addr_buf_at.getBase(),
            new_ab_base_map, new_ab_indices, new_ab_set, new_ab_order);
        pre_narrowed.push_back(new_addr_buf_at.getOperation());

        // Rebuild the fill load with the narrowed addr_buf access tile.
        mlir::RankedTensorType old_fill_type = mlir::cast<mlir::RankedTensorType>(
            fill_load.getResult().getType());
        mlir::OpBuilder fl_b(fill_load);
        auto new_fill_load = mlir::ktdp::LoadOp::create(
            fl_b, fill_load.getLoc(), new_addr_buf_at.getResult(),
            old_fill_type.getElementType());
        pre_narrowed.push_back(new_fill_load.getOperation());

        // Update the store's data_tile and access_tile.
        fill_store.getDataTileMutable().assign(new_fill_load.getResult());

        fill_load.erase();
        addr_buf_at.erase();
      }

      // Swap iab_at → new_iab_at in every store that references it.
      iab_at.getResult().replaceAllUsesWith(new_iab_at.getResult());
      iab_at.erase();
    }

    // Build the replacement op at the same position as current_op.
    mlir::OpBuilder replace_builder(current_op);
    auto new_op = mlir::ktdp_lowering::ConstructIndirectAccessTileOp::create(
        replace_builder, loc, new_result_type, base, new_iab_mv.getResult(),
        mlir::DenseI32ArrayAttr::get(ctx, new_iab_positions),
        mlir::ArrayAttr::get(ctx, new_subscript_maps), captured_vars,
        new_num_interm, new_vars_order, new_vars_set);

    // RAUW old → new, erase old op, then erase old IAB mv (now use-free).
    current_op.getResult().replaceAllUsesWith(new_op.getResult());
    current_op.erase();
    cur_iab_mv.erase();
    current_op = new_op;

    // Propagate the narrowed shape bidirectionally through the entire
    // shaped-value graph.  This replaces the previous fixed-depth load_consumers
    // / store_consumers loops; the worklist algorithm handles any topology.
    // pre_narrowed contains the IAB mv and IAB fill chain ops already rebuilt
    // by steps 4/5 so the walk does not attempt to re-narrow them.
    if (mlir::failed(propagateNarrowing(new_op, tile_dim_to_drop,
                                        for_op.getInductionVar(), iab_rank,
                                        pre_narrowed, loc, ctx)))
      return mlir::failure();
  }

  return mlir::success();
}

struct IndirectAddrBufLegalizationPass
    : public impl::IndirectAddrBufLegalizationPassBase<
          IndirectAddrBufLegalizationPass> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Collect all construct_indirect_access_tile ops in the module.
    llvm::SmallVector<
        mlir::ktdp_lowering::ConstructIndirectAccessTileOp, 4>
        indirect_ops;
    module.walk(
        [&](mlir::ktdp_lowering::ConstructIndirectAccessTileOp op) {
          indirect_ops.push_back(op);
        });

    // No-op: return early when no indirect access tiles are present.
    if (indirect_ops.empty()) return;

    // Query the hardware IAB size from the architecture specification.
    auto declaration =
        mlir::ktdf_arch::findDeviceDeclarationFor(indirect_ops.front());
    if (!declaration) {
      indirect_ops.front()->emitError(
          "could not find device declaration for indirect address buffer "
          "legalization");
      signalPassFailure();
      return;
    }
    mlir::ktdf_arch::Device device(declaration);

    int64_t iab_size = -1;
    device.getBodyRegion().walk([&](mlir::ktdf_arch::Resource resource) {
      if (iab_size >= 0) return;
      const auto iab_feature =
          resource.getFeature<mlir::ktdf_arch::feature::IndirectAddressBuffer>();
      if (!iab_feature) return;
      if (const auto num_entries = iab_feature.getNumEntries())
        iab_size = *num_entries;
    });
    if (iab_size < 0) {
      declaration->emitError(
          "device has no indirect address buffer resource with num_entries");
      signalPassFailure();
      return;
    }

    for (auto op : indirect_ops) {
      if (mlir::failed(materializeWindowLoops(op, iab_size))) {
        signalPassFailure();
        return;
      }
    }

    // TODO(sub-step-2b): materialize inner scf.for over ind_addr_buf entries;
    // gate ind_addr_buf fill with scf.if (%i2 == 0); thread ind_addr_buf
    // memref view as scf.for iter-arg (sentinel initial value outside, real
    // view yielded from scf.if); remove absorbed intermediate variable from
    // the hidden region and variables_space_set.
  }
};

}  // namespace

std::unique_ptr<mlir::Pass>
scheduler::createIndirectAddrBufLegalizationPass() {
  return std::make_unique<IndirectAddrBufLegalizationPass>();
}
