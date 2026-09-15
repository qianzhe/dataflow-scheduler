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

#include "dataflow-scheduler/Transforms/Utils/IndirectAccessTileNarrowing.h"

#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace scheduler {

mlir::IntegerSet dropDimFromIntegerSet(mlir::IntegerSet set,
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

mlir::IntegerSet pinDimInIntegerSet(mlir::IntegerSet set, unsigned pin_dim) {
  mlir::MLIRContext* ctx = set.getContext();
  mlir::AffineExpr dk = mlir::getAffineDimExpr(pin_dim, ctx);

  llvm::SmallVector<mlir::AffineExpr> new_constraints;
  llvm::SmallVector<bool> new_eq_flags;
  bool replaced_ub = false;
  for (unsigned i = 0; i < set.getNumConstraints(); ++i) {
    mlir::AffineExpr c = set.getConstraint(i);
    bool is_eq = set.isEq(i);
    // The upper bound on pin_dim is the first inequality involving pin_dim
    // that is not the bare lower-bound expression (d_k >= 0).
    if (!is_eq && !replaced_ub && c.isFunctionOfDim(pin_dim) && c != dk) {
      new_constraints.push_back(mlir::getAffineConstantExpr(0, ctx) - dk);
      new_eq_flags.push_back(false);
      replaced_ub = true;
      continue;
    }
    new_constraints.push_back(c);
    new_eq_flags.push_back(is_eq);
  }

  return mlir::IntegerSet::get(set.getNumDims(), set.getNumSymbols(),
                               new_constraints, new_eq_flags);
}

mlir::AffineMap dropDimFromAffineMap(mlir::AffineMap map, unsigned drop_dim) {
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

llvm::SmallVector<int64_t> dropShapeDim(llvm::ArrayRef<int64_t> shape,
                                        unsigned drop_dim) {
  llvm::SmallVector<int64_t> result;
  result.reserve(shape.size() - 1);
  for (unsigned i = 0; i < shape.size(); ++i) {
    if (i != drop_dim) result.push_back(shape[i]);
  }
  return result;
}

std::optional<int64_t> getTripCount(mlir::IntegerSet set, unsigned dim_idx) {
  for (unsigned i = 0; i < set.getNumConstraints(); ++i) {
    if (set.isEq(i)) continue;
    mlir::AffineExpr expr = set.getConstraint(i);
    // We expect a constraint that is purely a function of dim_idx and has the
    // form `-d + C >= 0` (i.e. d <= C). Concretely: constant + (-1)*d >= 0.
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

mlir::SmallVector<mlir::ReassociationIndices> makeLeadingFoldReassociation(
    unsigned result_rank, unsigned folded_leading_dims) {
  mlir::SmallVector<mlir::ReassociationIndices> reassoc;
  mlir::ReassociationIndices first_group;
  for (unsigned d = 0; d <= folded_leading_dims; ++d)
    first_group.push_back(static_cast<int64_t>(d));
  reassoc.push_back(first_group);
  for (unsigned d = folded_leading_dims + 1; d < result_rank; ++d)
    reassoc.push_back({static_cast<int64_t>(d)});
  return reassoc;
}

mlir::ktdp::ConstructAccessTilesOp rebuildAccessTilePinned(
    mlir::ktdp::ConstructAccessTilesOp at, unsigned pin_dim,
    llvm::ArrayRef<mlir::Value> window_ivs, mlir::Location loc,
    mlir::MLIRContext* ctx) {
  assert(pin_dim + 1 == window_ivs.size() &&
         "pin_dim must index the innermost window loop IV");
  mlir::IntegerSet old_set = at.getAccessTileSet().getValue();
  unsigned num_dims = old_set.getNumDims();
  assert(pin_dim < num_dims && "pin_dim out of range for access_tile_set");

  mlir::IntegerSet new_set = pinDimInIntegerSet(old_set, pin_dim);

  // access_tile_order and base_map keep the full base-memref rank.
  mlir::AffineMap new_order = at.getAccessTileOrder();
  mlir::AffineMap new_base_map =
      mlir::AffineMap::getMultiDimIdentityMap(num_dims, ctx);

  // Result shape: dim pin_dim becomes 1, all others unchanged.
  auto old_type =
      mlir::cast<mlir::ktdp::AccessTileType>(at.getResult().getType());
  llvm::SmallVector<int64_t> new_shape(old_type.getShape().begin(),
                                      old_type.getShape().end());
  new_shape[pin_dim] = 1;
  auto new_type =
      mlir::ktdp::AccessTileType::get(new_shape, old_type.getElementType());

  // Subscripts: one IV per pinned dim 0..pin_dim, %c0 for the free dims above.
  mlir::OpBuilder builder(at);
  mlir::Value c0 =
      mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  llvm::SmallVector<mlir::Value> new_indices(num_dims, c0);
  for (unsigned d = 0; d <= pin_dim; ++d) new_indices[d] = window_ivs[d];

  return mlir::ktdp::ConstructAccessTilesOp::create(
      builder, at.getLoc(), new_type, at.getBase(), new_base_map, new_indices,
      new_set, new_order);
}

std::pair<mlir::Operation*, mlir::Operation*> findSpliceBoundary(
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

std::pair<mlir::Operation*, mlir::Operation*> computeSpliceBoundary(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp current_op,
    bool is_first_loop) {
  if (is_first_loop) return findSpliceBoundary(current_op);

  mlir::Block* src_block = current_op->getBlock();
  mlir::Operation* splice_begin_op = &src_block->front();
  mlir::Operation* splice_end_op =
      &*std::prev(src_block->without_terminator().end());
  return {splice_begin_op, splice_end_op};
}

std::optional<DeferredExpand> pinOutputDescriptorAT(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp current_op,
    mlir::Block* scope_block, unsigned pin_dim,
    llvm::ArrayRef<mlir::Value> window_ivs,
    llvm::SmallVectorImpl<mlir::Operation*>& pre_narrowed, mlir::Location loc,
    mlir::MLIRContext* ctx) {
  std::optional<DeferredExpand> result;

  llvm::SmallVector<mlir::Value> worklist{current_op.getResult()};
  llvm::DenseSet<mlir::Value> visited{current_op.getResult()};

  while (!worklist.empty()) {
    mlir::Value cur_val = worklist.pop_back_val();
    for (mlir::Operation* user : cur_val.getUsers()) {
      if (user->getBlock() != scope_block) continue;

      if (auto out_store = mlir::dyn_cast<mlir::ktdp::StoreOp>(user)) {
        auto out_at =
            mlir::dyn_cast_if_present<mlir::ktdp::ConstructAccessTilesOp>(
                out_store.getAccessTile().getDefiningOp());
        if (!out_at) continue;

        if (auto prior_expand =
                mlir::dyn_cast_if_present<mlir::tensor::ExpandShapeOp>(
                    out_store.getDataTile().getDefiningOp())) {
          out_store.getDataTileMutable().assign(prior_expand.getSrc());
          prior_expand.erase();
        }

        auto new_out_at =
            rebuildAccessTilePinned(out_at, pin_dim, window_ivs, loc, ctx);
        pre_narrowed.push_back(new_out_at.getOperation());
        llvm::ArrayRef<int64_t> new_out_shape =
            mlir::cast<mlir::ktdp::AccessTileType>(
                new_out_at.getResult().getType())
                .getShape();

        auto out_reassoc = makeLeadingFoldReassociation(
            static_cast<unsigned>(new_out_shape.size()),
            /*folded_leading_dims=*/pin_dim + 1);

        mlir::Type out_elem_type =
            mlir::cast<mlir::RankedTensorType>(out_store.getDataTile().getType())
                .getElementType();

        DeferredExpand deferred;
        deferred.src_val = out_store.getDataTile();
        deferred.store = out_store;
        deferred.reassoc = out_reassoc;
        deferred.pinned_type =
            mlir::RankedTensorType::get(new_out_shape, out_elem_type);
        result = deferred;

        pre_narrowed.push_back(out_store.getOperation());

        out_at.getResult().replaceAllUsesWith(new_out_at.getResult());
        out_at.erase();
        continue;
      }

      for (mlir::Value res : user->getResults()) {
        if (mlir::isa<mlir::ktdp::AccessTileType, mlir::RankedTensorType>(
                res.getType()) &&
            visited.insert(res).second)
          worklist.push_back(res);
      }
    }
  }
  return result;
}

void pinSourceDescriptorAT(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp current_op,
    mlir::Block* scope_block, unsigned pin_dim,
    llvm::ArrayRef<mlir::Value> window_ivs,
    llvm::SmallVectorImpl<mlir::Operation*>& pre_narrowed, mlir::Location loc,
    mlir::MLIRContext* ctx) {
  for (mlir::Operation* user : current_op.getResult().getUsers()) {
    auto store = mlir::dyn_cast<mlir::ktdp::StoreOp>(user);
    if (!store || store->getBlock() != scope_block) continue;

    llvm::SmallVector<mlir::Value> worklist{store.getDataTile()};
    llvm::DenseSet<mlir::Value> visited{store.getDataTile()};

    while (!worklist.empty()) {
      mlir::Value cur_val = worklist.pop_back_val();
      mlir::Operation* def = cur_val.getDefiningOp();
      if (!def || def->getBlock() != scope_block) continue;

      if (auto src_load = mlir::dyn_cast<mlir::ktdp::LoadOp>(def)) {
        auto src_at =
            mlir::dyn_cast_if_present<mlir::ktdp::ConstructAccessTilesOp>(
                src_load.getAccessTile().getDefiningOp());
        if (src_at) {
          auto new_src_at =
              rebuildAccessTilePinned(src_at, pin_dim, window_ivs, loc, ctx);
          pre_narrowed.push_back(new_src_at.getOperation());
          llvm::ArrayRef<int64_t> new_src_shape =
              mlir::cast<mlir::ktdp::AccessTileType>(
                  new_src_at.getResult().getType())
                  .getShape();

          mlir::Type src_elem_type =
              mlir::cast<mlir::RankedTensorType>(src_load.getResult().getType())
                  .getElementType();

          mlir::OpBuilder sl_b(src_load);
          auto new_src_load = mlir::ktdp::LoadOp::create(
              sl_b, src_load.getLoc(), new_src_at.getResult(), src_elem_type);
          pre_narrowed.push_back(new_src_load.getOperation());

          unsigned src_pinned_dims = pin_dim + 1;
          auto src_reassoc = makeLeadingFoldReassociation(
              static_cast<unsigned>(new_src_shape.size()), src_pinned_dims);
          auto src_collapsed_type = mlir::RankedTensorType::get(
              new_src_shape.drop_front(src_pinned_dims), src_elem_type);

          mlir::OpBuilder cs_src_b(src_load->getNextNode());
          auto new_src_collapse = mlir::tensor::CollapseShapeOp::create(
              cs_src_b, src_load.getLoc(), src_collapsed_type,
              new_src_load.getResult(), src_reassoc);
          pre_narrowed.push_back(new_src_collapse.getOperation());

          for (mlir::OpOperand& load_use : llvm::make_early_inc_range(
                   src_load.getResult().getUses())) {
            mlir::Operation* consumer = load_use.getOwner();
            if (auto cs =
                    mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(consumer)) {
              cs.getResult().replaceAllUsesWith(new_src_collapse.getResult());
              cs.erase();
            } else {
              load_use.set(new_src_collapse.getResult());
            }
          }

          src_load.erase();
          src_at.getResult().replaceAllUsesWith(new_src_at.getResult());
          src_at.erase();
        }
        continue;
      }

      for (mlir::Value operand : def->getOperands()) {
        if (mlir::isa<mlir::ktdp::AccessTileType, mlir::RankedTensorType>(
                operand.getType()) &&
            visited.insert(operand).second)
          worklist.push_back(operand);
      }
    }
  }
}

namespace {

/// Return true if `type` carries dimension info that the narrowing propagation
/// must follow. MemRefs are excluded: base descriptor memrefs (desc_1, desc_2,
/// addr_buf) are out-of-scope, and the IAB memref is handled by the caller's
/// pin-not-drop / descriptor-AT steps before propagateNarrowing is called.
bool isNarrowable(mlir::Type type) {
  return mlir::isa<mlir::ktdp::AccessTileType, mlir::RankedTensorType>(type);
}

enum class PropDir { DOWN, UP };

/// Narrowing dispatch: mutate `op` in-place so that all shaped operands and
/// results reflect the dimension drop already applied to upstream values via
/// RAUW. Must NOT propagate further — that is the caller's job.
///
/// If the op was rebuilt (RAUW + erase), `*rebuilt_out` is set to the new op.
/// Otherwise `*rebuilt_out` is left unchanged (caller should initialize to
/// nullptr). Pass nullptr if the caller does not need this information.
///
/// Returns failure() if `op` is unknown or internally inconsistent.
mlir::LogicalResult narrowOp(
    mlir::Operation* op, unsigned tile_dim_to_drop,
    llvm::ArrayRef<mlir::Value> window_ivs, int64_t iab_rank,
    mlir::Location loc, mlir::MLIRContext* ctx,
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
  // Narrow shape, access_tile_set, access_tile_order, and indices (drop-based).
  // The loop IV (window_ivs.back()) is placed at tile_dim_to_drop; prior IVs
  // (window_ivs[0..k-2]) are placed at dims 0..k-2 for W>1 correctness;
  // c0 fills the remaining slots.
  if (auto at = mlir::dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(op)) {
    auto old_type =
        mlir::cast<mlir::ktdp::AccessTileType>(at.getResult().getType());
    auto new_type = mlir::ktdp::AccessTileType::get(
        dropShapeDim(old_type.getShape(), tile_dim_to_drop),
        old_type.getElementType());
    mlir::IntegerSet new_set =
        dropDimFromIntegerSet(at.getAccessTileSet().getValue(), tile_dim_to_drop);
    mlir::AffineMap new_order =
        dropDimFromAffineMap(at.getAccessTileOrder(), tile_dim_to_drop);
    unsigned new_rank = new_set.getNumDims();
    mlir::AffineMap new_base_map =
        mlir::AffineMap::getMultiDimIdentityMap(new_rank, ctx);

    mlir::OpBuilder ib(at);
    mlir::Value c0 =
        mlir::arith::ConstantIndexOp::create(ib, loc, 0).getResult();
    llvm::SmallVector<mlir::Value> new_indices(new_rank, c0);
    // Place prior window IVs at dims 0..k-2 (they have already been absorbed
    // into the outer loops and represent the current iteration coordinates).
    for (unsigned d = 0; d + 1 < window_ivs.size() && d < new_rank; ++d)
      new_indices[d] = window_ivs[d];
    // Place the current IV at tile_dim_to_drop (after the drop it shifts left
    // but new_indices is sized for new_rank so use the shifted position).
    if (tile_dim_to_drop < new_rank) new_indices[tile_dim_to_drop] = window_ivs.back();

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
  // Narrow indexing_maps, iterator_types, and result type (tensor space: drop).
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
  // Sink: no results, no shape-encoding attributes. Both operands are already
  // updated by RAUW from upstream rebuilds.
  if (mlir::isa<mlir::ktdp::StoreOp>(op)) return mlir::success();

  // ── Unknown ───────────────────────────────────────────────────────────────
  return op->emitError("unknown op in narrowing propagation");
}

}  // namespace

mlir::LogicalResult propagateNarrowing(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp new_op,
    unsigned tile_dim_to_drop, llvm::ArrayRef<mlir::Value> window_ivs,
    int64_t iab_rank, llvm::ArrayRef<mlir::Operation*> pre_narrowed,
    mlir::Location loc, mlir::MLIRContext* ctx) {
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
  // Pre-mark ops already rebuilt by the caller (IAB mv, IAB fill AT, addr_buf
  // AT, fill load; or descriptor-AT pin-not-drop) so the walk does not
  // attempt to narrow or propagate through them again.
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
          if (mlir::failed(narrowOp(user, tile_dim_to_drop, window_ivs,
                                    iab_rank, loc, ctx)))
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
        // narrowed. In the UP direction the AT may not be narrowed yet, so
        // we skip narrowOp here and let the generic propagation push UP from
        // the AT operand. narrowOp(load) will be called correctly via DOWN
        // once the AT is narrowed and its result is pushed DOWN.
        if (!mlir::isa<mlir::ktdp::LoadOp>(def_op)) {
          mlir::Operation* rebuilt = nullptr;
          if (mlir::failed(narrowOp(def_op, tile_dim_to_drop, window_ivs,
                                    iab_rank, loc, ctx, &rebuilt)))
            return mlir::failure();
          narrowed_ops.insert(def_op);
          if (rebuilt) {
            // Mark rebuilt op as already narrowed.
            narrowed_ops.insert(rebuilt);
            // Push rebuilt op's narrowable results DOWN so their users (e.g. a
            // ktdp.load using a just-narrowed construct_access_tile) can be
            // narrowed in turn. Necessary for indirect store where the source AT is
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

}  // namespace scheduler
