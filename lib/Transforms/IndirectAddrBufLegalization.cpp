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

/// Sub-step 2a: for one ConstructIndirectAccessTileOp, materialise all window
/// scf.for loops (all IAB subscript dimensions except the innermost per-entry
/// one), narrowing the IAB memref and updating the op on each iteration.
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
    // The window variable being absorbed is always the first intermediate
    // variable (index 0 in the region block args).  After each iteration the
    // vars_set has one fewer dim, so the leading dim is always 0.
    mlir::IntegerSet vars_set = current_op.getVariablesSpaceSet().getValue();
    auto trip_count = getTripCount(vars_set, /*dim_idx=*/0);
    if (!trip_count) {
      current_op.emitError()
          << "could not extract constant trip count for window variable " << k
          << " from variables_space_set";
      return mlir::failure();
    }
    int64_t N = *trip_count;

    // ── Step 2: determine splice boundary ────────────────────────────────
    // Find the earliest op to move into the loop body.  For k == 0: the
    // first ktdp.construct_access_tile that precedes current_op (covers both
    // gather — no preceding source tile, falls back to current_op — and
    // scatter — source tile precedes IAB fill).  For k > 0: current_op is
    // already inside the k-1 loop body so we splice from the very first op.
    mlir::Block* src_block = current_op->getBlock();
    mlir::Operation* splice_begin_op = current_op.getOperation();
    if (k == 0) {
      for (mlir::Operation& blk_op : *src_block) {
        if (mlir::isa<mlir::ktdp::ConstructAccessTilesOp>(&blk_op)) {
          splice_begin_op = &blk_op;
          break;
        }
        if (&blk_op == current_op.getOperation()) break;
      }
    } else {
      splice_begin_op = &src_block->front();
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

    // Splice [splice_begin_op, end-before-terminator) from src_block into
    // the for body.  for_op was inserted before splice_begin_op so it is
    // not in this range.
    mlir::Block::iterator splice_begin = splice_begin_op->getIterator();
    auto splice_end = src_block->without_terminator().end();
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
    // The window variable being absorbed is the first intermediate variable
    // (region block arg 0).  Its position in the unified dimension space is
    // capturedVars.size() + 0.
    unsigned num_captured = current_op.getCapturedVariables().size();
    unsigned absorbed_unified_dim = num_captured;  // first intermediate dim

    // Narrow variables_space_set and variables_space_order: drop dim 0
    // (the leading intermediate dim, which is always the window variable).
    mlir::IntegerSet new_vars_set = dropDimFromIntegerSet(
        current_op.getVariablesSpaceSet().getValue(), /*drop_dim=*/0);
    mlir::AffineMap new_vars_order = dropDimFromAffineMap(
        current_op.getVariablesSpaceOrder(), /*drop_dim=*/0);

    // New numIntermediateVariables: one fewer than before.
    unsigned new_num_interm =
        static_cast<unsigned>(current_op.getIntermediateVariables().size()) - 1;

    // Narrow ind_addr_buf_dim_positions: drop position 0 (the absorbed window
    // dim).  The remaining positions are shifted down by one because
    // absorbed_unified_dim is removed from the unified dimension space.
    llvm::ArrayRef<int32_t> old_iab_positions =
        current_op.getIndAddrBufDimPositions();
    llvm::SmallVector<int32_t> new_iab_positions;
    // old_iab_positions[0] is the absorbed window dim; skip it.
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

    // Result type: drop leading IAB window dim from the access tile shape.
    auto cur_result_type =
        mlir::cast<mlir::ktdp::AccessTileType>(current_op.getResult().getType());
    llvm::ArrayRef<int64_t> cur_shape = cur_result_type.getShape();
    llvm::SmallVector<int64_t> new_result_shape(cur_shape.begin() + 1,
                                                cur_shape.end());
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
    llvm::SmallVector<mlir::ktdp::ConstructAccessTilesOp> iab_fill_ats;
    for (mlir::OpOperand& use : cur_iab_mv.getResult().getUses()) {
      if (use.getOwner() == current_op.getOperation()) continue;
      if (auto at = mlir::dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(
              use.getOwner()))
        iab_fill_ats.push_back(at);
    }

    for (mlir::ktdp::ConstructAccessTilesOp iab_at : iab_fill_ats) {
      // Narrow the IAB fill access tile: drop dim 0 from set/order, replace
      // subscript[0] with c0 (the fill always starts at offset 0 in the row).
      auto old_at_type =
          mlir::cast<mlir::ktdp::AccessTileType>(iab_at.getResult().getType());
      llvm::ArrayRef<int64_t> old_at_shape = old_at_type.getShape();
      llvm::SmallVector<int64_t> new_at_shape(old_at_shape.begin() + 1,
                                              old_at_shape.end());
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

      // For each store that uses iab_at, also narrow the data-source chain:
      // store.data_tile → ktdp.load → ktdp.construct_access_tile (addr_buf_at).
      // The addr_buf access tile is narrowed from [c0, c0] to [for_iv, c0].
      for (mlir::OpOperand& at_use : iab_at.getResult().getUses()) {
        auto fill_store =
            mlir::dyn_cast<mlir::ktdp::StoreOp>(at_use.getOwner());
        if (!fill_store) continue;

        mlir::Value data_val = fill_store.getDataTile();
        auto fill_load = mlir::dyn_cast_if_present<mlir::ktdp::LoadOp>(
            data_val.getDefiningOp());
        if (!fill_load) continue;

        auto addr_buf_at =
            mlir::dyn_cast_if_present<mlir::ktdp::ConstructAccessTilesOp>(
                fill_load.getAccessTile().getDefiningOp());
        if (!addr_buf_at) continue;

        // Narrow addr_buf access tile: drop dim 0 from set/order/shape;
        // subscript[0] becomes the loop IV (selects the current window row).
        auto old_ab_type = mlir::cast<mlir::ktdp::AccessTileType>(
            addr_buf_at.getResult().getType());
        llvm::ArrayRef<int64_t> old_ab_shape = old_ab_type.getShape();
        llvm::SmallVector<int64_t> new_ab_shape(old_ab_shape.begin() + 1,
                                                old_ab_shape.end());
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

        // Rebuild the fill load with the narrowed addr_buf access tile.
        mlir::RankedTensorType old_fill_type = mlir::cast<mlir::RankedTensorType>(
            fill_load.getResult().getType());
        mlir::OpBuilder fl_b(fill_load);
        auto new_fill_load = mlir::ktdp::LoadOp::create(
            fl_b, fill_load.getLoc(), new_addr_buf_at.getResult(),
            old_fill_type.getElementType());

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

    // Collect consumers of the indirect access tile before RAUW.
    llvm::SmallVector<mlir::ktdp::LoadOp> load_consumers;
    llvm::SmallVector<mlir::ktdp::StoreOp> store_consumers;
    for (mlir::OpOperand& use : current_op.getResult().getUses()) {
      if (auto load = mlir::dyn_cast<mlir::ktdp::LoadOp>(use.getOwner()))
        load_consumers.push_back(load);
      else if (auto store =
                   mlir::dyn_cast<mlir::ktdp::StoreOp>(use.getOwner()))
        store_consumers.push_back(store);
    }

    // RAUW old → new, erase old op, then erase old IAB mv (now use-free).
    current_op.getResult().replaceAllUsesWith(new_op.getResult());
    current_op.erase();
    cur_iab_mv.erase();
    current_op = new_op;

    // Rebuild each ktdp.load — its result tensor type must match the narrowed
    // access tile shape.  After rebuilding, propagate the narrowed tensor shape
    // through the downstream compute chain:
    //   new_load → linalg.generic (narrow ins/outs/maps) → ktdp.store
    //           → ktdp.construct_access_tile (narrow shape & subscript[0]).
    for (mlir::ktdp::LoadOp old_load : load_consumers) {
      mlir::RankedTensorType old_tensor_type =
          mlir::cast<mlir::RankedTensorType>(old_load.getResult().getType());
      mlir::OpBuilder load_builder(old_load);
      auto new_load = mlir::ktdp::LoadOp::create(
          load_builder, old_load.getLoc(), new_op.getResult(),
          old_tensor_type.getElementType());
      old_load.getResult().replaceAllUsesWith(new_load.getResult());
      old_load.erase();

      // Propagate the narrowed tensor shape to linalg.generic and its output
      // access tile.  We trace: new_load.result → linalg.generic.ins[0]
      //   → linalg.generic.result → ktdp.store.data_tile
      //     → (store.tile =) ktdp.construct_access_tile → narrow it.
      for (mlir::OpOperand& use : new_load.getResult().getUses()) {
        auto generic =
            mlir::dyn_cast<mlir::linalg::GenericOp>(use.getOwner());
        if (!generic) continue;

        // Drop dim 0 from every indexing map of the generic.
        llvm::SmallVector<mlir::AffineMap> new_maps;
        for (mlir::AffineMap m : generic.getIndexingMapsArray())
          new_maps.push_back(dropDimFromAffineMap(m, 0));
        generic.setIndexingMapsAttr(mlir::ArrayAttr::get(
            ctx,
            llvm::to_vector(llvm::map_range(new_maps, [](mlir::AffineMap m) {
              return mlir::cast<mlir::Attribute>(mlir::AffineMapAttr::get(m));
            }))));

        // Drop the outermost iterator_type from the generic.
        auto old_iters = generic.getIteratorTypesArray();
        llvm::SmallVector<mlir::Attribute> new_iters;
        for (size_t i = 1; i < old_iters.size(); ++i)
          new_iters.push_back(
              mlir::linalg::IteratorTypeAttr::get(ctx, old_iters[i]));
        generic.setIteratorTypesAttr(mlir::ArrayAttr::get(ctx, new_iters));

        // Narrow the output tensor (outs operand): replace tensor.empty with
        // a narrowed one.
        for (mlir::OpOperand& out_use : generic.getOutputsMutable()) {
          auto empty = mlir::dyn_cast_if_present<mlir::tensor::EmptyOp>(
              out_use.get().getDefiningOp());
          if (!empty) continue;
          auto old_empty_type =
              mlir::cast<mlir::RankedTensorType>(empty.getResult().getType());
          llvm::ArrayRef<int64_t> old_empty_shape = old_empty_type.getShape();
          llvm::SmallVector<int64_t> new_empty_shape(
              old_empty_shape.begin() + 1, old_empty_shape.end());
          auto new_empty_type = mlir::RankedTensorType::get(
              new_empty_shape, old_empty_type.getElementType());
          mlir::OpBuilder eb(empty);
          auto new_empty = mlir::tensor::EmptyOp::create(eb, empty.getLoc(),
                                                         new_empty_type, {});
          out_use.set(new_empty.getResult());
          // Old empty is now unused if it had no other uses.
          if (empty.getResult().use_empty()) empty.erase();
        }

        // Update the generic result type to match the narrowed output.
        auto old_res_type =
            mlir::cast<mlir::RankedTensorType>(generic.getResult(0).getType());
        llvm::ArrayRef<int64_t> old_res_shape = old_res_type.getShape();
        llvm::SmallVector<int64_t> new_res_shape(old_res_shape.begin() + 1,
                                                 old_res_shape.end());
        generic.getResult(0).setType(mlir::RankedTensorType::get(
            new_res_shape, old_res_type.getElementType()));

        // Trace generic.result → ktdp.store → ktdp.construct_access_tile
        // and narrow the output access tile.
        for (mlir::OpOperand& res_use : generic.getResult(0).getUses()) {
          auto out_store =
              mlir::dyn_cast<mlir::ktdp::StoreOp>(res_use.getOwner());
          if (!out_store) continue;

          auto out_at =
              mlir::dyn_cast_if_present<mlir::ktdp::ConstructAccessTilesOp>(
                  out_store.getAccessTile().getDefiningOp());
          if (!out_at) continue;

          auto old_out_type = mlir::cast<mlir::ktdp::AccessTileType>(
              out_at.getResult().getType());
          llvm::ArrayRef<int64_t> old_out_shape = old_out_type.getShape();
          llvm::SmallVector<int64_t> new_out_shape(old_out_shape.begin() + 1,
                                                   old_out_shape.end());
          auto new_out_type = mlir::ktdp::AccessTileType::get(
              new_out_shape, old_out_type.getElementType());

          mlir::IntegerSet new_out_set = dropDimFromIntegerSet(
              out_at.getAccessTileSet().getValue(), 0);
          mlir::AffineMap new_out_order =
              dropDimFromAffineMap(out_at.getAccessTileOrder(), 0);
          unsigned new_out_rank = new_out_set.getNumDims();
          mlir::AffineMap new_out_base_map =
              mlir::AffineMap::getMultiDimIdentityMap(new_out_rank, ctx);

          // Indices: [for_iv, c0, c0, ...]
          mlir::OpBuilder oi_b(out_at);
          mlir::Value c0_out =
              mlir::arith::ConstantIndexOp::create(oi_b, loc, 0).getResult();
          llvm::SmallVector<mlir::Value> new_out_indices;
          new_out_indices.push_back(for_op.getInductionVar());
          for (unsigned i = 1; i < new_out_rank; ++i)
            new_out_indices.push_back(c0_out);

          mlir::OpBuilder oa_b(out_at);
          auto new_out_at = mlir::ktdp::ConstructAccessTilesOp::create(
              oa_b, out_at.getLoc(), new_out_type, out_at.getBase(),
              new_out_base_map, new_out_indices, new_out_set, new_out_order);

          out_store.getAccessTileMutable().assign(new_out_at.getResult());
          out_at.erase();
        }
      }
    }

    // For scatter: the ktdp.store's access_tile (indirect tile) now has a
    // narrowed shape; narrow the store's data_tile to match by tracing through
    // linalg.generic → tensor.empty + source load → ktdp.construct_access_tile.
    for (mlir::ktdp::StoreOp store_op : store_consumers) {
      // The data_tile is the result of a linalg.generic that computes over
      // the source data.  Trace: store.data_tile → linalg.generic → ins[0]
      // → ktdp.load → ktdp.construct_access_tile (source tile).
      mlir::Value data_val = store_op.getDataTile();
      auto generic = mlir::dyn_cast_if_present<mlir::linalg::GenericOp>(
          data_val.getDefiningOp());
      if (generic) {
        // Narrow linalg.generic indexing maps, iterator_types, outs tensor,
        // result type, and source access tile.
        llvm::SmallVector<mlir::AffineMap> new_maps;
        for (mlir::AffineMap m : generic.getIndexingMapsArray())
          new_maps.push_back(dropDimFromAffineMap(m, 0));
        generic.setIndexingMapsAttr(mlir::ArrayAttr::get(
            ctx,
            llvm::to_vector(llvm::map_range(new_maps, [](mlir::AffineMap m) {
              return mlir::cast<mlir::Attribute>(mlir::AffineMapAttr::get(m));
            }))));

        auto old_iters = generic.getIteratorTypesArray();
        llvm::SmallVector<mlir::Attribute> new_iters;
        for (size_t i = 1; i < old_iters.size(); ++i)
          new_iters.push_back(
              mlir::linalg::IteratorTypeAttr::get(ctx, old_iters[i]));
        generic.setIteratorTypesAttr(mlir::ArrayAttr::get(ctx, new_iters));

        for (mlir::OpOperand& out_use : generic.getOutputsMutable()) {
          auto empty = mlir::dyn_cast_if_present<mlir::tensor::EmptyOp>(
              out_use.get().getDefiningOp());
          if (!empty) continue;
          auto old_et =
              mlir::cast<mlir::RankedTensorType>(empty.getResult().getType());
          llvm::ArrayRef<int64_t> old_es = old_et.getShape();
          llvm::SmallVector<int64_t> new_es(old_es.begin() + 1, old_es.end());
          auto new_et = mlir::RankedTensorType::get(new_es, old_et.getElementType());
          mlir::OpBuilder eb(empty);
          auto new_empty =
              mlir::tensor::EmptyOp::create(eb, empty.getLoc(), new_et, {});
          out_use.set(new_empty.getResult());
          if (empty.getResult().use_empty()) empty.erase();
        }

        auto old_rt =
            mlir::cast<mlir::RankedTensorType>(generic.getResult(0).getType());
        llvm::ArrayRef<int64_t> old_rs = old_rt.getShape();
        llvm::SmallVector<int64_t> new_rs(old_rs.begin() + 1, old_rs.end());
        generic.getResult(0).setType(
            mlir::RankedTensorType::get(new_rs, old_rt.getElementType()));

        // Narrow the source access tile (ins operand).
        for (mlir::OpOperand& ins_use : generic.getInputsMutable()) {
          auto src_load = mlir::dyn_cast_if_present<mlir::ktdp::LoadOp>(
              ins_use.get().getDefiningOp());
          if (!src_load) continue;
          auto src_at =
              mlir::dyn_cast_if_present<mlir::ktdp::ConstructAccessTilesOp>(
                  src_load.getAccessTile().getDefiningOp());
          if (!src_at) continue;

          auto old_at_type = mlir::cast<mlir::ktdp::AccessTileType>(
              src_at.getResult().getType());
          llvm::ArrayRef<int64_t> old_at_shape = old_at_type.getShape();
          llvm::SmallVector<int64_t> new_at_shape(old_at_shape.begin() + 1,
                                                  old_at_shape.end());
          auto new_at_type = mlir::ktdp::AccessTileType::get(
              new_at_shape, old_at_type.getElementType());

          mlir::IntegerSet new_at_set = dropDimFromIntegerSet(
              src_at.getAccessTileSet().getValue(), 0);
          mlir::AffineMap new_at_order =
              dropDimFromAffineMap(src_at.getAccessTileOrder(), 0);
          unsigned new_at_rank = new_at_set.getNumDims();
          mlir::AffineMap new_at_base_map =
              mlir::AffineMap::getMultiDimIdentityMap(new_at_rank, ctx);

          mlir::OpBuilder si_b(src_at);
          mlir::Value c0_src =
              mlir::arith::ConstantIndexOp::create(si_b, loc, 0).getResult();
          llvm::SmallVector<mlir::Value> new_at_indices;
          new_at_indices.push_back(for_op.getInductionVar());
          for (unsigned i = 1; i < new_at_rank; ++i)
            new_at_indices.push_back(c0_src);

          mlir::OpBuilder sa_b(src_at);
          auto new_src_at = mlir::ktdp::ConstructAccessTilesOp::create(
              sa_b, src_at.getLoc(), new_at_type, src_at.getBase(),
              new_at_base_map, new_at_indices, new_at_set, new_at_order);

          auto old_ld_type = mlir::cast<mlir::RankedTensorType>(
              src_load.getResult().getType());
          mlir::OpBuilder sl_b(src_load);
          auto new_src_load = mlir::ktdp::LoadOp::create(
              sl_b, src_load.getLoc(), new_src_at.getResult(),
              old_ld_type.getElementType());
          ins_use.set(new_src_load.getResult());

          src_load.erase();
          src_at.erase();
        }
      }
    }
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
