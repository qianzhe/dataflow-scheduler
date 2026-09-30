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
// IndirectAddrBufLegalization: legalizes construct_indirect_access_tile ops
// against the hardware indirect address buffer (IAB), which can only hold
// `iab_size` scalar entries at a time. This pass materializes one or nested
// scf.for loops that iterate over windows of `iab_size` entries when the
// capacity of the address tensor exceeds `iab_size`, followed by a single
// per-entry scf.for loop that walks one IAB entry at a time.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "dataflow-scheduler/Dialect/KTDF/KTDFAttributes.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFEnums.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Transforms/Utils/IndirectAccessTileNarrowing.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "indirect-addr-buf-legalization"
#define DEBUG_TYPE PASS_NAME

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_INDIRECTADDRBUFLEGALIZATIONPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

/// Identifies the intermediate variable absorbed by the next window/entry
/// loop: the one driving the outermost remaining ind_addr_buf subscript
/// (`ind_addr_buf_dim_positions[0]`).
struct AbsorbedVarInfo {
  unsigned num_captured;         // op.getCapturedVariables().size()
  unsigned absorbed_interm_idx;  // index into intermediate_variables /
                                 // variables_space_set
  unsigned tile_dim_to_drop;     // access-tile dim the absorbed var maps to
  int64_t trip_count;            // constant trip count (N) for that dim
};

/// Shared by the window loop materialization (per window dim) and the entry
/// loop materialization (the final per-entry dim): read off the absorbed
/// variable's index, its constant trip count (from variables_space_set), and
/// the access-tile dimension it maps to (from variables_space_order, needed
/// for downstream narrowing).
static mlir::FailureOr<AbsorbedVarInfo> getAbsorbedVarInfo(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp op) {
  unsigned num_captured = op.getCapturedVariables().size();
  llvm::ArrayRef<int32_t> iab_positions = op.getIndAddrBufDimPositions();
  unsigned absorbed_interm_idx =
      static_cast<unsigned>(iab_positions[0]) - num_captured;

  mlir::IntegerSet vars_set = op.getVariablesSpaceSet().getValue();
  auto trip_count = getTripCount(vars_set, absorbed_interm_idx);
  if (!trip_count) {
    op.emitError() << "could not extract constant trip count for absorbed "
                      "variable (intermediate var index "
                   << absorbed_interm_idx << ") from variables_space_set";
    return mlir::failure();
  }

  mlir::AffineMap vars_order_map = op.getVariablesSpaceOrder();
  auto absorbed_dim_expr = mlir::dyn_cast<mlir::AffineDimExpr>(
      vars_order_map.getResult(absorbed_interm_idx));
  if (!absorbed_dim_expr) {
    op.emitError() << "variables_space_order result " << absorbed_interm_idx
                   << " is not a plain dim expression; "
                      "cannot determine access tile dimension to drop";
    return mlir::failure();
  }

  LDBG(1) << "getAbsorbedVarInfo: num_captured=" << num_captured
          << " absorbed_interm_idx=" << absorbed_interm_idx
          << " tile_dim_to_drop=" << absorbed_dim_expr.getPosition()
          << " trip_count=" << *trip_count;

  return AbsorbedVarInfo{num_captured, absorbed_interm_idx,
                         absorbed_dim_expr.getPosition(), *trip_count};
}

/// One ktdp.store found writing into an ind_addr_buf fill access tile, along
/// with the chain of ops that feed it.  `prior_collapses`, `fill_load`, and
/// `addr_buf_at` may be left empty/null when the walk back from `fill_store`
/// doesn't reach a recognisable load+AT (mirrors the caller's tolerant
/// "skip this part" behaviour rather than treating it as an error).
struct IabFillLink {
  mlir::ktdp::StoreOp fill_store;
  llvm::SmallVector<mlir::tensor::CollapseShapeOp> prior_collapses;
  mlir::ktdp::LoadOp fill_load;
  mlir::ktdp::ConstructAccessTilesOp addr_buf_at;
};

/// All ktdp.store links found for one ConstructAccessTilesOp built on the
/// IAB memref (normally exactly one).
struct IabFillChain {
  mlir::ktdp::ConstructAccessTilesOp iab_at;
  llvm::SmallVector<IabFillLink> links;
};

/// Discover the ind_addr_buf fill chain(s) feeding `iab_mv_result`: every
/// ConstructAccessTilesOp built on it (other than `exclude_op`'s own use of
/// the memref), and for each, every ktdp.store that targets it — walking back
/// through any tensor.collapse_shape chain inserted by a prior window
/// iteration to the ktdp.load and addr_buf ConstructAccessTilesOp that feed
/// the store.
///
/// The window loop materialization uses this discovery to *rebuild* the found
/// ops narrower; the entry loop materialization uses the same discovery to
/// *relocate* them (unchanged) into an `scf.if` `then` block — the walk
/// itself does not change.
static llvm::SmallVector<IabFillChain> findIabFillChains(
    mlir::Value iab_mv_result, mlir::Operation* exclude_op) {
  llvm::SmallVector<IabFillChain> chains;

  llvm::SmallVector<mlir::ktdp::ConstructAccessTilesOp> iab_fill_ats;
  for (mlir::OpOperand& use : iab_mv_result.getUses()) {
    if (use.getOwner() == exclude_op) continue;
    if (auto at =
            mlir::dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(use.getOwner()))
      iab_fill_ats.push_back(at);
  }

  for (mlir::ktdp::ConstructAccessTilesOp iab_at : iab_fill_ats) {
    IabFillChain chain;
    chain.iab_at = iab_at;

    for (mlir::OpOperand& at_use : iab_at.getResult().getUses()) {
      auto fill_store = mlir::dyn_cast<mlir::ktdp::StoreOp>(at_use.getOwner());
      if (!fill_store) continue;

      IabFillLink link;
      link.fill_store = fill_store;

      mlir::Value data_val = fill_store.getDataTile();
      while (auto collapse =
                 mlir::dyn_cast_if_present<mlir::tensor::CollapseShapeOp>(
                     data_val.getDefiningOp())) {
        link.prior_collapses.push_back(collapse);
        data_val = collapse.getSrc();
      }
      link.fill_load = mlir::dyn_cast_if_present<mlir::ktdp::LoadOp>(
          data_val.getDefiningOp());
      if (link.fill_load) {
        link.addr_buf_at =
            mlir::dyn_cast_if_present<mlir::ktdp::ConstructAccessTilesOp>(
                link.fill_load.getAccessTile().getDefiningOp());
      }
      chain.links.push_back(std::move(link));
    }
    chains.push_back(std::move(chain));
  }
  return chains;
}

/// Recursive helper for hoistAboveEnclosingForOps.  Appends `cur`, preceded by
/// every transitive operand definition that lives strictly inside
/// `dest_region` (i.e. inside the loop nest being escaped), to `ordered` so
/// that a definition always comes before its users.  `root` is only used for
/// diagnostics.
static mlir::LogicalResult collectHoistableDefs(
    mlir::Operation* cur, mlir::Region* dest_region,
    llvm::SmallPtrSetImpl<mlir::Operation*>& visited,
    llvm::SmallVectorImpl<mlir::Operation*>& ordered, mlir::Operation* root) {
  if (!visited.insert(cur).second) return mlir::success();

  for (mlir::Value operand : cur->getOperands()) {
    // Values defined at or above dest_region already dominate the hoist
    // point: anything in dest_region itself must precede the loop nest in
    // order to have dominated the original use inside it.
    if (!dest_region->isProperAncestor(operand.getParentRegion())) continue;

    mlir::Operation* def = operand.getDefiningOp();
    if (!def)
      return root->emitError()
             << "cannot hoist above the enclosing loop nest: operand is a "
                "block argument (e.g. an induction variable) of a loop in the "
                "nest";
    if (!mlir::isMemoryEffectFree(def))
      return root->emitError()
             << "cannot hoist above the enclosing loop nest: operand is "
                "defined inside the nest by '"
             << def->getName() << "', which is not side-effect free";
    if (mlir::failed(
            collectHoistableDefs(def, dest_region, visited, ordered, root)))
      return mlir::failure();
  }

  ordered.push_back(cur);
  return mlir::success();
}

/// Moves `op` to just before the outermost `scf.for` that currently encloses
/// it (a no-op if none does). Used for the IAB memory view: every one of its
/// operands (offset, static shape/stride attrs) is loop-invariant by
/// construction, so once built it can always be relocated to dominate the
/// whole loop nest rather than being reconstructed or threaded per
/// iteration.
///
/// Being loop-invariant is not the same as being *defined* outside the nest:
/// the offset operand is typically an `arith.constant` that was spliced into
/// a loop body together with the rest of the original code.  Leaving such a
/// definition behind would break dominance both for the hoisted op and for
/// its other in-nest users (which may later be relocated into the entry
/// loop's scf.if guard), so every transitive in-nest definition is hoisted
/// along with `op`.  Fails if one of them cannot be hoisted, rather than
/// producing invalid IR.
static mlir::LogicalResult hoistAboveEnclosingForOps(mlir::Operation* op) {
  mlir::Operation* hoist_point = op;
  while (auto parent_for = mlir::dyn_cast_if_present<mlir::scf::ForOp>(
             hoist_point->getParentOp()))
    hoist_point = parent_for.getOperation();
  if (hoist_point == op) return mlir::success();

  mlir::Region* dest_region = hoist_point->getParentRegion();
  llvm::SmallPtrSet<mlir::Operation*, 8> visited;
  llvm::SmallVector<mlir::Operation*> ordered;
  if (mlir::failed(collectHoistableDefs(op, dest_region, visited, ordered, op)))
    return mlir::failure();

  // `ordered` lists definitions before users, so moving each in turn to
  // immediately before hoist_point preserves that relative order.
  for (mlir::Operation* hoisted : ordered) hoisted->moveBefore(hoist_point);
  return mlir::success();
}

/// For one ConstructIndirectAccessTileOp, materialise all window scf.for
/// loops (all IAB subscript dimensions except the innermost per-entry one),
/// narrowing the IAB memref and updating the op on each iteration. The
/// access tile dimension to drop each iteration is derived from
/// variables_space_order.getResult(0) and is not assumed to be dim 0.
///
/// Returns the (possibly unchanged, when W == 0) op and the accumulated
/// window loop induction variables, which materializeEntryLoop needs to
/// continue from.
static mlir::FailureOr<
    std::pair<mlir::ktdp_lowering::ConstructIndirectAccessTileOp,
              llvm::SmallVector<mlir::Value>>>
materializeWindowLoops(mlir::ktdp_lowering::ConstructIndirectAccessTileOp op,
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
                   << ") does not equal hardware IAB size (" << iab_size << ")";
    return mlir::failure();
  }

  // W = number of window dimensions (all IAB dims except the innermost).
  int64_t W = iab_rank - 1;

  LDBG(1) << "materializeWindowLoops: op at " << op.getLoc()
          << " iab_rank=" << iab_rank << " iab_size=" << iab_size << " W=" << W;

  // If W == 0 the IAB is already 1-D; no window loops are needed.
  if (W == 0) {
    LDBG(1) << "  W == 0; no window loops needed";
    return std::make_pair(op, llvm::SmallVector<mlir::Value>{});
  }

  // The current op may be mutated (replaced) during the loop; keep a mutable
  // handle that always points to the live op.
  mlir::ktdp_lowering::ConstructIndirectAccessTileOp current_op = op;

  // Accumulate the loop IVs emitted on each iteration so that prior IVs can
  // be used as fixed subscripts in subsequent iterations.
  llvm::SmallVector<mlir::Value> window_ivs;

  for (int64_t k = 0; k < W; ++k) {
    LDBG(1) << "materializeWindowLoops: window iteration k=" << k
            << " of W=" << W;

    // ── Identify window variable w_k and its trip count ───────────────────
    // The window variable being absorbed is the one that drives the outermost
    // IAB subscript, i.e. ind_addr_buf_dim_positions[0].
    auto absorbed_info = getAbsorbedVarInfo(current_op);
    if (mlir::failed(absorbed_info)) return mlir::failure();
    unsigned absorbed_interm_idx = absorbed_info->absorbed_interm_idx;
    int64_t N = absorbed_info->trip_count;

    // ── Determine the splice boundary ─────────────────────────────────────
    // For k == 0 the boundary is derived from the def-use graph of
    // current_op; for k > 0 current_op is already inside the k-1 loop body,
    // so we splice the entire body of that loop.
    mlir::Block* src_block = current_op->getBlock();
    auto [splice_begin_op, splice_end_op] =
        computeSpliceBoundary(current_op, /*is_first_loop=*/k == 0);
    if (!splice_begin_op || !splice_end_op) {
      current_op.emitError()
          << "could not determine splice boundary for window loop " << k;
      return mlir::failure();
    }

    // ── Emit scf.for BEFORE splice_begin_op ───────────────────────────────
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
    auto parallel_attr =
        mlir::ktdf::LoopTypeAttr::get(ctx, mlir::ktdf::LoopType::ParallelLoop);
    for_op->setAttr("loop_type", parallel_attr);
    window_ivs.push_back(for_op.getInductionVar());

    LDBG(1) << "  emitted parallel scf.for for window k=" << k
            << " trip_count=" << N << " at " << for_op.getLoc();

    mlir::Block* dst_block = for_op.getBody();

    // Splice [splice_begin_op, splice_end_op] (inclusive) from src_block into
    // the for body.  for_op was inserted before splice_begin_op so it is
    // not in the range.  For k > 0 splice_end_op points to the last op before
    // the terminator so std::next reaches the terminator — same as before.
    mlir::Block::iterator splice_begin = splice_begin_op->getIterator();
    mlir::Block::iterator splice_end = std::next(splice_end_op->getIterator());
    dst_block->getOperations().splice(dst_block->begin(),
                                      src_block->getOperations(), splice_begin,
                                      splice_end);

    // ── Replace IAB construct_memory_view with a narrowed version ─────────
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

    LDBG(1) << "  narrowed IAB memref: " << cur_iab_memref_type << " -> "
            << new_iab_memref_type;

    // ── Compute the fields needed to rebuild construct_indirect_access_tile ─
    // The window variable being absorbed is the one that drives IAB dim 0
    // (the outermost IAB subscript).  Its index in the intermediate variable
    // list and its access-tile dimension were already computed above as
    // absorbed_interm_idx / absorbed_info->tile_dim_to_drop.  Its position in
    // the unified (captured..., intermediate...) dimension space is:
    unsigned absorbed_unified_dim =
        absorbed_info->num_captured + absorbed_interm_idx;
    unsigned tile_dim_to_drop = absorbed_info->tile_dim_to_drop;

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
    // map.  The absorbed dim index in the unified space is
    // absorbed_unified_dim.
    llvm::SmallVector<mlir::Attribute> new_subscript_maps;
    for (mlir::Attribute attr : current_op.getPerDimSubscriptMaps()) {
      mlir::AffineMap old_map =
          mlir::cast<mlir::AffineMapAttr>(attr).getValue();
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
      new_subscript_maps.push_back(
          mlir::AffineMapAttr::get(mlir::AffineMap::get(
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
    auto cur_result_type = mlir::cast<mlir::ktdp::AccessTileType>(
        current_op.getResult().getType());
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

    // ── Narrow the IAB fill chain ──────────────────────────────────────────
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

    auto iab_fill_chains =
        findIabFillChains(cur_iab_mv.getResult(), current_op.getOperation());
    LDBG(1) << "  found " << iab_fill_chains.size()
            << " ind_addr_buf fill chain(s) to narrow for window k=" << k;
    for (IabFillChain& chain : iab_fill_chains) {
      mlir::ktdp::ConstructAccessTilesOp iab_at = chain.iab_at;

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

      // For each store link found for iab_at, also narrow the data-source
      // chain: store.data_tile → (optional collapse_shape*) → ktdp.load
      //                        → ktdp.construct_access_tile (addr_buf_at).
      // On k=0 data_tile is directly a ktdp.load result.  On k>0 a prior
      // window iteration has inserted tensor.collapse_shape between the load
      // and the store; findIabFillChains already walked through any such ops
      // to reach fill_load / addr_buf_at (left null if not found).
      for (IabFillLink& link : chain.links) {
        mlir::ktdp::StoreOp fill_store = link.fill_store;
        pre_narrowed.push_back(fill_store.getOperation());

        if (!link.fill_load || !link.addr_buf_at) continue;

        // Rebuild the addr_buf access tile keeping the full base-memref rank:
        // on iteration k dim k is *pinned* to one element rather than dropped,
        // giving shape [1, 32, ...] (dims 0..k-1 were already pinned by prior
        // iterations).  See rebuildAccessTilePinned.
        auto new_addr_buf_at_or_failure =
            rebuildAccessTilePinned(link.addr_buf_at, static_cast<unsigned>(k),
                                    window_ivs.back(), loc, ctx);
        if (mlir::failed(new_addr_buf_at_or_failure)) return mlir::failure();
        auto new_addr_buf_at = *new_addr_buf_at_or_failure;
        pre_narrowed.push_back(new_addr_buf_at.getOperation());
        llvm::ArrayRef<int64_t> new_ab_shape =
            mlir::cast<mlir::ktdp::AccessTileType>(
                new_addr_buf_at.getResult().getType())
                .getShape();

        // Rebuild the fill load.  Its result type mirrors new_ab_shape.
        mlir::RankedTensorType old_fill_type =
            mlir::cast<mlir::RankedTensorType>(
                link.fill_load.getResult().getType());
        mlir::OpBuilder fl_b(link.fill_load);
        auto new_fill_load = mlir::ktdp::LoadOp::create(
            fl_b, link.fill_load.getLoc(), new_addr_buf_at.getResult(),
            old_fill_type.getElementType());
        pre_narrowed.push_back(new_fill_load.getOperation());

        // Insert tensor.collapse_shape to collapse the pinned leading dims of
        // tensor<1x...x{iab_shape}xindex> down to tensor<{iab_shape}xindex>,
        // matching the current IAB store operand shape (new_iab_shape).
        //
        // new_ab_shape:   [1, iab_d0, ..., iab_d{R-1}]   (k=0, W=1: [1,32])
        //                 [1, 1, iab_d0, ..., iab_d{R-1}] (k=1, W=2: [1,1,32])
        // new_iab_shape:  [iab_d0, ..., iab_d{R-1}]       (k=0, W=2: [2,32])
        //                 [iab_d{R-1}]                     (k=1, W=2: [32])
        //
        // Reassociation: fold leading (ab_result_rank - new_iab_rank) pinned
        // dims plus the first free dim into one group; remaining dims 1:1.
        //   e.g. k=0,W=1: ab=2, iab=1 → [[0,1]]          → tensor<32xindex>
        //        k=0,W=2: ab=3, iab=2 → [[0,1],[2]]      → tensor<2x32xindex>
        //        k=1,W=2: ab=3, iab=1 → [[0,1,2]]        → tensor<32xindex>
        unsigned ab_result_rank = static_cast<unsigned>(new_ab_shape.size());
        unsigned target_rank = static_cast<unsigned>(new_iab_shape.size());
        mlir::SmallVector<mlir::ReassociationIndices> reassoc =
            makeLeadingFoldReassociation(
                ab_result_rank,
                /*folded_leading_dims=*/ab_result_rank - target_rank);

        auto flat_type = mlir::RankedTensorType::get(
            llvm::SmallVector<int64_t>(new_iab_shape.begin(),
                                       new_iab_shape.end()),
            old_fill_type.getElementType());
        mlir::OpBuilder cs_b(fill_store);
        auto collapsed = mlir::tensor::CollapseShapeOp::create(
            cs_b, fill_store.getLoc(), flat_type, new_fill_load.getResult(),
            reassoc);
        pre_narrowed.push_back(collapsed.getOperation());

        // Update the store's data_tile to the new collapsed tensor.
        fill_store.getDataTileMutable().assign(collapsed.getResult());

        // Erase the prior collapse_shape chain (inserted by earlier iterations)
        // now that we have replaced it with the fresh one above.
        for (auto prior : link.prior_collapses) prior.erase();

        link.fill_load.erase();
        link.addr_buf_at.erase();
      }

      // Swap iab_at → new_iab_at in every store that references it.
      iab_at.getResult().replaceAllUsesWith(new_iab_at.getResult());
      iab_at.erase();
    }

    // ── Classify indirect load vs indirect store from current_op's users ──
    // Indirect load:  current_op is read by ktdp.load -> compute -> output
    // store Indirect store: current_op is written to by ktdp.store <- compute
    // <- source load
    bool is_indirect_load = false;
    bool is_indirect_store = false;
    for (mlir::Operation* user : current_op.getResult().getUsers()) {
      if (mlir::isa<mlir::ktdp::LoadOp>(user))
        is_indirect_load = true;
      else if (mlir::isa<mlir::ktdp::StoreOp>(user))
        is_indirect_store = true;
    }

    LDBG(1) << "  window k=" << k << ": is_indirect_load=" << is_indirect_load
            << " is_indirect_store=" << is_indirect_store;

    mlir::Block* const scope_block = dst_block;

    // ── Pin-not-drop for the destination access tile (indirect load) ──
    // The output ktdp.store stores the linalg.generic result into a destination
    // access tile via a ktdp.construct_access_tile.  That AT must keep full
    // base-memref rank (same invariant as the addr_buf AT).
    // pinDestAccessTile rebuilds it with dim k pinned to 1 and defers the
    // expand_shape insertion until after propagateNarrowing (the generic result
    // type is only drop-shaped after that call); it also self-heals any stale
    // expand left by a prior call (window iteration k-1) before doing so.
    std::optional<DeferredExpand> deferred_out;
    if (is_indirect_load) {
      auto pin_result = pinDestAccessTile(
          current_op, scope_block, window_ivs.back(), pre_narrowed, loc, ctx);
      if (mlir::failed(pin_result)) return mlir::failure();
      deferred_out = *pin_result;
    }

    // ── Pin-not-drop for source access tiles (indirect store) ────────────
    if (is_indirect_store) {
      if (mlir::failed(pinSourceAccessTile(current_op, scope_block,
                                           window_ivs.back(), pre_narrowed, loc,
                                           ctx)))
        return mlir::failure();
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

    LDBG(1) << "  window k=" << k << ": rebuilt construct_indirect_access_tile"
            << " with result type " << new_result_type;

    // Propagate the narrowed shape bidirectionally through the entire
    // shaped-value graph. The worklist algorithm handles any topology.
    // pre_narrowed contains the IAB mv and IAB fill chain ops already rebuilt
    // above so the walk does not attempt to re-narrow them.
    if (mlir::failed(propagateNarrowing(new_op, tile_dim_to_drop, window_ivs,
                                        iab_rank, pre_narrowed, loc, ctx)))
      return mlir::failure();

    LDBG(1) << "  window k=" << k << ": propagateNarrowing succeeded"
            << " (tile_dim_to_drop=" << tile_dim_to_drop << ")";

    // ── Deferred: insert expand_shape after propagateNarrowing ─────────────
    // Now that propagateNarrowing has drop-narrowed the compute result,
    // insert expand_shape immediately after its defining op to restore the
    // pinned shape for the output store AT.  Any prior-iteration expand was
    // already undone by this iteration's pinDestAccessTile call.
    if (deferred_out) {
      mlir::OpBuilder es_b(
          deferred_out->src_val.getDefiningOp()
              ? deferred_out->src_val.getDefiningOp()->getNextNode()
              : deferred_out->store.getOperation());
      auto out_expand = mlir::tensor::ExpandShapeOp::create(
          es_b, deferred_out->store.getLoc(), deferred_out->pinned_type,
          deferred_out->src_val, deferred_out->reassoc);
      deferred_out->store.getDataTileMutable().assign(out_expand.getResult());
    }
  }

  LDBG(1) << "materializeWindowLoops: done, materialized " << W
          << " window loop(s), final op at " << current_op.getLoc()
          << " result type " << current_op.getResult().getType();

  return std::make_pair(current_op, window_ivs);
}

/// Materialise the per-entry scf.for over individual ind_addr_buf entries.
/// Runs once, after materializeWindowLoops has absorbed every window
/// dimension; `window_ivs` is whatever that call returned (empty when there
/// were no window dimensions, i.e. W == 0).
///
/// Unlike the window loops, the IAB memref's rank does not change here — it
/// stays 1-D, size iab_size.  What changes is how that view is produced: the
/// ind_addr_buf fill chain (relocated, unmodified, from the rest of the body)
/// only runs on the first iteration, gated by an scf.if with no result.  The
/// IAB memory view itself (`cur_iab_mv`) is hoisted above the entire
/// window/entry loop nest — it is a pure declaration for a fixed hardware
/// region, so every operand is loop-invariant and no iter-arg is needed to
/// carry it between iterations.
static mlir::LogicalResult materializeEntryLoop(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp op,
    llvm::ArrayRef<mlir::Value> window_ivs, int64_t iab_size) {
  mlir::MLIRContext* ctx = op.getContext();
  mlir::Location loc = op.getLoc();
  mlir::ktdp_lowering::ConstructIndirectAccessTileOp current_op = op;

  LDBG(1) << "materializeEntryLoop: op at " << op.getLoc()
          << " window_ivs.size()=" << window_ivs.size()
          << " iab_size=" << iab_size;

  // ── Identify the entry variable and its trip count ─────────────────────
  auto absorbed_info = getAbsorbedVarInfo(current_op);
  if (mlir::failed(absorbed_info)) return mlir::failure();
  int64_t N = absorbed_info->trip_count;
  if (N != iab_size) {
    current_op.emitError() << "entry dimension trip count (" << N
                           << ") does not equal hardware IAB size (" << iab_size
                           << ")";
    return mlir::failure();
  }

  // ── Determine the splice boundary ─────────────────────────────────────
  bool is_first_loop = window_ivs.empty();
  mlir::Block* src_block = current_op->getBlock();
  auto [splice_begin_op, splice_end_op] =
      computeSpliceBoundary(current_op, is_first_loop);
  if (!splice_begin_op || !splice_end_op) {
    current_op.emitError()
        << "could not determine splice boundary for entry loop";
    return mlir::failure();
  }

  // ── Capture the current (rank-1) IAB memref op ──────────────────────────
  auto cur_iab_mv = mlir::cast<mlir::ktdp_lowering::ConstructMemoryViewOp>(
      current_op.getIndAddrBufMemref().getDefiningOp());

  // ── Emit c0/cN/c1 and the scf.for (no iter_args needed) ─────────────────
  mlir::OpBuilder builder(splice_begin_op);
  mlir::Value c0 =
      mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  mlir::Value cN =
      mlir::arith::ConstantIndexOp::create(builder, loc, N).getResult();
  mlir::Value c1 =
      mlir::arith::ConstantIndexOp::create(builder, loc, 1).getResult();

  // No loop_type annotation: the ind_addr_buf fill on iteration 0 must
  // happen-before every later iteration's read of the same entries, so this
  // loop is not safely reorderable/parallel even though it no longer carries
  // an SSA loop-carried value.
  auto for_op = mlir::scf::ForOp::create(builder, loc, c0, cN, c1);
  mlir::Value i2 = for_op.getInductionVar();

  LDBG(1) << "materializeEntryLoop: emitted entry scf.for trip_count=" << N
          << " at " << for_op.getLoc();

  llvm::SmallVector<mlir::Value> entry_ivs(window_ivs.begin(),
                                           window_ivs.end());
  entry_ivs.push_back(i2);

  mlir::Block* dst_block = for_op.getBody();

  // ── Splice [splice_begin_op, splice_end_op] into the for body ──────────
  // A no-iter-args ForOp::build auto-inserts an empty `scf.yield` terminator
  // into the body, so dst_block already ends in one; splicing at
  // dst_block->begin() lands the moved content before it, exactly where it
  // belongs, and no explicit re-termination is needed (mirrors
  // materializeWindowLoops, whose plain `scf.for` is handled the same way).
  mlir::Block::iterator splice_begin = splice_begin_op->getIterator();
  mlir::Block::iterator splice_end = std::next(splice_end_op->getIterator());
  dst_block->getOperations().splice(
      dst_block->begin(), src_block->getOperations(), splice_begin, splice_end);

  // ── Hoist the IAB memref view outside the entire loop nest ─────────────
  // ktdp_lowering.construct_memory_view is a pure declaration for a fixed
  // hardware region — all of its operands are loop-invariant — so it need
  // only be constructed once, above every window/entry loop, rather than
  // threaded through a loop-carried value. (materializeWindowLoops still
  // rebuilds a narrower view inside each window loop while the shape is
  // being reduced; only the final, fully-narrowed view reaching this
  // function gets hoisted.)
  if (mlir::failed(hoistAboveEnclosingForOps(cur_iab_mv.getOperation())))
    return mlir::failure();
  LDBG(1) << "materializeEntryLoop: hoisted IAB memory view to "
          << cur_iab_mv.getLoc();

  // ── Build the scf.if guard and relocate the fill chain ──────────────────
  // Insert `%eq0 = arith.cmpi eq, %i2, %c0` and the scf.if at the very front
  // of dst_block — everything currently there (the just-spliced body) ends
  // up after it.  The if has no result and no else region: %iab_mv is
  // already in scope (hoisted above the loop nest), so the then-branch is
  // purely the side-effecting fill.
  mlir::OpBuilder front_builder(dst_block, dst_block->begin());
  mlir::Value eq0 =
      mlir::arith::CmpIOp::create(front_builder, loc,
                                  mlir::arith::CmpIPredicate::eq, i2, c0)
          .getResult();
  auto if_op = mlir::scf::IfOp::create(front_builder, loc, mlir::TypeRange{},
                                       eq0, /*withElseRegion=*/false);
  mlir::Block& then_block = if_op.getThenRegion().front();
  // A zero-result IfOp::build auto-inserts an empty `scf.yield` terminator
  // into `then`.  Erase it now so the moves below (which all insert at
  // then_block.end()) land before the real terminator created further down,
  // rather than after it.
  then_block.getTerminator()->erase();

  auto fill_chains =
      findIabFillChains(cur_iab_mv.getResult(), current_op.getOperation());
  LDBG(1) << "materializeEntryLoop: found " << fill_chains.size()
          << " ind_addr_buf fill chain(s)";
  if (fill_chains.size() != 1 || fill_chains.front().links.size() != 1) {
    current_op.emitError()
        << "expected exactly one ind_addr_buf fill chain for the entry loop, "
           "found "
        << fill_chains.size() << " access tile(s) on the IAB memref";
    return mlir::failure();
  }
  IabFillChain& chain = fill_chains.front();
  IabFillLink& link = chain.links.front();
  if (!link.fill_load || !link.addr_buf_at) {
    current_op.emitError() << "could not discover the addr_buf load feeding "
                              "the ind_addr_buf fill";
    return mlir::failure();
  }

  // Move the upstream fill-prep ops (addr_buf AT, load, any collapse) into
  // `then` first, in their original relative block order.  Each of these
  // ops may have its own dedicated index/constant operands (e.g. the %c0
  // rebuildAccessTilePinned or narrowOp created immediately before it) that
  // are not part of the chain itself but must move along with it — moving
  // only the named op would strand such a private operand outside `then`,
  // after `if_op`, breaking dominance.  Close over every dst_block op whose
  // *every* use is already in the moving set (transitively) to catch these.
  auto closeOverPrivateOperands =
      [&](llvm::SmallPtrSetImpl<mlir::Operation*>& to_move) {
        bool changed = true;
        while (changed) {
          changed = false;
          for (mlir::Operation* op : llvm::SmallVector<mlir::Operation*>(
                   to_move.begin(), to_move.end())) {
            for (mlir::Value operand : op->getOperands()) {
              mlir::Operation* def = operand.getDefiningOp();
              if (!def || def->getBlock() != dst_block || to_move.count(def))
                continue;
              if (llvm::all_of(def->getUsers(), [&](mlir::Operation* user) {
                    return to_move.count(user);
                  })) {
                to_move.insert(def);
                changed = true;
              }
            }
          }
        }
      };

  llvm::SmallPtrSet<mlir::Operation*, 4> upstream_ops;
  upstream_ops.insert(link.addr_buf_at.getOperation());
  upstream_ops.insert(link.fill_load.getOperation());
  for (auto collapse : link.prior_collapses)
    upstream_ops.insert(collapse.getOperation());
  closeOverPrivateOperands(upstream_ops);
  for (mlir::Operation& blk_op : llvm::make_early_inc_range(*dst_block)) {
    if (upstream_ops.count(&blk_op))
      blk_op.moveBefore(&then_block, then_block.end());
  }

  // Move the fill AT (and any of its own private operands) and its store
  // into `then`.  No operand swap is needed: its base already references
  // `cur_iab_mv.getResult()`, which now dominates `then_block` directly
  // since `cur_iab_mv` was hoisted above the entire loop nest above.
  llvm::SmallPtrSet<mlir::Operation*, 4> at_store_ops;
  at_store_ops.insert(chain.iab_at.getOperation());
  at_store_ops.insert(link.fill_store.getOperation());
  closeOverPrivateOperands(at_store_ops);
  for (mlir::Operation& blk_op : llvm::make_early_inc_range(*dst_block)) {
    if (at_store_ops.count(&blk_op))
      blk_op.moveBefore(&then_block, then_block.end());
  }

  mlir::OpBuilder then_yield_b(&then_block, then_block.end());
  mlir::scf::YieldOp::create(then_yield_b, loc);

  LDBG(1) << "materializeEntryLoop: relocated ind_addr_buf fill chain into "
             "scf.if guard at "
          << if_op.getLoc();

  // ── Classify indirect load vs indirect store, as in materializeWindowLoops ─
  bool is_indirect_load = false;
  bool is_indirect_store = false;
  for (mlir::Operation* user : current_op.getResult().getUsers()) {
    if (mlir::isa<mlir::ktdp::LoadOp>(user))
      is_indirect_load = true;
    else if (mlir::isa<mlir::ktdp::StoreOp>(user))
      is_indirect_store = true;
  }

  LDBG(1) << "materializeEntryLoop: is_indirect_load=" << is_indirect_load
          << " is_indirect_store=" << is_indirect_store;

  // ── Pin-not-drop for access tiles ─────────────────────────────────────
  llvm::SmallVector<mlir::Operation*> pre_narrowed;
  std::optional<DeferredExpand> deferred_out;
  if (is_indirect_load) {
    auto pin_result =
        pinDestAccessTile(current_op, dst_block, i2, pre_narrowed, loc, ctx);
    if (mlir::failed(pin_result)) return mlir::failure();
    deferred_out = *pin_result;
  }
  if (is_indirect_store) {
    if (mlir::failed(pinSourceAccessTile(current_op, dst_block, i2,
                                         pre_narrowed, loc, ctx)))
      return mlir::failure();
  }

  // ── Rebuild construct_indirect_access_tile ──────────────────────────────
  // Unlike the window loops (which purely drop the absorbed dim), the entry
  // loop *relocates* the entry variable: it leaves the intermediate-variable
  // space and becomes a new captured variable (the entry loop's induction
  // variable), since the IAB memref stays rank 1 and its verifier requires
  // ind_addr_buf_dim_positions to have exactly one entry.  The entry
  // induction variable is appended right after the existing captured
  // variables, at unified position `num_captured`.
  unsigned num_captured = absorbed_info->num_captured;
  unsigned absorbed_unified_dim =
      num_captured + absorbed_info->absorbed_interm_idx;
  unsigned tile_dim_to_drop = absorbed_info->tile_dim_to_drop;

  mlir::IntegerSet new_vars_set =
      dropDimFromIntegerSet(current_op.getVariablesSpaceSet().getValue(),
                            absorbed_info->absorbed_interm_idx);
  mlir::AffineMap new_vars_order = dropDimFromAffineMap(
      current_op.getVariablesSpaceOrder(), absorbed_info->absorbed_interm_idx);

  unsigned new_num_interm =
      static_cast<unsigned>(current_op.getIntermediateVariables().size()) - 1;

  llvm::SmallVector<mlir::Value> captured_vars(
      current_op.getCapturedVariables().begin(),
      current_op.getCapturedVariables().end());
  captured_vars.push_back(i2);
  llvm::SmallVector<int32_t> new_iab_positions = {
      static_cast<int32_t>(num_captured)};

  // per_dim_subscript_maps' domain dimension *count* is unchanged: one
  // intermediate dim is removed, one captured dim is inserted, so the
  // unified total (captured + intermediate) stays the same.  The absorbed
  // dim itself is never referenced (an ind()-selecting variable is excluded
  // from direct per-dim subscripts by construction — see the op's
  // description), so it maps to an unused constant-0 expression.  Dims
  // strictly between the insertion point (num_captured) and the absorbed
  // slot (absorbed_unified_dim) — i.e. other intermediate variables that
  // preceded the entry variable — shift up by one to make room for the
  // newly-inserted entry induction variable at num_captured.  Dims beyond
  // the absorbed slot are unaffected: removing the absorbed dim and
  // inserting the entry captured dim cancel out for them.
  llvm::SmallVector<mlir::Attribute> new_subscript_maps;
  for (mlir::Attribute attr : current_op.getPerDimSubscriptMaps()) {
    mlir::AffineMap old_map = mlir::cast<mlir::AffineMapAttr>(attr).getValue();
    unsigned old_num_dims = old_map.getNumDims();
    llvm::SmallVector<mlir::AffineExpr> dim_repls(old_num_dims);
    for (unsigned d = 0; d < old_num_dims; ++d) {
      if (d == absorbed_unified_dim)
        dim_repls[d] = mlir::getAffineConstantExpr(0, ctx);  // never referenced
      else if (d >= num_captured && d < absorbed_unified_dim)
        dim_repls[d] = mlir::getAffineDimExpr(d + 1, ctx);
      else
        dim_repls[d] = mlir::getAffineDimExpr(d, ctx);
    }
    llvm::SmallVector<mlir::AffineExpr> new_results;
    for (unsigned r = 0; r < old_map.getNumResults(); ++r)
      new_results.push_back(
          old_map.getResult(r).replaceDimsAndSymbols(dim_repls, {}));
    new_subscript_maps.push_back(mlir::AffineMapAttr::get(mlir::AffineMap::get(
        old_num_dims, old_map.getNumSymbols(), new_results, ctx)));
  }

  mlir::Value base = current_op.getBase();

  auto cur_result_type =
      mlir::cast<mlir::ktdp::AccessTileType>(current_op.getResult().getType());
  llvm::SmallVector<int64_t> new_result_shape =
      dropShapeDim(cur_result_type.getShape(), tile_dim_to_drop);
  auto new_result_type = mlir::ktdp::AccessTileType::get(
      new_result_shape, cur_result_type.getElementType());

  mlir::OpBuilder replace_builder(current_op);
  auto new_op = mlir::ktdp_lowering::ConstructIndirectAccessTileOp::create(
      replace_builder, loc, new_result_type, base, cur_iab_mv.getResult(),
      mlir::DenseI32ArrayAttr::get(ctx, new_iab_positions),
      mlir::ArrayAttr::get(ctx, new_subscript_maps), captured_vars,
      new_num_interm, new_vars_order, new_vars_set);

  current_op.getResult().replaceAllUsesWith(new_op.getResult());
  current_op.erase();
  current_op = new_op;

  LDBG(1) << "materializeEntryLoop: rebuilt construct_indirect_access_tile"
          << " with result type " << new_result_type;

  // ── Propagate the narrowed shape through the compute chain ─────────────
  if (mlir::failed(propagateNarrowing(new_op, tile_dim_to_drop, entry_ivs,
                                      /*iab_rank=*/1, pre_narrowed, loc, ctx)))
    return mlir::failure();

  LDBG(1) << "materializeEntryLoop: propagateNarrowing succeeded"
          << " (tile_dim_to_drop=" << tile_dim_to_drop << ")";

  // ── Insert the deferred expand_shape, same as materializeWindowLoops ───
  if (deferred_out) {
    mlir::OpBuilder es_b(
        deferred_out->src_val.getDefiningOp()
            ? deferred_out->src_val.getDefiningOp()->getNextNode()
            : deferred_out->store.getOperation());
    auto out_expand = mlir::tensor::ExpandShapeOp::create(
        es_b, deferred_out->store.getLoc(), deferred_out->pinned_type,
        deferred_out->src_val, deferred_out->reassoc);
    deferred_out->store.getDataTileMutable().assign(out_expand.getResult());
  }

  // The scf.for body's auto-inserted terminator (see above) is still the
  // last op — no explicit re-termination needed.

  LDBG(1) << "materializeEntryLoop: done, final op at " << current_op.getLoc()
          << " result type " << current_op.getResult().getType();

  return mlir::success();
}

struct IndirectAddrBufLegalizationPass
    : public impl::IndirectAddrBufLegalizationPassBase<
          IndirectAddrBufLegalizationPass> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    LDBG(1) << "===== " PASS_NAME " =====";

    // Collect all construct_indirect_access_tile ops in each func.func,
    // enforcing that at most one indirect access tile exists per function.
    using FuncAndIndirectOp =
        std::pair<mlir::func::FuncOp,
                  mlir::ktdp_lowering::ConstructIndirectAccessTileOp>;
    llvm::SmallVector<FuncAndIndirectOp, 4> indirect_ops;

    // Walk the entire module to find all non-external func.funcs regardless of
    // nesting depth (direct children, inside child modules, etc.).
    unsigned num_funcs = 0;
    mlir::WalkResult outer_walk =
        module.walk([&](mlir::func::FuncOp func) -> mlir::WalkResult {
          if (func.isExternal()) return mlir::WalkResult::advance();
          ++num_funcs;
          mlir::ktdp_lowering::ConstructIndirectAccessTileOp func_indirect_op;
          mlir::WalkResult walk_res = func.walk(
              [&](mlir::ktdp_lowering::ConstructIndirectAccessTileOp op) {
                if (func_indirect_op) {
                  op->emitError(
                      "multiple construct_indirect_access_tile ops in the same "
                      "func.func are not supported");
                  return mlir::WalkResult::interrupt();
                }
                func_indirect_op = op;
                return mlir::WalkResult::advance();
              });
          if (walk_res.wasInterrupted()) return mlir::WalkResult::interrupt();
          if (func_indirect_op)
            indirect_ops.push_back({func, func_indirect_op});
          return mlir::WalkResult::advance();
        });
    if (outer_walk.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    LDBG(1) << "found " << indirect_ops.size()
            << " construct_indirect_access_tile op(s) across " << num_funcs
            << " func.func(s)";

    // No-op: return early when no indirect access tiles are present.
    if (indirect_ops.empty()) return;

    // Query the hardware IAB size from the architecture specification.
    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* const device = device_manager.getOrImportDevice();
    if (!device) {
      module->emitError(
          "Unable to import the device specification. This could happen if "
          "the device spec file is empty or contains multiple devices");
      signalPassFailure();
      return;
    }
    auto& resource_kinds =
        device_manager.getOrCreateView<mlir::ktdf_arch::ResourceKinds>(*device);

    int64_t iab_size = -1;
    for (const mlir::ktdf_arch::ResourceKinds::Kind& kind : resource_kinds) {
      const auto iab_feature =
          resource_kinds
              .getFeature<mlir::ktdf_arch::feature::IndirectAddressBuffer>(
                  kind);
      if (!iab_feature) continue;
      if (const auto num_entries = iab_feature.getNumEntries()) {
        iab_size = *num_entries;
        break;
      }
    }
    if (iab_size < 0) {
      module->emitWarning(
          "device has no indirect address buffer resource with num_entries; "
          "skipping indirect address buffer legalization");
      return;
    }

    LDBG(1) << "hardware indirect address buffer size = " << iab_size;

    for (auto& [func, op] : indirect_ops) {
      LDBG(1) << "legalizing construct_indirect_access_tile at " << op.getLoc();

      auto win_result = materializeWindowLoops(op, iab_size);
      if (mlir::failed(win_result)) {
        LDBG(1) << "  materializeWindowLoops FAILED";
        signalPassFailure();
        return;
      }
      LDBG(1) << "  materializeWindowLoops succeeded: emitted "
              << win_result->second.size() << " window loop(s), op now at "
              << win_result->first.getLoc();
      LDBG(2) << "IR after materializeWindowLoops:\n" << func;

      if (mlir::failed(materializeEntryLoop(win_result->first,
                                            win_result->second, iab_size))) {
        LDBG(1) << "  materializeEntryLoop FAILED";
        signalPassFailure();
        return;
      }
      LDBG(1) << "  materializeEntryLoop succeeded";
      LDBG(2) << "IR after materializeEntryLoop:\n" << func;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createIndirectAddrBufLegalizationPass() {
  return std::make_unique<IndirectAddrBufLegalizationPass>();
}
