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
#include "dataflow-scheduler/Transforms/Utils/IndirectAccessTileNarrowing.h"
#include "ktir/Dialect/KTDP/KTDP.h"
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
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"

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

/// Shared by sub-step 2a (per window dim) and sub-step 2b (the final
/// per-entry dim): read off the absorbed variable's index, its constant trip
/// count (from variables_space_set), and the access-tile dimension it maps to
/// (from variables_space_order, needed for downstream narrowing).
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
/// Sub-step 2a uses this discovery to *rebuild* the found ops narrower;
/// sub-step 2b uses the same discovery to *relocate* them (unchanged) into
/// an `scf.if` `then` block — the walk itself does not change.
static llvm::SmallVector<IabFillChain> findIabFillChains(
    mlir::Value iab_mv_result, mlir::Operation* exclude_op) {
  llvm::SmallVector<IabFillChain> chains;

  llvm::SmallVector<mlir::ktdp::ConstructAccessTilesOp> iab_fill_ats;
  for (mlir::OpOperand& use : iab_mv_result.getUses()) {
    if (use.getOwner() == exclude_op) continue;
    if (auto at = mlir::dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(
            use.getOwner()))
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

/// Sub-step 2a: for one ConstructIndirectAccessTileOp, materialise all window
/// scf.for loops (all IAB subscript dimensions except the innermost per-entry
/// one), narrowing the IAB memref and updating the op on each iteration.
/// The access tile dimension to drop each iteration is derived from
/// variables_space_order.getResult(0) and is not assumed to be dim 0.
///
/// Returns the (possibly unchanged, when W == 0) op and the accumulated
/// window loop induction variables, which sub-step 2b's materializeEntryLoop
/// needs to continue from.
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
                   << ") does not equal hardware IAB size (" << iab_size
                   << ")";
    return mlir::failure();
  }

  // W = number of window dimensions (all IAB dims except the innermost).
  int64_t W = iab_rank - 1;

  // If W == 0 the IAB is already 1-D; nothing to do for sub-step 2a.
  if (W == 0)
    return std::make_pair(op, llvm::SmallVector<mlir::Value>{});

  // The current op may be mutated (replaced) during the loop; keep a mutable
  // handle that always points to the live op.
  mlir::ktdp_lowering::ConstructIndirectAccessTileOp current_op = op;

  // Accumulate the loop IVs emitted on each iteration so that prior IVs can
  // be used as fixed subscripts in subsequent iterations.
  llvm::SmallVector<mlir::Value> window_ivs;

  for (int64_t k = 0; k < W; ++k) {
    // ── Step 1: identify window variable w_k and trip count ──────────────
    // The window variable being absorbed is the one that drives the outermost
    // IAB subscript, i.e. ind_addr_buf_dim_positions[0].
    auto absorbed_info = getAbsorbedVarInfo(current_op);
    if (mlir::failed(absorbed_info)) return mlir::failure();
    unsigned absorbed_interm_idx = absorbed_info->absorbed_interm_idx;
    int64_t N = absorbed_info->trip_count;

    // ── Step 2: determine splice boundary ────────────────────────────────
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
    window_ivs.push_back(for_op.getInductionVar());

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

    auto iab_fill_chains =
        findIabFillChains(cur_iab_mv.getResult(), current_op.getOperation());
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
        auto new_addr_buf_at = rebuildAccessTilePinned(
            link.addr_buf_at, /*pin_dim=*/static_cast<unsigned>(k), window_ivs,
            loc, ctx);
        pre_narrowed.push_back(new_addr_buf_at.getOperation());
        llvm::ArrayRef<int64_t> new_ab_shape =
            mlir::cast<mlir::ktdp::AccessTileType>(
                new_addr_buf_at.getResult().getType())
                .getShape();

        // Rebuild the fill load.  Its result type mirrors new_ab_shape.
        mlir::RankedTensorType old_fill_type = mlir::cast<mlir::RankedTensorType>(
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
            cs_b, fill_store.getLoc(), flat_type,
            new_fill_load.getResult(), reassoc);
        pre_narrowed.push_back(collapsed.getOperation());

        // Update the store's data_tile to the new collapsed tensor.
        fill_store.getDataTileMutable().assign(collapsed.getResult());

        // Erase the prior collapse_shape chain (inserted by earlier iterations)
        // now that we have replaced it with the fresh one above.
        for (auto prior : link.prior_collapses)
          prior.erase();

        link.fill_load.erase();
        link.addr_buf_at.erase();
      }

      // Swap iab_at → new_iab_at in every store that references it.
      iab_at.getResult().replaceAllUsesWith(new_iab_at.getResult());
      iab_at.erase();
    }

    // ── Classify indirect load vs indirect store from current_op's users ──
    // Indirect load:  current_op is read by ktdp.load -> compute -> output store
    // Indirect store: current_op is written to by ktdp.store <- compute <- source load
    bool is_indirect_load = false;
    bool is_indirect_store = false;
    for (mlir::Operation* user : current_op.getResult().getUsers()) {
      if (mlir::isa<mlir::ktdp::LoadOp>(user))
        is_indirect_load = true;
      else if (mlir::isa<mlir::ktdp::StoreOp>(user))
        is_indirect_store = true;
    }

    mlir::Block* const scope_block = dst_block;

    // ── Step 5b: pin-not-drop for the output descriptor AT (indirect load) ─
    // The output ktdp.store stores the linalg.generic result into an output
    // descriptor via a ktdp.construct_access_tile.  That AT must keep full
    // base-memref rank (same invariant as the addr_buf AT).  pinOutputDescriptorAT
    // rebuilds it with dim k pinned to 1 and defers the expand_shape insertion
    // until after propagateNarrowing (the generic result type is only
    // drop-shaped after that call); it also self-heals any stale expand left
    // by a prior call (window iteration k-1) before doing so.
    std::optional<DeferredExpand> deferred_out;
    if (is_indirect_load) {
      deferred_out = pinOutputDescriptorAT(current_op, scope_block,
                                           /*pin_dim=*/static_cast<unsigned>(k),
                                           window_ivs, pre_narrowed, loc, ctx);
    }

    // ── Step 5c: pin-not-drop for source descriptor ATs (indirect store) ──
    if (is_indirect_store) {
      pinSourceDescriptorAT(current_op, scope_block,
                            /*pin_dim=*/static_cast<unsigned>(k), window_ivs,
                            pre_narrowed, loc, ctx);
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
                                        window_ivs, iab_rank,
                                        pre_narrowed, loc, ctx)))
      return mlir::failure();

    // ── Step 5b deferred: insert expand_shape after propagateNarrowing ──
    // Now that propagateNarrowing has drop-narrowed the compute result,
    // insert expand_shape immediately after its defining op to restore the
    // pinned shape for the output store AT.  Any prior-iteration expand was
    // already undone by this iteration's pinOutputDescriptorAT call.
    if (deferred_out) {
      mlir::OpBuilder es_b(deferred_out->src_val.getDefiningOp()
                               ? deferred_out->src_val.getDefiningOp()->getNextNode()
                               : deferred_out->store.getOperation());
      auto out_expand = mlir::tensor::ExpandShapeOp::create(
          es_b, deferred_out->store.getLoc(), deferred_out->pinned_type,
          deferred_out->src_val, deferred_out->reassoc);
      deferred_out->store.getDataTileMutable().assign(out_expand.getResult());
    }
  }

  return std::make_pair(current_op, window_ivs);
}

/// Sub-step 2b: materialise the per-entry scf.for over individual
/// ind_addr_buf entries.  Runs once, after materializeWindowLoops has
/// absorbed every window dimension; `window_ivs` is whatever that call
/// returned (empty when sub-step 2a was a W == 0 no-op).
///
/// Unlike the window loops, the IAB memref's rank does not change here — it
/// stays 1-D, size iab_size.  What changes is how that view is produced: the
/// ind_addr_buf fill chain (relocated, unmodified, from the rest of the body)
/// only runs on the first iteration (%i2 == 0), gated by an scf.if, with the
/// resulting view threaded through the loop as an iter-arg.
static mlir::LogicalResult materializeEntryLoop(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp op,
    llvm::ArrayRef<mlir::Value> window_ivs, int64_t iab_size) {
  mlir::MLIRContext* ctx = op.getContext();
  mlir::Location loc = op.getLoc();
  mlir::ktdp_lowering::ConstructIndirectAccessTileOp current_op = op;

  // ── Step 1: identify the entry variable and its trip count ─────────────
  auto absorbed_info = getAbsorbedVarInfo(current_op);
  if (mlir::failed(absorbed_info)) return mlir::failure();
  int64_t N = absorbed_info->trip_count;
  if (N != iab_size) {
    current_op.emitError() << "entry dimension trip count (" << N
                           << ") does not equal hardware IAB size ("
                           << iab_size << ")";
    return mlir::failure();
  }

  // ── Step 2: splice boundary ──────────────────────────────────────────────
  bool is_first_loop = window_ivs.empty();
  mlir::Block* src_block = current_op->getBlock();
  auto [splice_begin_op, splice_end_op] =
      computeSpliceBoundary(current_op, is_first_loop);
  if (!splice_begin_op || !splice_end_op) {
    current_op.emitError()
        << "could not determine splice boundary for entry loop";
    return mlir::failure();
  }

  // ── Step 3: capture the current (rank-1) IAB memref op ──────────────────
  auto cur_iab_mv = mlir::cast<mlir::ktdp_lowering::ConstructMemoryViewOp>(
      current_op.getIndAddrBufMemref().getDefiningOp());
  mlir::MemRefType iab_mv_type =
      mlir::cast<mlir::MemRefType>(cur_iab_mv.getResult().getType());

  // Build an identical (but distinct) ConstructMemoryViewOp at the given
  // builder's insertion point.  Used for both the outer sentinel and the
  // then-local fill target — same attributes as cur_iab_mv, never narrowed.
  auto cloneIabMv = [&](mlir::OpBuilder& b) {
    return mlir::ktdp_lowering::ConstructMemoryViewOp::create(
        b, loc, iab_mv_type, cur_iab_mv.getOffset(),
        /*sizes=*/mlir::ValueRange{}, /*strides=*/mlir::ValueRange{},
        cur_iab_mv.getStaticSizes(), cur_iab_mv.getStaticStrides(),
        cur_iab_mv.getMemorySpace(), cur_iab_mv.getCoordinateSet());
  };

  // ── Step 4: emit c0/cN/c1, the sentinel, and the scf.for with iter_args ─
  mlir::OpBuilder builder(splice_begin_op);
  mlir::Value c0 =
      mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  mlir::Value cN =
      mlir::arith::ConstantIndexOp::create(builder, loc, N).getResult();
  mlir::Value c1 =
      mlir::arith::ConstantIndexOp::create(builder, loc, 1).getResult();
  auto iab_mv_init = cloneIabMv(builder);

  // No loop_type annotation: unlike window loops, this loop carries state
  // across iterations via the iter-arg, so parallel_loop does not apply.
  auto for_op = mlir::scf::ForOp::create(builder, loc, c0, cN, c1,
                                        mlir::ValueRange{iab_mv_init.getResult()});
  mlir::Value i2 = for_op.getInductionVar();
  mlir::Value iab_mv_carry = for_op.getRegionIterArg(0);

  llvm::SmallVector<mlir::Value> entry_ivs(window_ivs.begin(), window_ivs.end());
  entry_ivs.push_back(i2);
  unsigned pin_dim = static_cast<unsigned>(window_ivs.size());

  mlir::Block* dst_block = for_op.getBody();

  // ── Step 5: splice [splice_begin_op, splice_end_op] into the for body ───
  // The for body has no terminator yet (scf::ForOp with iter_args and no
  // bodyBuilder leaves that to the caller), so the spliced content simply
  // becomes the block's entire (as yet unterminated) contents.
  mlir::Block::iterator splice_begin = splice_begin_op->getIterator();
  mlir::Block::iterator splice_end = std::next(splice_end_op->getIterator());
  dst_block->getOperations().splice(dst_block->begin(), src_block->getOperations(),
                                    splice_begin, splice_end);

  // ── Step 6/7: build the scf.if guard and relocate the fill chain ────────
  // Insert `%eq0 = arith.cmpi eq, %i2, %c0` and the scf.if at the very front
  // of dst_block — everything currently there (the just-spliced body) ends
  // up after it, matching the design doc's IR shape.
  mlir::OpBuilder front_builder(dst_block, dst_block->begin());
  mlir::Value eq0 = mlir::arith::CmpIOp::create(
      front_builder, loc, mlir::arith::CmpIPredicate::eq, i2, c0)
                        .getResult();
  auto if_op = mlir::scf::IfOp::create(front_builder, loc,
                                       mlir::TypeRange{iab_mv_type}, eq0,
                                       /*withElseRegion=*/true);
  mlir::Block& then_block = if_op.getThenRegion().front();
  mlir::Block& else_block = if_op.getElseRegion().front();

  auto fill_chains =
      findIabFillChains(cur_iab_mv.getResult(), current_op.getOperation());
  if (fill_chains.size() != 1 || fill_chains.front().links.size() != 1) {
    current_op.emitError()
        << "expected exactly one ind_addr_buf fill chain for sub-step 2b, "
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
          for (mlir::Operation* op :
               llvm::SmallVector<mlir::Operation*>(to_move.begin(), to_move.end())) {
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

  // Then-local IAB memref clone: the fill's target, distinct from the outer
  // sentinel (`iab_mv_init`) — matches the design doc's separate `%iab_mv_`.
  mlir::OpBuilder then_clone_b(&then_block, then_block.end());
  auto iab_mv_then = cloneIabMv(then_clone_b);

  // Redirect the fill AT's base operand to the then-local clone — no shape
  // or attribute change, so an operand swap is enough — then move the AT
  // (and any of its own private operands) and its store after the clone.
  chain.iab_at.getBaseMutable().assign(iab_mv_then.getResult());
  llvm::SmallPtrSet<mlir::Operation*, 4> at_store_ops;
  at_store_ops.insert(chain.iab_at.getOperation());
  at_store_ops.insert(link.fill_store.getOperation());
  closeOverPrivateOperands(at_store_ops);
  for (mlir::Operation& blk_op : llvm::make_early_inc_range(*dst_block)) {
    if (at_store_ops.count(&blk_op))
      blk_op.moveBefore(&then_block, then_block.end());
  }

  mlir::OpBuilder then_yield_b(&then_block, then_block.end());
  mlir::scf::YieldOp::create(then_yield_b, loc,
                             mlir::ValueRange{iab_mv_then.getResult()});
  mlir::OpBuilder else_yield_b(&else_block, else_block.end());
  mlir::scf::YieldOp::create(else_yield_b, loc, mlir::ValueRange{iab_mv_carry});

  // ── Step 8: rewire the rest of the body to the scf.if's result ─────────
  cur_iab_mv.getResult().replaceAllUsesWith(if_op.getResult(0));
  cur_iab_mv.erase();

  // ── Classify indirect load vs indirect store, as in sub-step 2a ─────────
  bool is_indirect_load = false;
  bool is_indirect_store = false;
  for (mlir::Operation* user : current_op.getResult().getUsers()) {
    if (mlir::isa<mlir::ktdp::LoadOp>(user))
      is_indirect_load = true;
    else if (mlir::isa<mlir::ktdp::StoreOp>(user))
      is_indirect_store = true;
  }

  // ── Step 9: pin-not-drop for descriptor ATs ─────────────────────────────
  llvm::SmallVector<mlir::Operation*> pre_narrowed;
  std::optional<DeferredExpand> deferred_out;
  if (is_indirect_load) {
    deferred_out = pinOutputDescriptorAT(current_op, dst_block, pin_dim,
                                         entry_ivs, pre_narrowed, loc, ctx);
  }
  if (is_indirect_store) {
    pinSourceDescriptorAT(current_op, dst_block, pin_dim, entry_ivs,
                          pre_narrowed, loc, ctx);
  }

  // ── Step 10: rebuild construct_indirect_access_tile ─────────────────────
  // Unlike sub-step 2a (which purely drops the absorbed dim), sub-step 2b
  // *relocates* the entry variable: it leaves the intermediate-variable
  // space and becomes a new captured variable (%i2), since the IAB memref
  // stays rank 1 and its verifier requires ind_addr_buf_dim_positions to
  // have exactly one entry.  %i2 is appended right after the existing
  // captured variables, at unified position `num_captured`.
  unsigned num_captured = absorbed_info->num_captured;
  unsigned absorbed_unified_dim = num_captured + absorbed_info->absorbed_interm_idx;
  unsigned tile_dim_to_drop = absorbed_info->tile_dim_to_drop;

  mlir::IntegerSet new_vars_set = dropDimFromIntegerSet(
      current_op.getVariablesSpaceSet().getValue(),
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
  // description), so only dims strictly between the insertion point
  // (num_captured) and the absorbed slot (absorbed_unified_dim) need to
  // shift, to make room for the newly-inserted %i2 at num_captured. That
  // range is empty whenever the entry variable is the first remaining
  // intermediate variable (the frontend's convention: IAB-related
  // intermediate variables always precede direct ones), so in practice this
  // is a no-op; the general shift is implemented so it stays correct if
  // that convention ever changes.
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
      replace_builder, loc, new_result_type, base, if_op.getResult(0),
      mlir::DenseI32ArrayAttr::get(ctx, new_iab_positions),
      mlir::ArrayAttr::get(ctx, new_subscript_maps), captured_vars,
      new_num_interm, new_vars_order, new_vars_set);

  current_op.getResult().replaceAllUsesWith(new_op.getResult());
  current_op.erase();
  current_op = new_op;

  // ── Step 11: propagate the narrowed shape through the compute chain ────
  if (mlir::failed(propagateNarrowing(new_op, tile_dim_to_drop, entry_ivs,
                                      /*iab_rank=*/1, pre_narrowed, loc, ctx)))
    return mlir::failure();

  // ── Step 12: insert the deferred expand_shape, same as sub-step 2a ─────
  if (deferred_out) {
    mlir::OpBuilder es_b(deferred_out->src_val.getDefiningOp()
                             ? deferred_out->src_val.getDefiningOp()->getNextNode()
                             : deferred_out->store.getOperation());
    auto out_expand = mlir::tensor::ExpandShapeOp::create(
        es_b, deferred_out->store.getLoc(), deferred_out->pinned_type,
        deferred_out->src_val, deferred_out->reassoc);
    deferred_out->store.getDataTileMutable().assign(out_expand.getResult());
  }

  // Terminate the scf.for body: carry the scf.if's result to the next
  // iteration's iter-arg.
  mlir::OpBuilder for_yield_b(dst_block, dst_block->end());
  mlir::scf::YieldOp::create(for_yield_b, loc,
                             mlir::ValueRange{if_op.getResult(0)});

  return mlir::success();
}

struct IndirectAddrBufLegalizationPass
    : public impl::IndirectAddrBufLegalizationPassBase<
          IndirectAddrBufLegalizationPass> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Collect all construct_indirect_access_tile ops in each func.func,
    // enforcing that at most one indirect access tile exists per function.
    llvm::SmallVector<
        mlir::ktdp_lowering::ConstructIndirectAccessTileOp, 4>
        indirect_ops;

    for (auto func : module.getOps<mlir::func::FuncOp>()) {
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

      if (walk_res.wasInterrupted()) {
        signalPassFailure();
        return;
      }
      if (func_indirect_op) indirect_ops.push_back(func_indirect_op);
    }

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
      auto win_result = materializeWindowLoops(op, iab_size);
      if (mlir::failed(win_result)) {
        signalPassFailure();
        return;
      }
      if (mlir::failed(materializeEntryLoop(win_result->first,
                                            win_result->second, iab_size))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass>
scheduler::createIndirectAddrBufLegalizationPass() {
  return std::make_unique<IndirectAddrBufLegalizationPass>();
}
