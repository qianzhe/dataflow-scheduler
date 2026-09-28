//===-------------------------------------------------------------*- c++ -*-==//
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
// Shared helpers for narrowing a ktdp_lowering.construct_indirect_access_tile
// op by one dimension and propagating that narrowing through its downstream
// shaped-value graph.
//
// Two strategies eliminate a dimension's degree of freedom as each loop is
// materialized:
//   - Dropping: the dimension is deleted outright, so rank decreases by one
//     (dropDimFromIntegerSet, dropDimFromAffineMap, dropShapeDim). This is
//     the default, used for values with no base-memref rank to preserve.
//   - Pinning: the dimension is kept but its extent is fixed to 1 and its
//     subscript becomes the current loop induction variable, so rank is
//     unchanged (pinDimInIntegerSet, rebuildAccessTilePinned).
//
// Access tiles built on a base memref (addr_buf, destination access tile,
// source access tile) must use pinning instead of the drop-narrowing
// everything else gets, because dropping would break the
// ConstructAccessTilesOp invariant `access_tile_set.numDims ==
// base_map.numInputs == base memref rank`. This file calls that substitution
// "pin-not-drop": pinDestAccessTile()/pinSourceAccessTile() rebuild those
// access tiles pinned to a growing list of loop induction variables, in place
// of the drop that propagateNarrowing() would otherwise apply to them. Used
// by IndirectAddrBufLegalization's window and entry loop materialization.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_INDIRECTACCESSTILENARROWING_H_
#define DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_INDIRECTACCESSTILENARROWING_H_

#include <cstdint>
#include <optional>

#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace scheduler {

/// Drop a dimension from an integer set.
///
/// Constraints that reference @p drop_dim are removed; the remaining
/// dimensions above @p drop_dim are shifted down by one.
///
/// @param set      The integer set to narrow.
/// @param drop_dim Index of the dimension to remove.
/// @returns The narrowed integer set.
mlir::IntegerSet dropDimFromIntegerSet(mlir::IntegerSet set, unsigned drop_dim);

/// Pin a dimension of an integer set to the single value 0, by intersecting
/// the set with the equality `d_k == 0` and dropping any bound on @p pin_dim
/// that equality now makes redundant. All other constraints are kept, so the
/// set keeps its dimension count — unlike dropDimFromIntegerSet, which
/// removes the dimension entirely.
///
/// @param set     The integer set to pin.
/// @param pin_dim Index of the dimension to pin to 0.
/// @returns The pinned integer set.
mlir::IntegerSet pinDimInIntegerSet(mlir::IntegerSet set, unsigned pin_dim);

/// Drop a dimension from an affine map.
///
/// The result expression at position @p drop_dim is removed; remaining
/// expressions with dim indices above @p drop_dim are shifted down.
///
/// Note: input dim @p drop_dim and result position @p drop_dim are dropped
/// together. For a square permutation map (e.g. access_tile_order) that is
/// only the intended narrowing when result @p drop_dim is `d_drop_dim`; a
/// permuted map would need the input dim found at result @p drop_dim
/// instead.
///
/// @param map      The affine map to narrow.
/// @param drop_dim Index of the input dim / result position to remove.
/// @returns The narrowed affine map.
mlir::AffineMap dropDimFromAffineMap(mlir::AffineMap map, unsigned drop_dim);

/// Return a copy of a shape with one element removed.
///
/// @param shape    The shape to copy from.
/// @param drop_dim Index of the element to remove.
/// @returns A copy of @p shape with the element at @p drop_dim removed.
llvm::SmallVector<int64_t> dropShapeDim(llvm::ArrayRef<int64_t> shape,
                                        unsigned drop_dim);

/// Extract a constant trip count for a dimension of an integer set.
///
/// Looks for a constraint of the form `-d_{dim_idx} + (N-1) >= 0` and
/// returns N.
///
/// @param set     The integer set to search.
/// @param dim_idx Index of the dimension whose trip count is extracted.
/// @returns The trip count N, or std::nullopt if no such constraint is
///          found.
std::optional<int64_t> getTripCount(mlir::IntegerSet set, unsigned dim_idx);

/// Build the reassociation for a collapse/expand that folds the leading
/// unit dims of a shape into the first dim; the remaining dims map 1:1.
///
///   result_rank=2, folded=1 → [[0,1]]
///   result_rank=4, folded=1 → [[0,1],[2],[3]]
///   result_rank=5, folded=2 → [[0,1,2],[3],[4]]
///
/// @param result_rank         Rank of the resulting (expanded) shape.
/// @param folded_leading_dims Number of leading unit dims folded into the
///                            first dim.
/// @returns The reassociation indices.
mlir::SmallVector<mlir::ReassociationIndices> makeLeadingFoldReassociation(
    unsigned result_rank, unsigned folded_leading_dims);

/// Rebuild an access tile with one dimension *pinned* to a single element
/// instead of dropped: the access tile keeps its full base-memref rank, the
/// pinned dim gets extent 1, and its subscript becomes the current loop IV.
///
/// This is the "pin-not-drop" strategy required for every access tile built
/// on a base memref (addr_buf, destination access tile, source access tile):
/// dropping a dim there would break the ConstructAccessTilesOp invariant
/// `access_tile_set.numDims == base_map.numInputs == base memref rank`.
///
/// The subscripts for dims pinned by earlier calls (0..pin_dim-1) are
/// recovered from @p at's own current indices — this function set them on
/// every prior pin, so the caller need not track or pass them itself; dims
/// above @p pin_dim take %c0.
///
/// The caller owns the old op: this only inserts the replacement at @p at's
/// position (along with the %c0 it needs) and returns it.
///
/// @param at      The access tile op to rebuild.
/// @param pin_dim Index of the dimension this call pins. Callers typically
///                derive this as the length of the leading run of extent-1
///                dims in @p at's shape (the prefix already pinned by
///                earlier calls), so it lines up with whatever they also
///                need that run's length for (e.g. sizing a fold).
/// @param new_iv  Loop IV for the dimension this call pins.
/// @param loc     Location for newly-inserted ops.
/// @param ctx     MLIR context.
/// @returns The rebuilt, pinned access tile op, or failure (with a
///          diagnostic on @p at) if @p pin_dim is not a valid dimension of
///          @p at — the caller's signal that @p at's rank does not match how
///          many dimensions it still expected to absorb.
mlir::FailureOr<mlir::ktdp::ConstructAccessTilesOp> rebuildAccessTilePinned(
    mlir::ktdp::ConstructAccessTilesOp at, unsigned pin_dim, mlir::Value new_iv,
    mlir::Location loc, mlir::MLIRContext* ctx);

/// Collect all ops in an indirect op's block that must move into a new loop
/// body wrapping it.
///
/// The reachability set is seeded from @p indirect_op and grown by:
///   UP:   follow def-use edges through operands whose type is narrowable
///         (AccessTileType or RankedTensorType). When an IAB memref operand
///         is encountered, also include every other user of that memref
///         (the IAB fill chain) and walk their operands upward.
///   DOWN: follow def-use edges through users of @p indirect_op's result,
///         then UP from each user's non-indirect operands (pulls in the
///         indirect store source-data chain that is only reachable
///         downstream).
///
/// @param indirect_op The indirect access tile op to seed the walk from.
/// @returns {splice_begin_op, splice_end_op}: the earliest
///          ktdp.construct_access_tile and the latest ktdp.store in block
///          order among the collected set. Both are guaranteed to be
///          non-null because the input MLIR is in the canonical
///          "indirect_load + compute + store" form.
std::pair<mlir::Operation*, mlir::Operation*> findSpliceBoundary(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp indirect_op);

/// Pick the splice boundary for the next loop being materialised around an
/// indirect access tile op.
///
/// @param current_op    The indirect access tile op the new loop wraps.
/// @param is_first_loop True exactly when no loop has been materialised yet
///                      for this indirect op in the caller's traversal — in
///                      that case the boundary must be discovered via the
///                      def-use reachability walk (findSpliceBoundary()).
///                      Otherwise the entire body of the
///                      previously-materialised loop is the boundary.
/// @returns The {splice_begin_op, splice_end_op} boundary.
std::pair<mlir::Operation*, mlir::Operation*> computeSpliceBoundary(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp current_op,
    bool is_first_loop);

/// State needed to insert a deferred tensor.expand_shape after
/// propagateNarrowing has drop-narrowed the compute result (see
/// pinDestAccessTile).
struct DeferredExpand {
  mlir::Value src_val;
  mlir::ktdp::StoreOp store;
  mlir::SmallVector<mlir::ReassociationIndices> reassoc;
  mlir::RankedTensorType pinned_type;
};

/// Pin-not-drop the destination access tile for an indirect *load*
/// (gather), if @p current_op's narrowed result feeds one within
/// @p scope_block.
///
/// If a prior call already bridged this same output store with a
/// tensor.expand_shape (i.e. `out_store`'s data_tile is exactly that
/// expand's result), that stale expand is undone first so this call narrows
/// the raw compute value — this makes the function self-contained across
/// repeated calls (once per newly-materialised loop) without the caller
/// needing to thread any "prior expand" state between them.
///
/// **Known limitation** (inherited from the pre-refactor code): the
/// returned state is single-valued, so a body with more than one output
/// store would only get the last one bridged.
///
/// @param current_op   The indirect access tile op just narrowed.
/// @param scope_block  Block searched for a matching output store.
/// @param new_iv       Loop IV for the dimension this call pins (see
///                     rebuildAccessTilePinned()).
/// @param pre_narrowed Ops already rebuilt by this call, appended so a
///                     later propagateNarrowing() walk skips re-mutating
///                     them.
/// @param loc          Location for newly-inserted ops.
/// @param ctx          MLIR context.
/// @returns Failure if rebuildAccessTilePinned() fails on the discovered
///          access tile. Otherwise, state to be used by the caller, after
///          propagateNarrowing has run, to insert a fresh
///          tensor.expand_shape restoring the pinned shape and rewire the
///          store to it — std::nullopt when @p current_op is not an
///          indirect load, or no matching output store is found in scope.
mlir::FailureOr<std::optional<DeferredExpand>> pinDestAccessTile(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp current_op,
    mlir::Block* scope_block, mlir::Value new_iv,
    llvm::SmallVectorImpl<mlir::Operation*>& pre_narrowed, mlir::Location loc,
    mlir::MLIRContext* ctx);

/// Pin-not-drop the source access tile for an indirect *store*
/// (scatter), if @p current_op is written to by a store within
/// @p scope_block whose source data chain reaches a direct-access-tile load.
/// Unlike pinDestAccessTile() this is immediate (not deferred): the
/// collapse_shape it inserts must be in place before propagateNarrowing
/// narrows the generic that consumes it.
///
/// @param current_op   The indirect access tile op just narrowed.
/// @param scope_block  Block searched for a matching source-access-tile load.
/// @param new_iv       Loop IV for the dimension this call pins (see
///                     rebuildAccessTilePinned()).
/// @param pre_narrowed Ops already rebuilt by this call, appended so a
///                     later propagateNarrowing() walk skips re-mutating
///                     them.
/// @param loc          Location for newly-inserted ops.
/// @param ctx          MLIR context.
/// @returns Failure if rebuildAccessTilePinned() fails on a discovered
///          source access tile; success otherwise (including when no
///          matching source-access-tile load is found in scope).
mlir::LogicalResult pinSourceAccessTile(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp current_op,
    mlir::Block* scope_block, mlir::Value new_iv,
    llvm::SmallVectorImpl<mlir::Operation*>& pre_narrowed, mlir::Location loc,
    mlir::MLIRContext* ctx);

/// Bidirectional def-use narrowing propagation.
///
/// Starting from @p new_op (already rebuilt with a narrowed result type),
/// propagate the shape change through the shaped-value graph **within the
/// single basic block that contains @p new_op** (the just-materialised loop
/// body). Ops whose defining/owning block is different are silently skipped
/// — this prevents the walk from crossing into outer-scope ops (base memref
/// views, output desc memrefs, etc.) that must not be narrowed.
///
/// @param new_op           Already-rebuilt ConstructIndirectAccessTileOp.
/// @param tile_dim_to_drop Access-tile / tensor dimension to remove.
/// @param window_ivs       All loop IVs accumulated so far (including the
///                         current one). Passed through for index
///                         placement.
/// @param iab_rank         Rank of the IAB memref *before* this absorption,
///                         so the ConstructMemoryViewOp branch can identify
///                         it. Pass a value no in-scope memref can have
///                         (e.g. a negative rank) when there is no IAB
///                         memref to narrow.
/// @param pre_narrowed     Ops already correctly rebuilt by the caller (e.g.
///                         pin-not-drop of access tiles): pre-inserted
///                         into narrowed_ops so the walk skips re-mutating
///                         them.
/// @param loc              Location for newly-inserted ops.
/// @param ctx              MLIR context.
/// @returns Failure if an op reachable in the walk is not one of the known
///          narrowable op kinds; success otherwise.
mlir::LogicalResult propagateNarrowing(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp new_op,
    unsigned tile_dim_to_drop, llvm::ArrayRef<mlir::Value> window_ivs,
    int64_t iab_rank, llvm::ArrayRef<mlir::Operation*> pre_narrowed,
    mlir::Location loc, mlir::MLIRContext* ctx);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_INDIRECTACCESSTILENARROWING_H_
