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
// shaped-value graph, plus pin-not-drop rebuilding of descriptor access tiles
// pinned to a growing list of loop induction variables. Originally developed
// for IndirectAddrBufLegalization's window/entry loop materialization; the
// helpers here are dialect-agnostic with respect to *why* a dimension is
// being narrowed, so they are also used by IndirectAccessLoopMaterialization.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_INDIRECTACCESSTILENARROWING_H_
#define DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_INDIRECTACCESSTILENARROWING_H_

#include <cstdint>
#include <optional>

#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
#include "ktir/Dialect/KTDP/KTDP.h"
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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace scheduler {

/// Drop dimension `drop_dim` from `set` (which has `orig_num_dims` dimensions).
/// Constraints that reference `drop_dim` are removed; the remaining dimensions
/// above `drop_dim` are shifted down by one.
mlir::IntegerSet dropDimFromIntegerSet(mlir::IntegerSet set,
                                       unsigned drop_dim);

/// Pin dimension `pin_dim` of `set` to the single value 0 by replacing its
/// upper-bound inequality (`-d_k + (N-1) >= 0`) with `-d_k >= 0`. The
/// lower bound (`d_k >= 0`) and all constraints that do not involve `pin_dim`
/// are kept, so the set keeps its dimension count — unlike
/// dropDimFromIntegerSet, which removes the dimension entirely.
mlir::IntegerSet pinDimInIntegerSet(mlir::IntegerSet set, unsigned pin_dim);

/// Drop dimension `drop_dim` from `map` (which maps `orig_num_dims` dims).
/// The result expression at position `drop_dim` is removed; remaining
/// expressions with dim indices above `drop_dim` are shifted down.
///
/// Note: input dim `drop_dim` and result position `drop_dim` are dropped
/// together. For a square permutation map (e.g. access_tile_order) that is
/// only the intended narrowing when result `drop_dim` is `d_drop_dim`; a
/// permuted map would need the input dim found at result `drop_dim` instead.
mlir::AffineMap dropDimFromAffineMap(mlir::AffineMap map, unsigned drop_dim);

/// Return a copy of `shape` with the element at position `drop_dim` removed.
llvm::SmallVector<int64_t> dropShapeDim(llvm::ArrayRef<int64_t> shape,
                                        unsigned drop_dim);

/// Extract a constant trip count for dimension `dim_idx` from `set`.
/// Looks for a constraint of the form `-d_{dim_idx} + (N-1) >= 0` and returns
/// N. Returns std::nullopt if no such constraint is found.
std::optional<int64_t> getTripCount(mlir::IntegerSet set, unsigned dim_idx);

/// Build the reassociation for a collapse/expand that folds the leading
/// `folded_leading_dims` unit dims of a rank-`result_rank` shape into the
/// first following dim; the remaining dims map 1:1.
///   result_rank=2, folded=1 → [[0,1]]
///   result_rank=4, folded=1 → [[0,1],[2],[3]]
///   result_rank=5, folded=2 → [[0,1,2],[3],[4]]
mlir::SmallVector<mlir::ReassociationIndices> makeLeadingFoldReassociation(
    unsigned result_rank, unsigned folded_leading_dims);

/// Rebuild `at` with dimension `pin_dim` *pinned* to a single element instead
/// of dropped: the access tile keeps its full base-memref rank, dim `pin_dim`
/// gets extent 1, and its subscript becomes the current loop IV.
///
/// This is the "pin-not-drop" strategy required for every access tile built on
/// a base descriptor memref (addr_buf, output descriptor, source descriptor):
/// dropping a dim there would break the ConstructAccessTilesOp invariant
/// `access_tile_set.numDims == base_map.numInputs == base memref rank`.
///
/// `window_ivs` holds the IVs of all loops materialised so far, current
/// (innermost) one last; `pin_dim` must be its index. Dims 0..pin_dim-1 were
/// pinned by earlier iterations and take their own IV as subscript; dims above
/// pin_dim take %c0.
///
/// The caller owns the old op: this only inserts the replacement at `at`'s
/// position (along with the %c0 it needs) and returns it.
mlir::ktdp::ConstructAccessTilesOp rebuildAccessTilePinned(
    mlir::ktdp::ConstructAccessTilesOp at, unsigned pin_dim,
    llvm::ArrayRef<mlir::Value> window_ivs, mlir::Location loc,
    mlir::MLIRContext* ctx);

/// Collect all ops in `indirect_op`'s block that must move into a new loop
/// body wrapping it.
///
/// The reachability set is seeded from `indirect_op` and grown by:
///   UP:   follow def-use edges through operands whose type is narrowable
///         (AccessTileType or RankedTensorType). When an IAB memref operand
///         is encountered, also include every other user of that memref
///         (the IAB fill chain) and walk their operands upward.
///   DOWN: follow def-use edges through users of `indirect_op`'s result,
///         then UP from each user's non-indirect operands (pulls in the
///         indirect store source-data chain that is only reachable downstream).
///
/// Returns {splice_begin_op, splice_end_op}: the earliest
/// ktdp.construct_access_tile and the latest ktdp.store in block order
/// among the collected set. Both are guaranteed to be non-null because
/// the input MLIR is in the canonical "indirect_load + compute + store"
/// form.
std::pair<mlir::Operation*, mlir::Operation*> findSpliceBoundary(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp indirect_op);

/// Pick the splice boundary for the next loop being materialised around
/// `current_op`. `is_first_loop` is true exactly when no loop has been
/// materialised yet for this indirect op in the caller's traversal — in that
/// case the boundary must be discovered via the def-use reachability walk
/// (`findSpliceBoundary`). Otherwise the entire body of the
/// previously-materialised loop is the boundary.
std::pair<mlir::Operation*, mlir::Operation*> computeSpliceBoundary(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp current_op,
    bool is_first_loop);

/// State needed to insert a deferred tensor.expand_shape after
/// propagateNarrowing has drop-narrowed the compute result (see
/// pinOutputDescriptorAT).
struct DeferredExpand {
  mlir::Value src_val;
  mlir::ktdp::StoreOp store;
  mlir::SmallVector<mlir::ReassociationIndices> reassoc;
  mlir::RankedTensorType pinned_type;
};

/// Pin-not-drop the output descriptor access tile for an indirect *load*
/// (gather), if `current_op`'s narrowed result feeds one within
/// `scope_block`. `pin_dim` must be `window_ivs.size() - 1` (see
/// rebuildAccessTilePinned).
///
/// If a prior call already bridged this same output store with a
/// tensor.expand_shape (i.e. `out_store`'s data_tile is exactly that
/// expand's result), that stale expand is undone first so this call narrows
/// the raw compute value — this makes the function self-contained across
/// repeated calls (once per newly-materialised loop) without the caller
/// needing to thread any "prior expand" state between them.
///
/// The returned state (when present) must be used by the caller, after
/// propagateNarrowing has run, to insert a fresh tensor.expand_shape
/// restoring the pinned shape and rewire the store to it. Absent when
/// `current_op` is not an indirect load, or no matching output store is
/// found in scope. **Known limitation** (inherited from the pre-refactor
/// code): this state is single-valued, so a body with more than one output
/// store would only get the last one bridged.
std::optional<DeferredExpand> pinOutputDescriptorAT(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp current_op,
    mlir::Block* scope_block, unsigned pin_dim,
    llvm::ArrayRef<mlir::Value> window_ivs,
    llvm::SmallVectorImpl<mlir::Operation*>& pre_narrowed, mlir::Location loc,
    mlir::MLIRContext* ctx);

/// Pin-not-drop the source descriptor access tile for an indirect *store*
/// (scatter), if `current_op` is written to by a store within `scope_block`
/// whose source data chain reaches a direct-descriptor load. Unlike
/// pinOutputDescriptorAT this is immediate (not deferred): the collapse_shape
/// it inserts must be in place before propagateNarrowing narrows the generic
/// that consumes it.
void pinSourceDescriptorAT(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp current_op,
    mlir::Block* scope_block, unsigned pin_dim,
    llvm::ArrayRef<mlir::Value> window_ivs,
    llvm::SmallVectorImpl<mlir::Operation*>& pre_narrowed, mlir::Location loc,
    mlir::MLIRContext* ctx);

/// Bidirectional def-use narrowing propagation.
///
/// Starting from `new_op` (already rebuilt with a narrowed result type),
/// propagate the shape change through the shaped-value graph **within the
/// single basic block that contains `new_op`** (the just-materialised loop
/// body). Ops whose defining/owning block is different are silently skipped —
/// this prevents the walk from crossing into outer-scope ops (base memref
/// views, output desc memrefs, etc.) that must not be narrowed.
///
/// Parameters:
///   new_op           — Already-rebuilt ConstructIndirectAccessTileOp.
///   tile_dim_to_drop — Access-tile / tensor dimension to remove.
///   window_ivs       — All loop IVs accumulated so far (including the current
///                      one). Passed through for index placement.
///   iab_rank         — Rank of the IAB memref *before* this absorption, so
///                      the ConstructMemoryViewOp branch can identify it.
///                      Pass a value no in-scope memref can have (e.g. a
///                      negative rank) when there is no IAB memref to narrow.
///   pre_narrowed     — Ops already correctly rebuilt by the caller (e.g.
///                      pin-not-drop of descriptor ATs): pre-inserted into
///                      narrowed_ops so the walk skips re-mutating them.
///   loc / ctx        — Needed when building new ops.
mlir::LogicalResult propagateNarrowing(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp new_op,
    unsigned tile_dim_to_drop, llvm::ArrayRef<mlir::Value> window_ivs,
    int64_t iab_rank, llvm::ArrayRef<mlir::Operation*> pre_narrowed,
    mlir::Location loc, mlir::MLIRContext* ctx);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_INDIRECTACCESSTILENARROWING_H_
