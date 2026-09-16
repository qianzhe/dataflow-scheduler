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
// IndirectAccessLoopMaterialization: materialize scf.for loops for any
// remaining direct-subscript intermediate variable of a
// ktdp_lowering.construct_indirect_access_tile whose $base dimension cannot
// be serviced by a single hardware indirect transfer (fails dense-packing
// or full-coverage).
//
//===----------------------------------------------------------------------===//

#include <memory>
#include <optional>

#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFAttributes.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFEnums.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Transforms/Utils/IndirectAccessTileNarrowing.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "indirect-access-loop-materialization"
#define DEBUG_TYPE PASS_NAME

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_INDIRECTACCESSLOOPMATERIALIZATIONPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

/// One remaining direct-subscript intermediate variable, classified against
/// $base's own dimension `base_dim`.
struct DirectVarInfo {
  unsigned
      interm_idx;  // position in intermediate_variables at classification time
  unsigned base_dim;   // dimension of $base this variable subscripts
  int64_t trip_count;  // constant trip count (N) from variables_space_set
};

/// Returns the single dimension position `map`'s (one-result) expression is
/// a function of, restricted to positions >= `min_dim`. Returns failure if
/// zero or more than one such position is found.
mlir::FailureOr<unsigned> findReferencedDim(mlir::AffineMap map,
                                            unsigned min_dim) {
  if (map.getNumResults() != 1) return mlir::failure();
  mlir::AffineExpr expr = map.getResult(0);
  int found = -1;
  for (unsigned d = min_dim; d < map.getNumDims(); ++d) {
    if (expr.isFunctionOfDim(d)) {
      if (found >= 0) return mlir::failure();
      found = static_cast<int>(d);
    }
  }
  if (found < 0) return mlir::failure();
  return static_cast<unsigned>(found);
}

/// Classify every remaining intermediate variable of `op` against `$base`'s
/// own dimension order: a variable is retained when its dimension is both
/// densely packed with its neighbor and fully covered by the tile; otherwise
/// it needs a materialized loop. `loop_vars` is returned sorted by `base_dim`
/// ascending (outermost $base dimension first).
mlir::LogicalResult classifyDirectVariables(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp op,
    llvm::ArrayRef<int64_t> base_sizes, llvm::ArrayRef<int64_t> base_strides,
    llvm::SmallVectorImpl<DirectVarInfo>& loop_vars,
    llvm::SmallVectorImpl<DirectVarInfo>& retained_vars) {
  unsigned base_rank = static_cast<unsigned>(base_sizes.size());
  unsigned num_captured = op.getCapturedVariables().size();
  unsigned num_interm = op.getIntermediateVariables().size();
  mlir::ArrayAttr per_dim_maps = op.getPerDimSubscriptMaps();
  mlir::IntegerSet vars_set = op.getVariablesSpaceSet().getValue();

  LDBG(1) << "classifyDirectVariables: base_rank=" << base_rank
          << " num_captured=" << num_captured << " num_interm=" << num_interm;

  for (unsigned i = 0; i < num_interm; ++i) {
    unsigned target_unified_dim = num_captured + i;

    int found_k = -1;
    for (unsigned k = 0; k < base_rank; ++k) {
      auto map = mlir::cast<mlir::AffineMapAttr>(per_dim_maps[k]).getValue();
      if (map.getNumResults() != 1) {
        op.emitError() << "per_dim_subscript_maps[" << k
                       << "] does not have exactly one result";
        return mlir::failure();
      }
      if (map.getResult(0).isFunctionOfDim(target_unified_dim)) {
        if (found_k >= 0) {
          op.emitError() << "intermediate variable " << i
                         << " is referenced by more than one "
                            "per_dim_subscript_maps entry";
          return mlir::failure();
        }
        found_k = static_cast<int>(k);
      }
    }
    if (found_k < 0) {
      op.emitError() << "intermediate variable " << i
                     << " is not referenced by any per_dim_subscript_maps "
                        "entry";
      return mlir::failure();
    }
    unsigned k = static_cast<unsigned>(found_k);

    auto trip_count = getTripCount(vars_set, i);
    if (!trip_count) {
      op.emitError() << "could not extract constant trip count for "
                        "intermediate variable "
                     << i << " from variables_space_set";
      return mlir::failure();
    }

    bool dense_packed =
        (k == base_rank - 1)
            ? (base_strides[k] == 1)
            : (base_strides[k] == base_sizes[k + 1] * base_strides[k + 1]);
    bool full_coverage = (*trip_count == base_sizes[k]);

    DirectVarInfo info{i, k, *trip_count};
    LDBG(1) << "  interm_idx=" << i << " -> base_dim=" << k
            << " trip_count=" << *trip_count << " dense_packed=" << dense_packed
            << " full_coverage=" << full_coverage << " -> "
            << (dense_packed && full_coverage ? "retained" : "loop_var");
    if (dense_packed && full_coverage)
      retained_vars.push_back(info);
    else
      loop_vars.push_back(info);
  }

  llvm::sort(loop_vars, [](const DirectVarInfo& a, const DirectVarInfo& b) {
    return a.base_dim < b.base_dim;
  });
  LDBG(1) << "classifyDirectVariables: " << loop_vars.size() << " loop var(s), "
          << retained_vars.size() << " retained var(s)";
  return mlir::success();
}

/// Snapshot of the source/destination access tile's current state,
/// read (never mutated) before this pass's own loop starts.
struct AccessTilePeek {
  unsigned rank;
  // Indices for exactly the leading dims already pinned (extent-1) by a
  // prior pass; does NOT include not-yet-pinned (still full-extent) dims,
  // which must stay defaulted to a zero index by rebuildAccessTilePinned
  // until this pass actually reaches them.
  llvm::SmallVector<mlir::Value> pinned_prefix_indices;
};

/// Read-only mirror of pinDestAccessTile/pinSourceAccessTile's
/// discovery walk: locates the destination access tile (indirect load
/// / gather case, discovered downstream of `op`'s result) or the source
/// access tile (indirect store / scatter case, discovered
/// upstream of the store that writes into `op`'s result), without rebuilding
/// anything. Used once, before this pass's own loop, to recover how many
/// access tile dimensions IndirectAddrBufLegalization already pinned (and
/// their subscripts), so this pass can continue the same monotonic
/// dimension numbering without needing to re-derive
/// IndirectAddrBufLegalization's window/entry induction variables from
/// scratch. The already-pinned leading dims are identified by extent == 1
/// in the discovered tile's own shape (set explicitly by
/// rebuildAccessTilePinned on every prior pin), not by inspecting
/// access_tile_set constraint forms.
std::optional<AccessTilePeek> peekAccessTile(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp op,
    mlir::Block* scope_block, bool is_indirect_load) {
  LDBG(1) << "peekAccessTile: op at " << op.getLoc()
          << " is_indirect_load=" << is_indirect_load;
  mlir::ktdp::ConstructAccessTilesOp found;

  if (is_indirect_load) {
    llvm::SmallVector<mlir::Value> worklist{op.getResult()};
    llvm::DenseSet<mlir::Value> visited{op.getResult()};
    while (!worklist.empty() && !found) {
      mlir::Value cur_val = worklist.pop_back_val();
      for (mlir::Operation* user : cur_val.getUsers()) {
        if (user->getBlock() != scope_block) continue;
        if (auto store = mlir::dyn_cast<mlir::ktdp::StoreOp>(user)) {
          found = mlir::dyn_cast_if_present<mlir::ktdp::ConstructAccessTilesOp>(
              store.getAccessTile().getDefiningOp());
          if (found) break;
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
  } else {
    for (mlir::Operation* user : op.getResult().getUsers()) {
      auto store = mlir::dyn_cast<mlir::ktdp::StoreOp>(user);
      if (!store || store->getBlock() != scope_block) continue;

      llvm::SmallVector<mlir::Value> worklist{store.getDataTile()};
      llvm::DenseSet<mlir::Value> visited{store.getDataTile()};
      while (!worklist.empty() && !found) {
        mlir::Value cur_val = worklist.pop_back_val();
        mlir::Operation* def = cur_val.getDefiningOp();
        if (!def || def->getBlock() != scope_block) continue;

        if (auto load = mlir::dyn_cast<mlir::ktdp::LoadOp>(def)) {
          found = mlir::dyn_cast_if_present<mlir::ktdp::ConstructAccessTilesOp>(
              load.getAccessTile().getDefiningOp());
          continue;
        }
        for (mlir::Value operand : def->getOperands()) {
          if (mlir::isa<mlir::ktdp::AccessTileType, mlir::RankedTensorType>(
                  operand.getType()) &&
              visited.insert(operand).second)
            worklist.push_back(operand);
        }
      }
      if (found) break;
    }
  }

  if (!found) {
    LDBG(1) << "  no source/destination access tile found";
    return std::nullopt;
  }
  AccessTilePeek peek;
  auto ty = mlir::cast<mlir::ktdp::AccessTileType>(found.getResult().getType());
  llvm::ArrayRef<int64_t> shape = ty.getShape();
  peek.rank = static_cast<unsigned>(shape.size());

  unsigned pinned_prefix = 0;
  while (pinned_prefix < shape.size() && shape[pinned_prefix] == 1)
    ++pinned_prefix;

  mlir::OperandRange indices = found.getIndices();
  peek.pinned_prefix_indices.assign(indices.begin(),
                                    indices.begin() + pinned_prefix);
  LDBG(1) << "  found access tile at " << found.getLoc()
          << " rank=" << peek.rank
          << " already-pinned prefix=" << pinned_prefix;
  return peek;
}

/// Finds the `Kind` co-located with `iab_kind` (i.e. sharing the same nearest
/// kinded enclosing group, per Resource::getFeature's exemplar semantics)
/// whose exemplar carries `FeatureAttr`. Returns a null `Kind` if none
/// exists.
template <class FeatureAttr>
mlir::ktdf_arch::ResourceKinds::Kind findColocatedUnit(
    const mlir::ktdf_arch::ResourceKinds& resource_kinds,
    mlir::ktdf_arch::ResourceKinds::Kind iab_kind) {
  auto iab_parents = iab_kind.getParentKinds();
  for (const mlir::ktdf_arch::ResourceKinds::Kind& kind : resource_kinds) {
    if (!kind.getExemplar().getFeature<FeatureAttr>()) continue;
    for (mlir::ktdf_arch::KindAttr parent : kind.getParentKinds())
      if (iab_parents.count(parent)) return kind;
  }
  return {};
}

/// Minimum permissible transfer size, in bytes, that `unit` (a Load- or
/// Store-capable exec unit) supports for `memory_space`. `std::nullopt` if
/// `unit` is null or lacks granularity info for `memory_space`.
template <class FeatureAttr>
std::optional<int64_t> getMinTransferSizeInBytes(mlir::ktdf_arch::Resource unit,
                                                 mlir::Attribute memory_space) {
  if (!unit) return std::nullopt;
  auto feature = unit.getFeature<FeatureAttr>();
  if (!feature) return std::nullopt;
  auto accesses = feature.getAccessGranularity(memory_space);
  if (!accesses) return std::nullopt;
  auto smallest =
      accesses.fitAccess(1, mlir::ktdf_arch::AccessGranularityAttr::kMaxAlign);
  if (!smallest) return std::nullopt;
  return static_cast<int64_t>(smallest.getSizeInWords()) *
         static_cast<int64_t>(feature.getWordSize(memory_space));
}

/// Device-level info resolved once in the pass driver and threaded through to
/// materializeDirectAccessLoops, needed to validate a single-transfer
/// region's minimum size against the actual hardware load/store units
/// co-located with the indirect address buffer (as opposed to the IAB's own
/// entry capacity, which is a separate, unrelated resource property).
struct TransferSizeInfo {
  llvm::DenseMap<mlir::Attribute, mlir::Attribute> mem_space_map;
  mlir::ktdf_arch::Resource load_unit;   // null if none co-located with IAB
  mlir::ktdf_arch::Resource store_unit;  // null if none co-located with IAB

  /// Resolves a ktdp-level logical memory space (e.g.
  /// `#ktdp.memory_space<global>`) to the concrete arch Kind attribute
  /// declared for it in the device's `mem_space_mapping`, or returns
  /// `declared` unchanged if no mapping applies.
  [[nodiscard]] mlir::Attribute resolveMemorySpace(
      mlir::Attribute declared) const {
    if (const auto mapped = mem_space_map.lookup(declared)) return mapped;
    return declared;
  }
};

/// Validates that `retained_count` elements of `base_mv` (used as an
/// indirect load per `is_indirect_load` and/or a store per
/// `is_indirect_store`) form a region at least as large as the smallest
/// transfer the corresponding hardware unit(s) in `transfer_info` can
/// service. Emits a diagnostic on `op` and returns failure if the region is
/// too small, or if the minimum could not be determined at all.
mlir::LogicalResult checkMinimumTransferSize(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp op,
    mlir::ktdp::ConstructMemoryViewOp base_mv, int64_t retained_count,
    bool is_indirect_load, bool is_indirect_store,
    const TransferSizeInfo& transfer_info) {
  auto elem_size_bytes = scheduler::tryGetSizeInBytes(
      mlir::cast<mlir::MemRefType>(base_mv.getResult().getType())
          .getElementType());
  if (!elem_size_bytes) {
    op.emitError() << "could not determine the byte size of $base's element "
                      "type";
    return mlir::failure();
  }
  int64_t retained_bytes =
      retained_count * static_cast<int64_t>(*elem_size_bytes);

  mlir::Attribute resolved_space =
      transfer_info.resolveMemorySpace(base_mv.getMemorySpace());

  std::optional<int64_t> min_transfer_bytes;
  if (is_indirect_load) {
    min_transfer_bytes =
        getMinTransferSizeInBytes<mlir::ktdf_arch::feature::Load>(
            transfer_info.load_unit, resolved_space);
  }
  if (is_indirect_store) {
    auto store_min = getMinTransferSizeInBytes<mlir::ktdf_arch::feature::Store>(
        transfer_info.store_unit, resolved_space);
    if (store_min && (!min_transfer_bytes || *store_min > *min_transfer_bytes))
      min_transfer_bytes = store_min;
  }
  if (!min_transfer_bytes) {
    op.emitError() << "could not determine the minimum hardware transfer "
                      "size for $base's memory space";
    return mlir::failure();
  }
  if (retained_bytes < *min_transfer_bytes) {
    op.emitError() << "retained element count (" << retained_count << ", "
                   << retained_bytes
                   << " bytes) is below the minimum hardware transfer size ("
                   << *min_transfer_bytes << " bytes)";
    return mlir::failure();
  }
  return mlir::success();
}

/// Materialize one scf.for per direct-subscript intermediate variable of
/// `op` that cannot be serviced by a single hardware indirect transfer,
/// outermost-$base-dimension-first. No-op (returns success without changes)
/// when every remaining variable is retainable.
mlir::LogicalResult materializeDirectAccessLoops(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp op,
    const TransferSizeInfo& transfer_info) {
  mlir::MLIRContext* ctx = op.getContext();
  mlir::Location loc = op.getLoc();

  LDBG(1) << "materializeDirectAccessLoops: op at " << op.getLoc();

  auto base_mv = mlir::cast<mlir::ktdp::ConstructMemoryViewOp>(
      op.getBase().getDefiningOp());
  llvm::ArrayRef<int64_t> base_sizes = base_mv.getStaticSizes();
  llvm::ArrayRef<int64_t> base_strides = base_mv.getStaticStrides();
  unsigned base_rank = static_cast<unsigned>(base_sizes.size());

  // ── Step 1: classify ────────────────────────────────────────────────────
  llvm::SmallVector<DirectVarInfo> loop_vars, retained_vars;
  if (mlir::failed(classifyDirectVariables(op, base_sizes, base_strides,
                                           loop_vars, retained_vars)))
    return mlir::failure();

  // How $base is used, needed by both Step 2 and Step 4
  bool is_indirect_load = false;
  bool is_indirect_store = false;
  for (mlir::Operation* user : op.getResult().getUsers()) {
    if (mlir::isa<mlir::ktdp::LoadOp>(user))
      is_indirect_load = true;
    else if (mlir::isa<mlir::ktdp::StoreOp>(user))
      is_indirect_store = true;
  }

  // ── Step 2: validate ────────────────────────────────────────────────────
  if (!loop_vars.empty() && !retained_vars.empty()) {
    unsigned max_loop_dim = loop_vars.back().base_dim;
    unsigned min_retained_dim = retained_vars.front().base_dim;
    for (const DirectVarInfo& r : retained_vars)
      min_retained_dim = std::min(min_retained_dim, r.base_dim);
    if (max_loop_dim >= min_retained_dim) {
      op.emitError() << "retained direct-subscript variables are not "
                        "innermost relative to the variables requiring a "
                        "materialized loop";
      return mlir::failure();
    }
  }

  int64_t retained_count = 1;
  for (const DirectVarInfo& r : retained_vars) retained_count *= r.trip_count;
  if (mlir::failed(checkMinimumTransferSize(op, base_mv, retained_count,
                                            is_indirect_load, is_indirect_store,
                                            transfer_info)))
    return mlir::failure();

  if (base_strides[base_rank - 1] != 1) {
    op.emitError() << "innermost dimension of $base does not have unit "
                      "stride; no contiguous region exists to transfer";
    return mlir::failure();
  }

  LDBG(1) << "materializeDirectAccessLoops: validated, retained_count="
          << retained_count << " loop_vars=" << loop_vars.size();

  // ── Step 3: no-op path ──────────────────────────────────────────────────
  if (loop_vars.empty()) {
    LDBG(1) << "  no loop vars remain; no-op";
    return mlir::success();
  }

  // ── Step 4: materialize, outermost-$base-dimension-first ───────────────
  mlir::ktdp_lowering::ConstructIndirectAccessTileOp current_op = op;

  // Seed window_ivs/pin_dim from whatever IndirectAddrBufLegalization
  // already pinned on the source/destination's own access tile (read-only
  // peek; see peekAccessTile above).
  llvm::SmallVector<mlir::Value> window_ivs;
  if (auto peek = peekAccessTile(current_op, current_op->getBlock(),
                                 is_indirect_load)) {
    window_ivs.assign(peek->pinned_prefix_indices.begin(),
                      peek->pinned_prefix_indices.end());
    if (peek->rank !=
        window_ivs.size() + loop_vars.size() + retained_vars.size()) {
      current_op.emitError()
          << "source/destination access tile rank (" << peek->rank
          << ") does not match already-pinned dims (" << window_ivs.size()
          << ") plus loop/retained variable count ("
          << loop_vars.size() + retained_vars.size() << ")";
      return mlir::failure();
    }
  }
  unsigned pin_dim_base = static_cast<unsigned>(window_ivs.size());

  LDBG(1) << "materializeDirectAccessLoops: seeded " << window_ivs.size()
          << " already-pinned window iv(s), pin_dim_base=" << pin_dim_base
          << " is_indirect_load=" << is_indirect_load
          << " is_indirect_store=" << is_indirect_store;

  for (unsigned m = 0; m < loop_vars.size(); ++m) {
    unsigned target_base_dim = loop_vars[m].base_dim;
    LDBG(1) << "materializeDirectAccessLoops: loop var m=" << m << " of "
            << loop_vars.size() << " target_base_dim=" << target_base_dim;

    // Re-locate the variable's current intermediate-list index: dropping
    // earlier variables shifts indices, so this must be re-derived on the
    // live (possibly-already-mutated) op rather than assumed.
    unsigned cur_num_captured = current_op.getCapturedVariables().size();
    mlir::AffineMap target_map =
        mlir::cast<mlir::AffineMapAttr>(
            current_op.getPerDimSubscriptMaps()[target_base_dim])
            .getValue();
    auto referenced_dim = findReferencedDim(target_map, cur_num_captured);
    if (mlir::failed(referenced_dim)) {
      current_op.emitError()
          << "could not re-locate the intermediate variable for $base "
             "dimension "
          << target_base_dim;
      return mlir::failure();
    }
    unsigned absorbed_interm_idx = *referenced_dim - cur_num_captured;
    unsigned absorbed_unified_dim = cur_num_captured + absorbed_interm_idx;

    auto trip_count = getTripCount(current_op.getVariablesSpaceSet().getValue(),
                                   absorbed_interm_idx);
    if (!trip_count) {
      current_op.emitError()
          << "could not extract constant trip count for intermediate "
             "variable index "
          << absorbed_interm_idx << " from variables_space_set";
      return mlir::failure();
    }
    int64_t N = *trip_count;

    auto tile_dim_expr = mlir::dyn_cast<mlir::AffineDimExpr>(
        current_op.getVariablesSpaceOrder().getResult(absorbed_interm_idx));
    if (!tile_dim_expr) {
      current_op.emitError()
          << "variables_space_order result " << absorbed_interm_idx
          << " is not a plain dim expression; cannot determine access tile "
             "dimension to drop";
      return mlir::failure();
    }
    unsigned tile_dim_to_drop = tile_dim_expr.getPosition();

    LDBG(1) << "  absorbed_interm_idx=" << absorbed_interm_idx
            << " trip_count(N)=" << N
            << " tile_dim_to_drop=" << tile_dim_to_drop;

    // ── Splice boundary + emit the scf.for ────────────────────────────────
    // IndirectAddrBufLegalization always materializes at least the entry
    // scf.for before this pass runs, so current_op's block is always some
    // enclosing loop's body — the splice boundary is therefore always that
    // entire body, never the narrower def-use-reachable subset
    // findSpliceBoundary computes. This matters because that enclosing
    // loop's body may also hold an ind_addr_buf fill chain relocated behind
    // an scf.if by IndirectAddrBufLegalization's entry-loop materialization:
    // that fill chain is independent of (not def-use-reachable from) the
    // variable being absorbed here, so a reachability-based boundary would
    // leave the scf.if as a sibling of the new loop instead of nested inside
    // it, and downstream passes require perfectly nested loops.
    mlir::Block* src_block = current_op->getBlock();
    auto enclosing_for =
        mlir::dyn_cast_if_present<mlir::scf::ForOp>(src_block->getParentOp());
    if (!enclosing_for) {
      current_op.emitError()
          << "could not find an enclosing scf.for loop for direct-access "
             "loop "
          << m;
      return mlir::failure();
    }

    mlir::Operation* splice_begin_op = &src_block->front();
    mlir::Operation* splice_end_op =
        &*std::prev(src_block->without_terminator().end());

    // enclosing_for carries no iter_args (its ind_addr_buf fill chain writes
    // into a hoisted, loop-invariant IAB memref by side effect rather than
    // threading an SSA value through the loop), so moving its entire body —
    // relocated fill chain included — into the new loop leaves nothing to
    // re-thread: src_block's terminator is a bare, argument-less scf.yield
    // both before and after the splice.
    mlir::OpBuilder builder(splice_begin_op);
    mlir::Value c0 =
        mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult();
    mlir::Value cN =
        mlir::arith::ConstantIndexOp::create(builder, loc, N).getResult();
    mlir::Value c1 =
        mlir::arith::ConstantIndexOp::create(builder, loc, 1).getResult();
    auto for_op = mlir::scf::ForOp::create(builder, loc, c0, cN, c1);
    auto parallel_attr =
        mlir::ktdf::LoopTypeAttr::get(ctx, mlir::ktdf::LoopType::ParallelLoop);
    for_op->setAttr("loop_type", parallel_attr);
    mlir::Value i_m = for_op.getInductionVar();

    mlir::Block* dst_block = for_op.getBody();
    mlir::Block::iterator splice_begin = splice_begin_op->getIterator();
    mlir::Block::iterator splice_end = std::next(splice_end_op->getIterator());
    dst_block->getOperations().splice(dst_block->begin(),
                                      src_block->getOperations(), splice_begin,
                                      splice_end);

    mlir::Block* const scope_block = dst_block;

    LDBG(1) << "  emitted parallel scf.for trip_count=" << N
            << " (enclosing loop at " << enclosing_for.getLoc() << ")";

    // ── Access Tile pin-dim continuation ────────────────────────────────
    window_ivs.push_back(i_m);
    unsigned pin_dim = pin_dim_base + m;

    llvm::SmallVector<mlir::Operation*> pre_narrowed;
    std::optional<DeferredExpand> deferred_out;
    if (is_indirect_load) {
      deferred_out = pinDestAccessTile(current_op, scope_block, pin_dim,
                                       window_ivs, pre_narrowed, loc, ctx);
    }
    if (is_indirect_store) {
      pinSourceAccessTile(current_op, scope_block, pin_dim, window_ivs,
                          pre_narrowed, loc, ctx);
    }

    LDBG(1) << "  pinned access tile dim=" << pin_dim << ", "
            << pre_narrowed.size() << " op(s) pre-narrowed";

    // ── Rebuild construct_indirect_access_tile ────────────────────────────
    mlir::IntegerSet new_vars_set = dropDimFromIntegerSet(
        current_op.getVariablesSpaceSet().getValue(), absorbed_interm_idx);
    mlir::AffineMap new_vars_order = dropDimFromAffineMap(
        current_op.getVariablesSpaceOrder(), absorbed_interm_idx);
    unsigned new_num_interm =
        static_cast<unsigned>(current_op.getIntermediateVariables().size()) - 1;

    // per_dim_subscript_maps: the absorbed slot maps to the newly inserted
    // captured position (unlike IndirectAddrBufLegalization's IAB entry
    // absorption, this variable *is* referenced by per-dim maps, so it cannot
    // become a dangling constant 0).
    llvm::SmallVector<mlir::Attribute> new_subscript_maps;
    for (mlir::Attribute attr : current_op.getPerDimSubscriptMaps()) {
      mlir::AffineMap old_map =
          mlir::cast<mlir::AffineMapAttr>(attr).getValue();
      unsigned old_num_dims = old_map.getNumDims();
      llvm::SmallVector<mlir::AffineExpr> dim_repls(old_num_dims);
      for (unsigned d = 0; d < old_num_dims; ++d) {
        if (d < cur_num_captured)
          dim_repls[d] = mlir::getAffineDimExpr(d, ctx);
        else if (d == absorbed_unified_dim)
          dim_repls[d] = mlir::getAffineDimExpr(cur_num_captured, ctx);
        else if (d < absorbed_unified_dim)
          dim_repls[d] = mlir::getAffineDimExpr(d + 1, ctx);
        else
          dim_repls[d] = mlir::getAffineDimExpr(d, ctx);
      }
      llvm::SmallVector<mlir::AffineExpr> new_results;
      for (unsigned r = 0; r < old_map.getNumResults(); ++r)
        new_results.push_back(
            old_map.getResult(r).replaceDimsAndSymbols(dim_repls, {}));
      new_subscript_maps.push_back(
          mlir::AffineMapAttr::get(mlir::AffineMap::get(
              old_num_dims, old_map.getNumSymbols(), new_results, ctx)));
    }

    llvm::SmallVector<mlir::Value> captured_vars(
        current_op.getCapturedVariables().begin(),
        current_op.getCapturedVariables().end());
    captured_vars.push_back(i_m);

    mlir::Value base_val = current_op.getBase();
    mlir::Value iab_memref_val = current_op.getIndAddrBufMemref();
    auto iab_positions = mlir::DenseI32ArrayAttr::get(
        ctx, current_op.getIndAddrBufDimPositions());

    auto cur_result_type = mlir::cast<mlir::ktdp::AccessTileType>(
        current_op.getResult().getType());
    llvm::SmallVector<int64_t> new_result_shape =
        dropShapeDim(cur_result_type.getShape(), tile_dim_to_drop);
    auto new_result_type = mlir::ktdp::AccessTileType::get(
        new_result_shape, cur_result_type.getElementType());

    mlir::OpBuilder replace_builder(current_op);
    auto new_op = mlir::ktdp_lowering::ConstructIndirectAccessTileOp::create(
        replace_builder, loc, new_result_type, base_val, iab_memref_val,
        iab_positions, mlir::ArrayAttr::get(ctx, new_subscript_maps),
        captured_vars, new_num_interm, new_vars_order, new_vars_set);

    current_op.getResult().replaceAllUsesWith(new_op.getResult());
    current_op.erase();
    current_op = new_op;

    LDBG(1) << "  rebuilt construct_indirect_access_tile at "
            << current_op.getLoc() << " new_num_interm=" << new_num_interm;

    // ── Propagate narrowing through the downstream compute chain ─────────
    // iab_rank = -1: no in-scope memref can have a negative rank, so the
    // ConstructMemoryViewOp branch never fires (this pass never touches
    // $ind_addr_buf_memref).
    if (mlir::failed(propagateNarrowing(
            new_op, tile_dim_to_drop, llvm::ArrayRef<mlir::Value>{i_m},
            /*iab_rank=*/-1, pre_narrowed, loc, ctx)))
      return mlir::failure();
    LDBG(1) << "  propagateNarrowing succeeded";

    if (deferred_out) {
      mlir::OpBuilder es_b(
          deferred_out->src_val.getDefiningOp()
              ? deferred_out->src_val.getDefiningOp()->getNextNode()
              : deferred_out->store.getOperation());
      auto out_expand = mlir::tensor::ExpandShapeOp::create(
          es_b, deferred_out->store.getLoc(), deferred_out->pinned_type,
          deferred_out->src_val, deferred_out->reassoc);
      deferred_out->store.getDataTileMutable().assign(out_expand.getResult());
      LDBG(1) << "  applied deferred destination expand_shape at "
              << deferred_out->store.getLoc();
    }
  }

  LDBG(1) << "materializeDirectAccessLoops: done, materialized "
          << loop_vars.size() << " loop(s), final op at "
          << current_op.getLoc();
  return mlir::success();
}

struct IndirectAccessLoopMaterializationPass
    : public impl::IndirectAccessLoopMaterializationPassBase<
          IndirectAccessLoopMaterializationPass> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    LDBG(1) << "===== " PASS_NAME " =====";

    llvm::SmallVector<mlir::ktdp_lowering::ConstructIndirectAccessTileOp, 4>
        indirect_ops;

    for (auto func : module.getOps<mlir::func::FuncOp>()) {
      mlir::ktdp_lowering::ConstructIndirectAccessTileOp func_indirect_op;
      mlir::WalkResult walk_res =
          func.walk([&](mlir::ktdp_lowering::ConstructIndirectAccessTileOp op) {
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

    LDBG(1) << "found " << indirect_ops.size()
            << " construct_indirect_access_tile op(s)";

    // No-op: return early when no indirect access tiles are present.
    if (indirect_ops.empty()) return;

    // Find the device's indirect address buffer resource, needed below to
    // locate the load/store units co-located with it — the IAB's own
    // capacity (num_entries) is not needed by this pass at all; that's a
    // separate resource property used for window sizing by
    // IndirectAddrBufLegalization.
    auto declaration =
        mlir::ktdf_arch::findDeviceDeclarationFor(indirect_ops.front());
    if (!declaration) {
      indirect_ops.front()->emitError(
          "could not find device declaration for indirect access loop "
          "materialization");
      signalPassFailure();
      return;
    }
    mlir::ktdf_arch::DeviceRef device(declaration, getAnalysisManager());

    LDBG(1) << "found device declaration at " << declaration->getLoc();

    auto& resource_kinds =
        device.getOrCreateView<mlir::ktdf_arch::ResourceKinds>();

    mlir::ktdf_arch::ResourceKinds::Kind iab_kind;
    for (const mlir::ktdf_arch::ResourceKinds::Kind& kind : resource_kinds) {
      if (resource_kinds
              .getFeature<mlir::ktdf_arch::feature::IndirectAddressBuffer>(
                  kind)) {
        iab_kind = kind;
        break;
      }
    }
    if (!iab_kind) {
      declaration->emitWarning(
          "device has no indirect address buffer resource; skipping "
          "indirect access loop materialization");
      return;
    }

    // The unit(s) that actually perform the hardware indirect transfer are
    // whichever load/store-capable exec units are co-located with the IAB
    // (e.g. MNILU for gather, MNISU for scatter) — that's where the real
    // minimum transfer size (access_granularity) lives.
    auto load_kind = findColocatedUnit<mlir::ktdf_arch::feature::Load>(
        resource_kinds, iab_kind);
    auto store_kind = findColocatedUnit<mlir::ktdf_arch::feature::Store>(
        resource_kinds, iab_kind);

    llvm::DenseMap<mlir::Attribute, mlir::Attribute> mem_space_map;
    if (const auto mapping =
            device.get().getAttrOfType<mlir::ktdf_arch::MapAttr>(
                "mem_space_mapping");
        mapping) {
      mem_space_map.insert_range(mapping);
    }

    TransferSizeInfo transfer_info{std::move(mem_space_map),
                                   load_kind.getExemplar(),
                                   store_kind.getExemplar()};

    LDBG(1) << "found IAB-colocated load unit="
            << static_cast<bool>(transfer_info.load_unit)
            << " store unit=" << static_cast<bool>(transfer_info.store_unit);

    for (auto op : indirect_ops) {
      LDBG(1) << "materializing direct-access loops for "
                 "construct_indirect_access_tile at "
              << op.getLoc();
      if (mlir::failed(materializeDirectAccessLoops(op, transfer_info))) {
        LDBG(1) << "  materializeDirectAccessLoops FAILED";
        signalPassFailure();
        return;
      }
      LDBG(1) << "  materializeDirectAccessLoops succeeded";
      LDBG(2) << "IR after materializeDirectAccessLoops:\n" << module;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass>
scheduler::createIndirectAccessLoopMaterializationPass() {
  return std::make_unique<IndirectAccessLoopMaterializationPass>();
}
