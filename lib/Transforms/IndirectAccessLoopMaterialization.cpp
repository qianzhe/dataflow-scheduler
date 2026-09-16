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

#include "dataflow-scheduler/Dialect/KTDF/KTDFAttributes.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFEnums.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Transforms/Utils/IndirectAccessTileNarrowing.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "llvm/ADT/DenseSet.h"
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
    if (dense_packed && full_coverage)
      retained_vars.push_back(info);
    else
      loop_vars.push_back(info);
  }

  llvm::sort(loop_vars, [](const DirectVarInfo& a, const DirectVarInfo& b) {
    return a.base_dim < b.base_dim;
  });
  return mlir::success();
}

/// Snapshot of the output/source descriptor access tile's current state,
/// read (never mutated) before Phase 3's own loop starts.
struct DescriptorPeek {
  unsigned rank;
  // Indices for exactly the leading dims already pinned (extent-1) by a
  // prior pass; does NOT include not-yet-pinned (still full-extent) dims,
  // which must stay defaulted to %c0 by rebuildAccessTilePinned until
  // Phase 3 actually reaches them.
  llvm::SmallVector<mlir::Value> pinned_prefix_indices;
};

/// Read-only mirror of pinOutputDescriptorAT/pinSourceDescriptorAT's
/// discovery walk: locates the output descriptor access tile (indirect load
/// / gather case, discovered downstream of `op`'s result) or the source
/// descriptor access tile (indirect store / scatter case, discovered
/// upstream of the store that writes into `op`'s result), without rebuilding
/// anything. Used once, before Phase 3's own loop, to recover how many
/// descriptor dimensions Phase 2 already pinned (and their subscripts), so
/// Phase 3 can continue the same monotonic dimension numbering without
/// needing to re-derive Phase 2's window/entry induction variables from
/// scratch. The already-pinned leading dims are identified by extent == 1
/// in the discovered tile's own shape (set explicitly by
/// rebuildAccessTilePinned on every prior pin), not by inspecting
/// access_tile_set constraint forms.
std::optional<DescriptorPeek> peekDescriptorAT(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp op,
    mlir::Block* scope_block, bool is_indirect_load) {
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

  if (!found) return std::nullopt;
  DescriptorPeek peek;
  auto ty = mlir::cast<mlir::ktdp::AccessTileType>(found.getResult().getType());
  llvm::ArrayRef<int64_t> shape = ty.getShape();
  peek.rank = static_cast<unsigned>(shape.size());

  unsigned pinned_prefix = 0;
  while (pinned_prefix < shape.size() && shape[pinned_prefix] == 1)
    ++pinned_prefix;

  mlir::OperandRange indices = found.getIndices();
  peek.pinned_prefix_indices.assign(indices.begin(),
                                    indices.begin() + pinned_prefix);
  return peek;
}

/// Materialize one scf.for per direct-subscript intermediate variable of
/// `op` that cannot be serviced by a single hardware indirect transfer,
/// outermost-$base-dimension-first. No-op (returns success without changes)
/// when every remaining variable is retainable.
mlir::LogicalResult materializeDirectAccessLoops(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp op, int64_t iab_size) {
  mlir::MLIRContext* ctx = op.getContext();
  mlir::Location loc = op.getLoc();

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
  if (retained_count < iab_size) {
    op.emitError() << "retained element count (" << retained_count
                   << ") is below the minimum hardware transfer size ("
                   << iab_size << ")";
    return mlir::failure();
  }

  if (base_strides[base_rank - 1] != 1) {
    op.emitError() << "innermost dimension of $base does not have unit "
                      "stride; no contiguous region exists to transfer";
    return mlir::failure();
  }

  // ── Step 3: no-op path ──────────────────────────────────────────────────
  if (loop_vars.empty()) return mlir::success();

  // ── Step 4: materialize, outermost-$base-dimension-first ───────────────
  mlir::ktdp_lowering::ConstructIndirectAccessTileOp current_op = op;

  bool is_indirect_load = false;
  bool is_indirect_store = false;
  for (mlir::Operation* user : op.getResult().getUsers()) {
    if (mlir::isa<mlir::ktdp::LoadOp>(user))
      is_indirect_load = true;
    else if (mlir::isa<mlir::ktdp::StoreOp>(user))
      is_indirect_store = true;
  }

  // Seed window_ivs/pin_dim from whatever Phase 2 already pinned on the
  // output/source descriptor's own access tile (read-only peek; see the
  // "Descriptor-AT pin-dim continuation" design note).
  llvm::SmallVector<mlir::Value> window_ivs;
  if (auto peek = peekDescriptorAT(current_op, current_op->getBlock(),
                                   is_indirect_load)) {
    window_ivs.assign(peek->pinned_prefix_indices.begin(),
                      peek->pinned_prefix_indices.end());
    if (peek->rank !=
        window_ivs.size() + loop_vars.size() + retained_vars.size()) {
      current_op.emitError()
          << "output/source descriptor access tile rank (" << peek->rank
          << ") does not match already-pinned dims (" << window_ivs.size()
          << ") plus loop/retained variable count ("
          << loop_vars.size() + retained_vars.size() << ")";
      return mlir::failure();
    }
  }
  unsigned pin_dim_base = static_cast<unsigned>(window_ivs.size());

  for (unsigned m = 0; m < loop_vars.size(); ++m) {
    unsigned target_base_dim = loop_vars[m].base_dim;

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

    // ── Splice boundary + emit the scf.for ────────────────────────────────
    mlir::Block* src_block = current_op->getBlock();
    auto [splice_begin_op, splice_end_op] =
        computeSpliceBoundary(current_op, /*is_first_loop=*/m == 0);
    if (!splice_begin_op || !splice_end_op) {
      current_op.emitError()
          << "could not determine splice boundary for direct-access loop " << m;
      return mlir::failure();
    }

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

    // ── Descriptor AT pin-dim continuation ────────────────────────────────
    window_ivs.push_back(i_m);
    unsigned pin_dim = pin_dim_base + m;

    llvm::SmallVector<mlir::Operation*> pre_narrowed;
    std::optional<DeferredExpand> deferred_out;
    if (is_indirect_load) {
      deferred_out = pinOutputDescriptorAT(current_op, scope_block, pin_dim,
                                           window_ivs, pre_narrowed, loc, ctx);
    }
    if (is_indirect_store) {
      pinSourceDescriptorAT(current_op, scope_block, pin_dim, window_ivs,
                            pre_narrowed, loc, ctx);
    }

    // ── Rebuild construct_indirect_access_tile ────────────────────────────
    mlir::IntegerSet new_vars_set = dropDimFromIntegerSet(
        current_op.getVariablesSpaceSet().getValue(), absorbed_interm_idx);
    mlir::AffineMap new_vars_order = dropDimFromAffineMap(
        current_op.getVariablesSpaceOrder(), absorbed_interm_idx);
    unsigned new_num_interm =
        static_cast<unsigned>(current_op.getIntermediateVariables().size()) - 1;

    // per_dim_subscript_maps: the absorbed slot maps to the newly inserted
    // captured position (unlike sub-step 2b's IAB entry absorption, this
    // variable *is* referenced by per-dim maps, so it cannot become a
    // dangling constant 0).
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

    // ── Propagate narrowing through the downstream compute chain ─────────
    // iab_rank = -1: no in-scope memref can have a negative rank, so the
    // ConstructMemoryViewOp branch never fires (Phase 3 never touches
    // $ind_addr_buf_memref).
    if (mlir::failed(propagateNarrowing(
            new_op, tile_dim_to_drop, llvm::ArrayRef<mlir::Value>{i_m},
            /*iab_rank=*/-1, pre_narrowed, loc, ctx)))
      return mlir::failure();

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

  return mlir::success();
}

struct IndirectAccessLoopMaterializationPass
    : public impl::IndirectAccessLoopMaterializationPassBase<
          IndirectAccessLoopMaterializationPass> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

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

    // No-op: return early when no indirect access tiles are present.
    if (indirect_ops.empty()) return;

    // Query the hardware IAB size from the architecture specification, same
    // as IndirectAddrBufLegalization: it is also the minimum element count
    // a single hardware indirect transfer must retain.
    auto declaration =
        mlir::ktdf_arch::findDeviceDeclarationFor(indirect_ops.front());
    if (!declaration) {
      indirect_ops.front()->emitError(
          "could not find device declaration for indirect access loop "
          "materialization");
      signalPassFailure();
      return;
    }
    mlir::ktdf_arch::Device device(declaration);

    int64_t iab_size = -1;
    device.getBodyRegion().walk([&](mlir::ktdf_arch::Resource resource) {
      if (iab_size >= 0) return;
      const auto iab_feature =
          resource
              .getFeature<mlir::ktdf_arch::feature::IndirectAddressBuffer>();
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
      if (mlir::failed(materializeDirectAccessLoops(op, iab_size))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass>
scheduler::createIndirectAccessLoopMaterializationPass() {
  return std::make_unique<IndirectAccessLoopMaterializationPass>();
}
