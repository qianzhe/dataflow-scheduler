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

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "indirect-addr-buf-legalization"
#define DEBUG_TYPE PASS_NAME

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_INDIRECTADDRBUFLEGALIZATIONPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

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

    // TODO(sub-step-2a): materialize outer scf.for over ind_addr_buf windows
    // (max 32 entries per window); narrow address-load access tile to one row
    // slice; remove absorbed intermediate variable from intermediate_variables
    // and variables_space_set; annotate loop with
    // loop_type = #ktdf.loop_type<parallel_loop>.

    // TODO(sub-step-2b): materialize inner scf.for over ind_addr_buf entries;
    // gate ind_addr_buf fill with scf.if (%i2 == 0); thread ind_addr_buf
    // memref view as scf.for iter-arg (sentinel initial value outside, real
    // view yielded from scf.if); remove absorbed intermediate variable from
    // intermediate_variables and variables_space_set.
  }
};

}  // namespace

std::unique_ptr<mlir::Pass>
scheduler::createIndirectAddrBufLegalizationPass() {
  return std::make_unique<IndirectAddrBufLegalizationPass>();
}
