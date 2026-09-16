//===-- Passes.td ------------------------------------------*- tablegen -*-===//
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
// This file declares all the scheduler passes.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_TRANSFORMS_PASSES_TD
#define DATAFLOW_SCHEDULER_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def PathExpansionPass : Pass<"path-expansion", "mlir::ModuleOp"> {
  let summary = "Expand data movement paths to match architecture graph";
  let description = [{
    This pass legalizes data movement in ktdf.pipeline operations by expanding
    paths to match the architecture graph. It inserts intermediate transfer
    stages and resources where needed to ensure all data movement follows legal
    hardware routing.
  }];
  let constructor = "::scheduler::createPathExpansionPass()";
}

def NormalizeGridTo1DPass : Pass<"normalize-grid-to-1d", "mlir::ModuleOp"> {
  let summary = "Flatten N-D grids to 1-D and delinearize compute tile ids";
  let description = [{
    Rewrites each function's `grid` attribute from an N-D array to a single
    1-D entry equal to the row-major product of its dimensions, and converts
    the matching multi-result `ktdp.get_compute_tile_id` into a single-result
    flat id whose original per-dimension coordinates are reconstructed with
    row-major `arith.divui`/`arith.remui`. 1-D grids pass through unchanged.
  }];
  let constructor = "::scheduler::createNormalizeGridTo1DPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::ktdp::KtdpDialect"
  ];
}

def NormalizeSCFForLoopsPass : Pass<"normalize-scf-for-loops", "mlir::ModuleOp"> {
  let summary = "Normalize scf.for loops to lb=0, step=1";
  let description = [{
    Rewrites every scf.for loop in the module so that it has lower bound 0 and
    step 1.  The induction variable is denormalized inside the body so that all
    existing uses continue to see the original iteration values:

      denormalized_iv = orig_lb + normalized_iv * orig_step

    The denormalization is emitted as an affine.apply immediately inside the
    loop body, then sunk to just before each use to minimise register pressure.
  }];
  let constructor = "::scheduler::createNormalizeSCFForLoopsPass()";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::scf::SCFDialect"
  ];
}

def TileSCFForLoopsPass : Pass<"tile-scf-for-loops", "mlir::ModuleOp"> {
  let summary = "Tile perfectly nested scf.for loops";
  let description = [{
    This pass tiles perfectly nested scf.for loops with four execution modes:
    
    1. Static mode with custom tiling (tile-sizes specified, fallback-scf-tiling=false):
       Uses command-line tile sizes with custom normalized tiling scheme
    2. Static mode with fallback SCF tiling (tile-sizes specified, fallback-scf-tiling=true):
       Uses command-line tile sizes with SCF's standard tiling (for testing)
    3. Dynamic mode with custom tiling (tile-sizes not specified, fallback-scf-tiling=false):
       Uses LoopTilingAnalysis to identify candidates and custom normalized tiling
    4. Dynamic mode with fallback SCF tiling (tile-sizes not specified, fallback-scf-tiling=true):
       Uses LoopTilingAnalysis with SCF's standard tiling (for testing)
    
    Custom tiling scheme example (uses ktdf.tiling.derive_size and ktdf.tiling.linearize_index):
    ```
    // Original normalized loops (lb=0, step=1)
    scf.for %i = %c0 to %c23 step %c1 {
      scf.for %j = %c0 to %c12 step %c1 {
        %sum = arith.addi %i, %j : index
      }
    }
    
    // After custom normalized tiling with tile sizes [5, 5]
    %n_tiles_i = arith.ceildivui %c23, %c5 : index
    scf.for %tile_i = %c0 to %n_tiles_i step %c1 {
      %n_tiles_j = arith.ceildivui %c12, %c5 : index
      scf.for %tile_j = %c0 to %n_tiles_j step %c1 {
        %tile_size_i = ktdf.tiling.derive_size [%tile_i : %c5], total_size = %c23
        %tile_size_j = ktdf.tiling.derive_size [%tile_j : %c5], total_size = %c12
        scf.for %point_i = %c0 to %tile_size_i step %c1 {
          scf.for %point_j = %c0 to %tile_size_j step %c1 {
            %i = ktdf.tiling.linearize_index[%tile_i : %c5], [%point_i : %c1]
            %j = ktdf.tiling.linearize_index[%tile_j : %c5], [%point_j : %c1]
            %sum = arith.addi %i, %j : index
          }
        }
      }
    }
    
    // After applying another round of custom tiling [3, 3] to innermost loops
    %n_tiles_i = arith.ceildivui %c23, %c5 : index
    scf.for %tile_i = %c0 to %n_tiles_i step %c1 {
      %n_tiles_j = arith.ceildivui %c12, %c5 : index
      scf.for %tile_j = %c0 to %n_tiles_j step %c1 {
        %tile_size_i = ktdf.tiling.derive_size [%tile_i : %c5], total_size = %c23
        %tile_size_j = ktdf.tiling.derive_size [%tile_j : %c5], total_size = %c12
        %n_tiles_pi = arith.ceildivui %tile_size_i, %c3 : index
        scf.for %tile_pi = %c0 to %n_tiles_pi step %c1 {
          %n_tiles_pj = arith.ceildivui %tile_size_j, %c3 : index
          scf.for %tile_pj = %c0 to %n_tiles_pj step %c1 {
            %tile_size_pi = ktdf.tiling.derive_size [%tile_pi : %c3], total_size = %tile_size_i
            %tile_size_pj = ktdf.tiling.derive_size [%tile_pj : %c3], total_size = %tile_size_j
            scf.for %point_pi = %c0 to %tile_size_pi step %c1 {
              scf.for %point_pj = %c0 to %tile_size_pj step %c1 {
                %point_i = ktdf.tiling.linearize_index[%tile_pi : %c3], [%point_pi : %c1]
                %point_j = ktdf.tiling.linearize_index[%tile_pj : %c3], [%point_pj : %c1]
                %i = ktdf.tiling.linearize_index[%tile_i : %c5], [%point_i : %c1]
                %j = ktdf.tiling.linearize_index[%tile_j : %c5], [%point_j : %c1]
                %sum = arith.addi %i, %j : index
              }
            }
          }
        }
      }
    }
    ```
    
    Standard SCF tiling example (fallback-scf-tiling=true):
    ```
    scf.for %i = 0 to 100 step 1 {
      scf.for %j = 0 to 100 step 1 { ... }
    }
    ```
    becomes:
    ```
    scf.for %i0 = 0 to 100 step 32 {
      scf.for %j0 = 0 to 100 step 32 {
        scf.for %i1 = %i0 to min(%i0+32, 100) step 1 {
          scf.for %j1 = %j0 to min(%j0+32, 100) step 1 { ... }
        }
      }
    }
    ```
  }];
  let constructor = "::scheduler::createTileSCFForLoopsPass()";
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t",
               "Tile sizes for each loop dimension (for testing only)">,
    Option<"normalizeLoops", "normalize-loops", "bool",
           /*default=*/"true",
           "Normalize tiled loops to start at 0 and add outer IV to uses">,
    Option<"useFallbackSCFTiling", "fallback-scf-tiling", "bool",
           /*default=*/"false",
           "Use SCF's standard tiling instead of custom normalized tiling (for testing)">
  ];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::ktdf::KTDFDialect",
    "mlir::scf::SCFDialect"
  ];
}

def StripMineSCFForLoopsPass : Pass<"strip-mine-scf-for-loops", "mlir::ModuleOp"> {
  let summary = "Strip-mine scf.for loops";
  let description = [{
    This pass performs strip-mining on scf.for loops.
    Strip-mining splits a loop into an outer loop and an inner loop,
    where the outer loop iterates over blocks and the inner loop
    iterates within each block.
    
    When strip-mine-size is specified on the command line, it is used as a
    static value for all loops (testing mode). The size must be positive.
    
    When strip-mine-size is not specified, the pass uses LoopTilingAnalysis to
    identify strip-mining candidates and materializes ktdf.tiling.reserve_size 
    ops with metadata-based attributes for each candidate loop.
    
    Example transformation:
    ```
    scf.for %i = 0 to 100 step 1 { ... }
    ```
    becomes:
    ```
    scf.for %i0 = 0 to 100 step 32 {
      scf.for %i1 = %i0 to min(%i0+32, 100) step 1 { ... }
    }
    ```
  }];
  let constructor = "::scheduler::createStripMineSCFForLoopsPass()";
  let options = [
    ListOption<"stripMineSize", "strip-mine-size", "int64_t",
               "Strip-mine factor (block size, for testing only)">,
    Option<"normalizeLoops", "normalize-loops", "bool",
           /*default=*/"true",
           "Normalize loops to start at 0 and add outer IV to uses">
  ];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::ktdf::KTDFDialect",
    "mlir::scf::SCFDialect"
  ];
}

def ParallelizeLoopsAcrossInstancesPass
    : Pass<"parallelize-loops-across-instances", "mlir::ModuleOp"> {
  let summary = "Map parallel loops onto hardware instances via ktdf.parallel";
  let description = [{
    For each ktdf.pipeline whose work is corelet-local (no core-shared
    units in the union of its immediate stages' applicable_units), find
    the outermost parallel-annotated scf.for in the pipeline's enclosing
    scope whose statically-known trip count permits a split across
    `num_instances` (preferring divisible trips; falling back to imbalanced
    when none qualifies), and rewrite that scf.for into a ktdf.parallel op
    with `distribute(num_instances = N)`. `N` is sourced from
    SchedulerExtContext::getNumCorelets() at pass construction; the
    pass itself is target-neutral.

    Body legality (cross-iteration dependencies, L1 sharing) is trusted
    via the loop_type=parallel_loop annotation set by upstream lowering.
    Reduction loops (loop_type=reduction_loop or unannotated) are not
    candidates.
  }];
  let constructor = "::scheduler::createParallelizeLoopsAcrossInstancesPass()";
  let dependentDialects = ["mlir::ktdf::KTDFDialect", "mlir::scf::SCFDialect"];
}

def DoubleBufferingPass
    : Pass<"double-buffering", "mlir::ModuleOp"> {
  let summary = "Double-buffer scratchpad memrefs handed off between sibling stages";
  let description = [{
    For each ktdf.pipeline whose enclosing scope contains at least one scf.for
    and which has no pre-existing modulo attribute, find scratchpad memrefs
    yielded by the pipeline's ktdf.private region that are written by exactly
    one sibling stage and read by one or more sibling stages (via
    ktdf.data_transfer ops). For each such buffer, allocate two physical copies
    just outside the pipeline scope, insert one shared ktdf.buffer_phase and
    per-buffer ktdf.select_memref before the pipeline, rewrite the pipeline's
    references to use the selected memrefs, and set modulo(size: 2) on the
    pipeline.

    Memory-space eligibility is determined by querying the MemoryTree to check
    if the memory space is per-core scratchpad memory (e.g., L1 on Spyre).

    The pass leaves ktdf.private's signature consistent: dropped result slots
    are removed from the op and from ktdf.private_yield, and orphaned
    memref.alloc ops in the private region are erased.
  }];
  let constructor = "::scheduler::createDoubleBufferingPass()";
  let dependentDialects = [
      "mlir::ktdf::KTDFDialect",
      "mlir::ktdp::KtdpDialect",
      "mlir::scf::SCFDialect",
      "mlir::memref::MemRefDialect"];
}

def AffineMinCanonicalizationPass : Pass<"affine-min-canonicalization", "mlir::ModuleOp"> {
  let summary = "Canonicalize arith.addi + arith.minsi + affine.apply to affine.min";
  let description = [{
    This pass transforms the pattern:
      %add = arith.addi %iv, %step
      %min = arith.minsi %add, %ub
      %result = affine.apply ()[%iv, %min] -> (-iv + min)
    
    Into the canonical form:
      %result = affine.min (%iv)[%ub, %step] -> (ub - iv, step)
    
    This simplifies the IR and makes it easier for subsequent passes like
    address-assignment to analyze bounds. The transformation is mathematically
    equivalent: both compute min(ub - iv, step).
  }];
  let constructor = "::scheduler::createAffineMinCanonicalizationPass()";
  let dependentDialects = [
      "mlir::affine::AffineDialect",
      "mlir::arith::ArithDialect"];
}

def AddressAssignmentPass : Pass<"address-assignment", "mlir::ModuleOp"> {
  let summary = "Analyze memory allocation and assign addresses";
  let description = [{
    This pass collects all memref.alloc operations in the module and groups
    them by memory space (L1, DDR), analyses their sizes and assigns addresses.

  }];

  let constructor = "::scheduler::createAddressAssignmentPass()";
  let dependentDialects = [
      "mlir::ktdf::KTDFDialect",
      "mlir::memref::MemRefDialect",
      "mlir::affine::AffineDialect"];

  let statistics = [
    Statistic<"num_processed", "num-processed", "Number of processed allocations">,
    Statistic<"num_failed", "num-failed", "Number of failed allocations">
  ];
}

def ScalarBroadcastLegalizationPass
    : Pass<"scalar-broadcast-legalization", "mlir::ModuleOp"> {
  let summary = "Legalize scalar broadcast inputs to linalg.generic compute ops";
  let description = [{
    Detects linalg.generic operations in compute stages that have input
    indexing maps with a constant in the vector dimension (scalar broadcast).
    When the data path hardware cannot legally carry sub-vector transfers or
    the compute unit does not support scalar broadcast semantics natively,
    this pass legalizes the path by:
      1. Widening memref-side transfers to the full vector granularity.
      2. Marking the L1->FIFO transfer with transfer_mode="splat".
      3. Updating read_from_fifo result type and linalg.generic indexing map
         to use the full vector identity map.
  }];
  let constructor = "::scheduler::createScalarBroadcastLegalizationPass()";
}

def IndirectAddrBufFillLegalizationPass
    : Pass<"indirect-addr-buf-fill-legalization", "mlir::ModuleOp"> {
  let summary = "Stage indirect address buffer fills through a reachable memory";
  let description = [{
    Runs after path expansion. A `ktdf.data_transfer` that fills an indirect
    address buffer (a memory node carrying
    `ktdf_arch.feature.indirect_address_buffer`) is executed by the unit its
    stage is mapped to, so it can only read from a memory that has a datapath
    into that unit. A store unit, for instance, reads from L1 and cannot fill
    the buffer straight from global memory.

    For every illegal fill this pass:
      1. Allocates a buffer in the memory the filling unit does read from,
         adding it to the pipeline's `ktdf.private` resources.
      2. Copies the index data into that buffer from an earlier stage whose
         unit has a datapath from the original memory to the staging memory,
         under the same guard the fill carries.
      3. Repoints the fill at the staging buffer.

    Fills whose source memory already feeds the filling unit are left alone.
  }];
  let constructor =
      "::scheduler::createIndirectAddrBufFillLegalizationPass()";
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect"
  ];
}

def EnsureDeviceDeclarationPass
    : Pass<"ensure-device-declaration", "mlir::ModuleOp"> {
  let summary = "Ensure that there is a ktdf_arch.device declaration";
  let description = [{
    Detect if there is a ktdf_arch.device declaration. 
    If not, insert a device declaration based on the option given to this pass.
    And also, ensure ktdf_arch::DeviceManager is instantiated.
  }];
  let constructor = "::scheduler::createEnsureDeviceDeclarationPass()";
  let options = [
    Option<"deviceFileName", "device-filename", "std::string",
           /*default=*/"",
           "Path to the file containing the architecture description">,
    Option<"deviceName", "device-name", "std::string",
           /*default=*/"",
           "Name of the device definition. If unspecified, the only device in the file is chosen">
  ]; 
  let dependentDialects = [
    "mlir::ktdf_arch::KTDFArchDialect"
  ];
}

def MaterializeRegistersPass : Pass<"materialize-registers", "mlir::ModuleOp"> {
  let summary = "Make the registers a generic body uses real, in front of it";
  let description = [{
    Visits every nested `linalg.generic` and makes the registers it uses real
    in front of it: an allocation the body carries, and an `arith.constant`
    with a `ktdf_arch.maps_to` attribute, which needs an allocation and a fill
    of its own.

    This requires the cardinality of the `linalg.generic` iteration domain to
    be statically known, and all consumers of the registers to be either
    `memref.load`, `memref.store` or `ktdf.opaque`. Register sizes are
    increased to match the iteration domain, but scalar users remain fixed to
    the first element of the allocated segment. A store of a value the body
    does not compute per element fills every element instead.

    Example:
    ```mlir
    linalg.generic ins(%in: tensor<1x64xf16>) ... {
    ^bb0(%arg0: f16);
      %cst = arith.constant {ktdf_arch.maps_to = "SFU_REG"} 1.0 : f16
      %reg = memref.alloc() : memref<f16>
      memref.store %value, %reg[] : ...
      %ktdf.opaque ins(%arg0, %cst: ...) outs(%reg: ...)
    }
    ```
    
    Is transformed into:
    ```mlir
    %cst0 = arith.constant 0 : index
    %cst = arith.constant {ktdf_arch.maps_to = "SFU_REG"} 1.0 : f16
    %cst_reg = memref.alloc() : memref<64xf16, "SFU_REG">
    linalg.fill ins(%cst: ...) outs(%cst_reg) ...
    %reg = memref.alloc() : memref<64xf16>
    linalg.generic ins(%in: tensor<1x64xf16>) ... {
    ^bb0(%arg0: f16);
      memref.store %value, %reg[%cst0] : ...
      %ktdf.opaque ins(%arg0, %cst_reg: ...) outs(%reg: ...)
    }
    ```
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::ktdf::KTDFDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def ApplyDevicePatternsPass : Pass<"apply-device-patterns", "mlir::func::FuncOp"> {
  let summary = "Applies patterns defined in the `ktdf_arch.device`";
  let description = [{
    Given a root operation that is mapped to a `ktdf_arch.device`, this pass
    applies all the (enabled groups of) patterns defined within the device to
    the IR nested below the root operation.
  }];

  let options = [
    ListOption<"enabled_groups", "groups", "std::string", 
      "List of pattern groups to enable, or empty for ungrouped patterns only">
  ];

  let dependentDialects = [
    "mlir::agen::AgenDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::ktdf_arch::KTDFArchDialect",
    "mlir::memref::MemRefDialect",
    "mlir::pdl::PDLDialect",
    "mlir::pdl_interp::PDLInterpDialect",
    "mlir::vectorchain::VectorChainDialect"
  ];
}

def HoistInvariantsPass : Pass<"hoist-invariants", "mlir::func::FuncOp"> {
  let summary = "Hoist invariants out of pipelines and loops";
  let description = [{
    Given a pure operation that is invariant under the pipeline or loop 
    structure it is contained in, hoists the invariant to the surrounding scope.

    Example:
    ```mlir
    scf.for ... {
      ktdf.pipeline {
        ktdf.stage {
          linalg.generic ... {
          ^bb0(%arg0: i32):
            %cst0 = arith.constant 0 : i32
            arith.addi %cst0, %arg0 : i32
          }
        }
      }
    }
    ```

    Is transformed into:
    ```mlir
    %cst0 = arith.constant 0 : i32
    scf.for ... {
      ktdf.pipeline {
        ktdf.stage {
          linalg.generic ... {
          ^bb0(%arg0: i32):
            arith.addi %cst0, %arg0 : i32
          }
        }
      }
    }
    ```
  }];

  let statistics = [
    Statistic<"num_hoisted", "num-hoisted", "Number of hoisted operations">,
  ];
}

def HoistConstantStoragePass : Pass<"hoist-constant-storage", "mlir::func::FuncOp"> {
  let summary = "Hoists memory allocations with constant contents";
  let description = [{
    Given an allocation that is used to hold a constant value, hoists the 
    allocation and its write to the surrounding scope. If the write is in a
    pipeline, it will be hoisted into a sub-pipeline.

    Example:
    ```mlir
    scf.for ... {
      ktdf.pipeline {
        ktdf.stage {
          %cst = arith.constant 1.0 : f16
          %alloc = memref.alloc() : memref<f16, "UNIT">
          memref.store %cst, %alloc : memref<f16, "UNIT">
          linalg.generic ...
            ins(%alloc: memref<f16, "UNIT">)
            outs(...)
        } {applicable_units = ["UNIT"]}
      }
    }
    ```

    Is transformed into:
    ```mlir
    %cst = arith.constant 1.0 : f16
    %alloc = memref.alloc() : memref<f16, "UNIT">
    ktdf.pipeline {
      ktdf.stage {
        memref.store %cst, %alloc : memref<f16, "UNIT">
      } {applicable_units = ["UNIT"]}
    }
    scf.for ... {
      ktdf.pipeline {
        ktdf.stage {
          linalg.generic ...
            ins(%alloc: memref<f16, "UNIT">)
            outs(...)
        } {applicable_units = ["UNIT"]}
      }
    }
    ```
  }];

  let dependentDialects = [
    "mlir::ktdf::KTDFDialect"
  ];

  let statistics = [
    Statistic<"num_hoisted", "num-hoisted", "Number of hoisted constant storage allocations">,
  ];
}

def IndirectAddrBufLegalizationPass
    : Pass<"indirect-addr-buf-legalization", "mlir::ModuleOp"> {
  let summary = "Legalize indirect address buffer usage";
  let description = [{
    Materializes the ind_addr_buf window loop and per-entry loop required by
    the hardware indirect address buffer constraint.
    Outer scf.for over ind_addr_buf windows (annotated parallel), then inner
    scf.for over entries; the ind_addr_buf fill only runs on the entry loop's
    first iteration, guarded by an scf.if on the entry induction variable,
    with the ind_addr_buf memref view hoisted once above the entire
    window/entry loop nest instead of threaded as an scf.for iter-arg.
    No-op when no ktdp_lowering.construct_indirect_access_tile is present.
  }];
  let constructor = "::scheduler::createIndirectAddrBufLegalizationPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::ktdf::KTDFDialect"
  ];
}

def IndirectAccessLoopMaterializationPass
    : Pass<"indirect-access-loop-materialization", "mlir::ModuleOp"> {
  let summary = "Materialize loops for non-contiguous direct subscripts of "
                "indirect access tiles";
  let description = [{
    Inspects the remaining direct-subscript intermediate variables of each
    ktdp_lowering.construct_indirect_access_tile. A variable is materialized
    into a scf.for loop when its dimension fails the dense-packing condition
    (stride[k] != size[k+1]*stride[k+1], or stride != 1 for the innermost
    dimension of $base) OR the full-coverage condition (variables_space_set
    bound < dimension size). Variables that satisfy both conditions are
    retained so that ConstructThreeStagePipeline can read the per-transfer
    tile size. Materialized loops are annotated parallel_loop.
    No-op when all remaining variables satisfy both conditions, or when no
    ktdp_lowering.construct_indirect_access_tile is present.
  }];
  let constructor = "::scheduler::createIndirectAccessLoopMaterializationPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::ktdf::KTDFDialect"
  ];
}

#endif // DATAFLOW_SCHEDULER_TRANSFORMS_PASSES_TD
