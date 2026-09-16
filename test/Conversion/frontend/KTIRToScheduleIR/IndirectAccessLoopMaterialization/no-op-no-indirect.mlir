// RUN: dataflow-scheduler-opt --mlir-print-local-scope --indirect-access-loop-materialization %s | FileCheck %s

// No ktdp_lowering.construct_indirect_access_tile is present: the pass must
// complete without error and leave the IR unchanged.

// CHECK-LABEL: func.func @no_indirect
// CHECK-NOT:   scf.for
// CHECK:       ktdp.construct_access_tile
// unchanged: no indirect access tile present, so this pass never touches
// access_tile_order/set -- carried over verbatim from the input.
// CHECK-SAME:      access_tile_order = affine_map<(d0) -> (d0)>
// CHECK-SAME:      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
// CHECK:       ktdp.load
// CHECK:       ktdp.store

module @local_schedule_1 {
  func.func @no_indirect() attributes {grid = [1 : index]} {
    %c0 = arith.constant 0 : index
    %desc = ktdp.construct_memory_view %c0,
        sizes: [64], strides: [1]
        {coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>,
         memory_space = #ktdp.memory_space<global>}
        : memref<64xf16>
    %at = ktdp.construct_access_tile %desc[%c0]
        {access_tile_order = affine_map<(d0) -> (d0)>,
         access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>}
        : memref<64xf16> -> !ktdp.access_tile<64xindex>
    %tile = ktdp.load %at : !ktdp.access_tile<64xindex> -> tensor<64xf16>
    ktdp.store %tile, %at : tensor<64xf16>, !ktdp.access_tile<64xindex>
    return
  }
}
