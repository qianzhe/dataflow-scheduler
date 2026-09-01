// RUN: dataflow-scheduler-opt --indirect-addr-buf-legalization %s | FileCheck %s

// Iter-arg threading: verifies that the ind_addr_buf memref view is correctly
// threaded as an scf.for iter-arg after sub-step 2b, with a sentinel initial
// value outside both loops and the real view yielded from the scf.if branch.
// This uses a 2×32 IAB (same shape as basic-gather) to focus specifically on
// the iter-arg plumbing that sub-step 2b must produce.
//
// Sub-steps 2a and 2b are not yet implemented.
// TODO(sub-step-2a/2b): replace the placeholder comment below with real
// CHECK lines verifying iter_args, scf.if, and scf.yield once the loops are
// materialized.

// CHECK-LABEL: func.func @iter_arg_threading
// CHECK: ktdp_lowering.construct_indirect_access_tile

#set_iab_2x32 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_vars     = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#map_vars     = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @local_schedule_1 {
  func.func @iter_arg_threading(%arg5: index, %arg6: index, %arg7: index, %arg8: index)
      attributes {grid = [1 : index]} {
    %c0 = arith.constant 0 : index

    // IAB memory view (2×32 entries, pre-legalization)
    %iab_mv = ktdp_lowering.construct_memory_view %c0,
        sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set_iab_2x32, memory_space = "IAB"}
        : memref<2x32xindex, "IAB">

    %desc_1 = ktdp.construct_memory_view %c0,
        sizes: [64, 2, 64], strides: [64, 4096, 1]
        {coordinate_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>,
         memory_space = #ktdp.memory_space<global>}
        : memref<64x2x64xf16>

    %tile = ktdp_lowering.construct_indirect_access_tile
        intermediate_variables(%arg5, %arg6, %arg7, %arg8)
        base_ptr = %iab_mv[%arg5, %arg6]
        %desc_1[%c0, %arg7, %arg8]
        {variables_space_order = #map_vars,
         variables_space_set = #set_vars}
        : memref<64x2x64xf16>, memref<2x32xindex, "IAB">
        -> !ktdp.access_tile<2x32x2x64xindex>

    %result = ktdp.load %tile
        : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>
    return
  }
}
