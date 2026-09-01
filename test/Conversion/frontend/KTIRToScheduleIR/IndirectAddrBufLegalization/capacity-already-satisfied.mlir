// RUN: dataflow-scheduler-opt --indirect-addr-buf-legalization %s | FileCheck %s

// IAB capacity already satisfied: the ind_addr_buf view already has exactly
// one row (1×32 = 32 entries ≤ 32 capacity limit). Sub-step 2a should
// recognise that no window loop is needed and emit only the inner per-entry
// loop for sub-step 2b.
//
// Sub-steps 2a and 2b are not yet implemented.
// TODO(sub-step-2a/2b): replace the placeholder comment below with real
// CHECK lines once the loops are materialized.

// CHECK-LABEL: func.func @capacity_satisfied
// CHECK: ktdp_lowering.construct_indirect_access_tile

#set_iab_1x32 = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#set_vars     = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#map_vars     = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module @local_schedule_1 {
  func.func @capacity_satisfied(%arg6: index, %arg7: index, %arg8: index)
      attributes {grid = [1 : index]} {
    %c0 = arith.constant 0 : index

    // 1-D IAB: already within the 32-entry hardware limit
    %iab_mv = ktdp_lowering.construct_memory_view %c0,
        sizes: [32], strides: [1]
        {coordinate_set = #set_iab_1x32, memory_space = "IAB"}
        : memref<32xindex, "IAB">

    %desc_1 = ktdp.construct_memory_view %c0,
        sizes: [64, 64], strides: [64, 1]
        {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
         memory_space = #ktdp.memory_space<global>}
        : memref<64x64xf16>

    %tile = ktdp_lowering.construct_indirect_access_tile
        intermediate_variables(%arg6, %arg7, %arg8)
        base_ptr = %iab_mv[%arg6]
        %desc_1[%arg7, %arg8]
        {variables_space_order = #map_vars,
         variables_space_set = #set_vars}
        : memref<64x64xf16>, memref<32xindex, "IAB">
        -> !ktdp.access_tile<32x2x64xindex>

    %result = ktdp.load %tile
        : !ktdp.access_tile<32x2x64xindex> -> tensor<32x2x64xf16>
    return
  }
}
