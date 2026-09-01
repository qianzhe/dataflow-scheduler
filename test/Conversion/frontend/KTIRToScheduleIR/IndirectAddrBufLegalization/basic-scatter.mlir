// RUN: dataflow-scheduler-opt --indirect-addr-buf-legalization %s | FileCheck %s

// Scatter case: the source-data load and compute precede the addr_buf fill,
// so the block-move boundary for sub-step 2a starts from the source tile op
// (outermost dependent block) rather than from the IAB fill.
// After Phase 2 the same outer/inner scf.for structure must wrap all ops
// from the source tile through the indirect store.
//
// Sub-steps 2a and 2b are not yet implemented.
// TODO(sub-step-2a/2b): replace the placeholder comment below with real
// CHECK lines once the loops are materialized.

// CHECK-LABEL: func.func @basic_scatter
// CHECK: ktdp_lowering.construct_indirect_access_tile

#set_iab_2x32 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_src      = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#set_vars     = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#map_vars     = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @local_schedule_1 {
  func.func @basic_scatter(%arg5: index, %arg6: index, %arg7: index, %arg8: index)
      attributes {grid = [1 : index]} {
    %c0 = arith.constant 0 : index
    %c1000 = arith.constant 1000 : index

    // Source data load (comes before IAB fill in scatter ordering)
    %desc_src = ktdp.construct_memory_view %c1000,
        sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1]
        {coordinate_set = #set_src,
         memory_space = #ktdp.memory_space<global>}
        : memref<2x32x2x64xf16>
    %src_tile = ktdp.construct_access_tile %desc_src[%c0, %c0, %c0, %c0] {
        access_tile_set = #set_src,
        access_tile_order = #map_vars
    } : memref<2x32x2x64xf16> -> !ktdp.access_tile<2x32x2x64xindex>
    %src = ktdp.load %src_tile
        : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>

    // IAB memory view (2×32 entries, pre-legalization; fill comes after source)
    %iab_mv = ktdp_lowering.construct_memory_view %c0,
        sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set_iab_2x32, memory_space = "IAB"}
        : memref<2x32xindex, "IAB">

    // Indirect destination tile
    %desc_dst = ktdp.construct_memory_view %c0,
        sizes: [64, 2, 64], strides: [64, 4096, 1]
        {coordinate_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>,
         memory_space = #ktdp.memory_space<global>}
        : memref<64x2x64xf16>
    %dst_tile = ktdp_lowering.construct_indirect_access_tile
        intermediate_variables(%arg5, %arg6, %arg7, %arg8)
        base_ptr = %iab_mv[%arg5, %arg6]
        %desc_dst[%c0, %arg7, %arg8]
        {variables_space_order = #map_vars,
         variables_space_set = #set_vars}
        : memref<64x2x64xf16>, memref<2x32xindex, "IAB">
        -> !ktdp.access_tile<2x32x2x64xindex>
    ktdp.store %src, %dst_tile
        : tensor<2x32x2x64xf16>, !ktdp.access_tile<2x32x2x64xindex>
    return
  }
}
