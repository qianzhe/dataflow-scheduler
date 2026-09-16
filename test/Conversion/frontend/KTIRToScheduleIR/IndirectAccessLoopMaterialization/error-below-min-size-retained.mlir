// RUN: dataflow-scheduler-opt --indirect-access-loop-materialization -verify-diagnostics %s

// Both remaining direct-subscript intermediate variables of %desc_1 (sizes
// [2, 8], strides [8, 1]) satisfy dense-packing and full coverage, so both
// would ordinarily be retained -- but their combined element count (2*8 =
// 16 elements, 32 bytes of f16) is below the hardware minimum transfer size:
// sample_device.mlir's MNILU (co-located with the IAB) declares an
// access_granularity of 1 word for "DDR", with a 64-byte word size, i.e. a
// 64-byte minimum. The pass must emit an error rather than silently
// producing a sub-minimum transfer.

#set_iab = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#set_base = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 7 >= 0)>
#set_vars = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 7 >= 0)>
#set_desc2 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 7 >= 0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module @local_schedule_1 {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @error_sub_stick() attributes {grid = [1 : index]} {
    %c0     = arith.constant 0     : index
    %c10000 = arith.constant 10000 : index
    %addr_buf_base = arith.constant 3000 : index

    %addr_buf = ktdp.construct_memory_view %addr_buf_base,
        sizes: [32], strides: [1]
        {coordinate_set = #set_iab, memory_space = #ktdp.memory_space<global>}
        : memref<32xindex, #ktdp.memory_space<global>>

    // The ind_addr_buf view is a pure descriptor for a fixed hardware region,
    // hoisted above the entry loop entirely rather than threaded as an
    // iter-arg.
    %iab_mv = ktdp_lowering.construct_memory_view %c0,
        sizes: [32], strides: [1]
        {coordinate_set = #set_iab, memory_space = "IAB"} : memref<32xindex, "IAB">

    // Row-major, both dims retainable (packing + full coverage pass), but
    // the retained element count (2*8 = 16 elements, 32 bytes of f16) is
    // below the hardware minimum transfer size (64 bytes).
    %desc_1 = ktdp.construct_memory_view %c0,
        sizes: [2, 8], strides: [8, 1]
        {coordinate_set = #set_base, memory_space = #ktdp.memory_space<global>}
        : memref<2x8xf16>

    %c0_i2 = arith.constant 0  : index
    %c32   = arith.constant 32 : index
    %c1_i2 = arith.constant 1  : index
    scf.for %i2 = %c0_i2 to %c32 step %c1_i2 {
      %eq0 = arith.cmpi eq, %i2, %c0_i2 : index
      scf.if %eq0 {
        %addr_at = ktdp.construct_access_tile %addr_buf[%c0]
            {access_tile_order = #map1, access_tile_set = #set_iab}
            : memref<32xindex, #ktdp.memory_space<global>> -> !ktdp.access_tile<32xindex>
        %addr_stick = ktdp.load %addr_at : !ktdp.access_tile<32xindex> -> tensor<32xindex>
        %iab_at = ktdp.construct_access_tile %iab_mv[%c0]
            {access_tile_order = #map1, access_tile_set = #set_iab}
            : memref<32xindex, "IAB"> -> !ktdp.access_tile<32xindex>
        ktdp.store %addr_stick, %iab_at : tensor<32xindex>, !ktdp.access_tile<32xindex>
      }

      %tmp1 = ktdp_lowering.construct_indirect_access_tile
          intermediate_variables(%arg7, %arg8)
          base_ptr = %iab_mv[%i2]
          %desc_1[%arg7, %arg8]
          {variables_space_order = #map2, variables_space_set = #set_vars}
          : memref<2x8xf16>, memref<32xindex, "IAB"> -> !ktdp.access_tile<2x8xindex>
      // expected-error@-6 {{retained element count (16, 32 bytes) is below the minimum hardware transfer size (64 bytes)}}
      %tmp1_0 = ktdp.load %tmp1 : !ktdp.access_tile<2x8xindex> -> tensor<2x8xf16>

      %e = tensor.empty() : tensor<2x8xf16>
      %computed = linalg.generic {
          indexing_maps = [#map2, #map2],
          iterator_types = ["parallel", "parallel"]}
          ins(%tmp1_0 : tensor<2x8xf16>)
          outs(%e : tensor<2x8xf16>) {
      ^bb0(%in: f16, %out: f16):
        %cf10 = arith.constant 10.000000e+00 : f16
        %added = arith.addf %in, %cf10 : f16
        linalg.yield %added : f16
      } -> tensor<2x8xf16>

      %expanded = tensor.expand_shape %computed [[0, 1], [2]] output_shape [1, 2, 8]
          : tensor<2x8xf16> into tensor<1x2x8xf16>
      %desc_2 = ktdp.construct_memory_view %c10000,
          sizes: [32, 2, 8], strides: [16, 8, 1]
          {coordinate_set = #set_desc2, memory_space = #ktdp.memory_space<global>}
          : memref<32x2x8xf16>
      %c0_2 = arith.constant 0 : index
      %desc_2_at = ktdp.construct_access_tile %desc_2[%i2, %c0_2, %c0_2]
          {access_tile_order = #map3, access_tile_set = #set_desc2}
          : memref<32x2x8xf16> -> !ktdp.access_tile<1x2x8xindex>
      ktdp.store %expanded, %desc_2_at : tensor<1x2x8xf16>, !ktdp.access_tile<1x2x8xindex>
    }
    return
  }
}
