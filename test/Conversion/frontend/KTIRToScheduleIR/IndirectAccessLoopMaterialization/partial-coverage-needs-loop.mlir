// RUN: dataflow-scheduler-opt --mlir-print-local-scope --indirect-access-loop-materialization %s | FileCheck %s

// %desc_1 is row-major (sizes [3, 64], strides [64, 1]): dim 1 (innermost)
// has stride 1, and dim 0's stride (64) equals size[1]*stride[1] (64*1) --
// dense-packing passes for BOTH dimensions. %arg7 (dim 0) nonetheless needs
// a materialized loop, because variables_space_set only gives it a trip
// count of 2 out of desc_1's full extent of 3 -- partial coverage, checked
// independently of (and despite passing) dense-packing. %arg8 (dim 1,
// innermost, full 64-element coverage) is retained.

// CHECK-LABEL: func.func @partial
// CHECK:       scf.for [[I2:%arg[0-9]+]] = {{.*}} to %c32 step
// CHECK:         scf.for [[I3:%arg[0-9]+]] = %c0{{.*}} to %c2 step %c1
// CHECK:           [[TILE:%.+]] = ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME:          intermediate_variables([[ARG8:%arg[0-9]+]])
// CHECK-SAME:          base_ptr = {{.*}}{{\[}}[[I2]]{{\]}}
// CHECK-SAME:          {{\[}}[[I3]], [[ARG8]]{{\]}}
// dropped: [[I3]]'s absorbed dim (dim 0, partial coverage: trip count 2 of
// 3) vanishes; only [[ARG8]]'s dim (dim 1, full 64-element coverage)
// survives.
// CHECK-SAME:          variables_space_order = affine_map<(d0) -> (d0)>
// CHECK-SAME:          variables_space_set = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
// CHECK-SAME:          -> !ktdp.access_tile<64xindex>
// CHECK:           ktdp.load [[TILE]] : <64xindex> -> tensor<64xf16>
// CHECK:           ktdp.construct_access_tile {{.*}}{{\[}}[[I2]], [[I3]], %c0{{.*}}{{\]}}
// pinned: continues numbering from IndirectAddrBufLegalization's own pin
// (dim 0, already pinned to [[I2]]) at the next dimension ([[I3]]), rather
// than restarting from 0; the direct dim keeps its full 64-element range.
// CHECK-SAME:          access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-SAME:          access_tile_set = affine_set<(d0, d1, d2) : (d1 == 0, d0 >= 0, -d0 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK-SAME:          -> !ktdp.access_tile<1x1x64xindex>
// CHECK:           ktdp.store {{.*}} tensor<1x1x64xf16>, <1x1x64xindex>
// CHECK:         } {loop_type = #ktdf.loop_type<parallel_loop>}

#set_iab = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#set_base = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// %arg7's trip count is 2 (partial: 2 of desc_1's 3 rows), %arg8's is the
// full 64 -- a *narrower* bound than #set_base's own d0 extent, modeling
// partial coverage independent of packing.
#set_vars = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set_desc2 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module @local_schedule_1 {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @partial() attributes {grid = [1 : index]} {
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

    // Row-major desc_1: dim1 (innermost) stride 1; dim0's stride (64)
    // equals size[1]*stride[1] (64*1) -- packing passes for BOTH dims.
    %desc_1 = ktdp.construct_memory_view %c0,
        sizes: [3, 64], strides: [64, 1]
        {coordinate_set = #set_base, memory_space = #ktdp.memory_space<global>}
        : memref<3x64xf16>

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

      // %arg7 has trip count 2 (partial coverage: desc_1 dim 0 has size 3,
      // but only 2 rows are ever visited here) despite packing being fine.
      %tmp1 = ktdp_lowering.construct_indirect_access_tile
          intermediate_variables(%arg7, %arg8)
          base_ptr = %iab_mv[%i2]
          %desc_1[%arg7, %arg8]
          {variables_space_order = #map2, variables_space_set = #set_vars}
          : memref<3x64xf16>, memref<32xindex, "IAB"> -> !ktdp.access_tile<2x64xindex>
      %tmp1_0 = ktdp.load %tmp1 : !ktdp.access_tile<2x64xindex> -> tensor<2x64xf16>

      %e = tensor.empty() : tensor<2x64xf16>
      %computed = linalg.generic {
          indexing_maps = [#map2, #map2],
          iterator_types = ["parallel", "parallel"]}
          ins(%tmp1_0 : tensor<2x64xf16>)
          outs(%e : tensor<2x64xf16>) {
      ^bb0(%in: f16, %out: f16):
        %cf10 = arith.constant 10.000000e+00 : f16
        %added = arith.addf %in, %cf10 : f16
        linalg.yield %added : f16
      } -> tensor<2x64xf16>

      %expanded = tensor.expand_shape %computed [[0, 1], [2]] output_shape [1, 2, 64]
          : tensor<2x64xf16> into tensor<1x2x64xf16>
      %desc_2 = ktdp.construct_memory_view %c10000,
          sizes: [32, 2, 64], strides: [128, 64, 1]
          {coordinate_set = #set_desc2, memory_space = #ktdp.memory_space<global>}
          : memref<32x2x64xf16>
      %c0_2 = arith.constant 0 : index
      %desc_2_at = ktdp.construct_access_tile %desc_2[%i2, %c0_2, %c0_2]
          {access_tile_order = #map3, access_tile_set = #set_desc2}
          : memref<32x2x64xf16> -> !ktdp.access_tile<1x2x64xindex>
      ktdp.store %expanded, %desc_2_at : tensor<1x2x64xf16>, !ktdp.access_tile<1x2x64xindex>
    }
    return
  }
}
