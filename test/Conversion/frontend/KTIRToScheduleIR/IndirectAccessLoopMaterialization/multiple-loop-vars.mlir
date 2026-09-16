// RUN: dataflow-scheduler-opt --indirect-access-loop-materialization %s | FileCheck %s

// Input is post-Phase-2 IR (1-D IAB, W = 0, single entry loop already
// materialized) with three remaining direct-subscript intermediate
// variables indexing %desc_1 (sizes [64, 2, 64], strides [64, 4096, 1]):
//   %arg7 (dim 0, stride 64):   size[1]*stride[1] = 2*4096 = 8192 != 64   -> fails packing -> loop
//   %arg8 (dim 1, stride 4096): size[2]*stride[2] = 64*1   = 64   != 4096 -> fails packing -> loop
//   %arg9 (dim 2, stride 1):    innermost, stride 1 -> packs; full 64-element coverage -> retained
//
// Both %arg7 and %arg8 need materialized loops (outermost-$base-dimension
// first); %arg9 (innermost) stays retained. The output descriptor access
// tile continues Phase 2's pin-not-drop numbering across both new loops.

// CHECK-LABEL: func.func @multi
// CHECK:       scf.for [[I2:%arg[0-9]+]] = {{.*}} to %c32 step
// CHECK:         scf.for [[I3:%arg[0-9]+]] = %c0{{.*}} to %c64 step %c1
// CHECK:           scf.for [[I4:%arg[0-9]+]] = %c0{{.*}} to %c2 step %c1
// CHECK:             [[TILE:%.+]] = ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME:            intermediate_variables([[ARG9:%arg[0-9]+]])
// CHECK-SAME:            base_ptr = {{.*}}{{\[}}[[I2]]{{\]}}
// CHECK-SAME:            {{\[}}[[I3]], [[I4]], [[ARG9]]{{\]}}
// CHECK-SAME:            -> !ktdp.access_tile<64xindex>
// CHECK:             ktdp.load [[TILE]] : <64xindex> -> tensor<64xf16>
// CHECK:             linalg.generic
// CHECK-SAME:            iterator_types = ["parallel"]
// CHECK:             tensor.expand_shape {{.*}} tensor<64xf16> into tensor<1x1x1x64xf16>
// CHECK:             ktdp.construct_access_tile {{.*}}{{\[}}[[I2]], [[I3]], [[I4]], %c0{{.*}}{{\]}}
// CHECK-SAME:            -> !ktdp.access_tile<1x1x1x64xindex>
// CHECK:             ktdp.store {{.*}} tensor<1x1x1x64xf16>, <1x1x1x64xindex>
// CHECK:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:         } {loop_type = #ktdf.loop_type<parallel_loop>}

#set_iab = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#set_base = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set_vars = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set_desc2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#map1 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @local_schedule_1 {
  ktdf_arch.device @test_device {
    memory { kind = "IAB", ktdf_arch.features = { ktdf_arch.feature.indirect_address_buffer = { num_entries = 32 } } }
  }
  func.func @multi() attributes {grid = [1 : index]} {
    %c0     = arith.constant 0     : index
    %c10000 = arith.constant 10000 : index
    %addr_buf_base = arith.constant 3000 : index

    %addr_buf = ktdp.construct_memory_view %addr_buf_base,
        sizes: [32], strides: [1]
        {coordinate_set = #set_iab, memory_space = #ktdp.memory_space<global>}
        : memref<32xindex, #ktdp.memory_space<global>>

    %iab_mv_init = ktdp_lowering.construct_memory_view %c0,
        sizes: [32], strides: [1]
        {coordinate_set = #set_iab, memory_space = "IAB"} : memref<32xindex, "IAB">

    %desc_1 = ktdp.construct_memory_view %c0,
        sizes: [64, 2, 64], strides: [64, 4096, 1]
        {coordinate_set = #set_base, memory_space = #ktdp.memory_space<global>}
        : memref<64x2x64xf16>

    %c0_i2 = arith.constant 0  : index
    %c32   = arith.constant 32 : index
    %c1_i2 = arith.constant 1  : index
    %iab_mv_final = scf.for %i2 = %c0_i2 to %c32 step %c1_i2
        iter_args(%iab_mv_carry = %iab_mv_init) -> memref<32xindex, "IAB"> {
      %eq0 = arith.cmpi eq, %i2, %c0_i2 : index
      %iab_mv = scf.if %eq0 -> memref<32xindex, "IAB"> {
        %addr_at = ktdp.construct_access_tile %addr_buf[%c0]
            {access_tile_order = #map1, access_tile_set = #set_iab}
            : memref<32xindex, #ktdp.memory_space<global>> -> !ktdp.access_tile<32xindex>
        %addr_stick = ktdp.load %addr_at : !ktdp.access_tile<32xindex> -> tensor<32xindex>
        %iab_mv_ = ktdp_lowering.construct_memory_view %c0,
            sizes: [32], strides: [1]
            {coordinate_set = #set_iab, memory_space = "IAB"} : memref<32xindex, "IAB">
        %iab_at = ktdp.construct_access_tile %iab_mv_[%c0]
            {access_tile_order = #map1, access_tile_set = #set_iab}
            : memref<32xindex, "IAB"> -> !ktdp.access_tile<32xindex>
        ktdp.store %addr_stick, %iab_at : tensor<32xindex>, !ktdp.access_tile<32xindex>
        scf.yield %iab_mv_ : memref<32xindex, "IAB">
      } else {
        scf.yield %iab_mv_carry : memref<32xindex, "IAB">
      }

      %tmp1 = ktdp_lowering.construct_indirect_access_tile
          intermediate_variables(%arg7, %arg8, %arg9)
          base_ptr = %iab_mv[%i2]
          %desc_1[%arg7, %arg8, %arg9]
          {variables_space_order = #map3, variables_space_set = #set_vars}
          : memref<64x2x64xf16>, memref<32xindex, "IAB"> -> !ktdp.access_tile<64x2x64xindex>
      %tmp1_0 = ktdp.load %tmp1 : !ktdp.access_tile<64x2x64xindex> -> tensor<64x2x64xf16>

      %e = tensor.empty() : tensor<64x2x64xf16>
      %computed = linalg.generic {
          indexing_maps = [#map3, #map3],
          iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%tmp1_0 : tensor<64x2x64xf16>)
          outs(%e : tensor<64x2x64xf16>) {
      ^bb0(%in: f16, %out: f16):
        %cf10 = arith.constant 10.000000e+00 : f16
        %added = arith.addf %in, %cf10 : f16
        linalg.yield %added : f16
      } -> tensor<64x2x64xf16>

      %expanded = tensor.expand_shape %computed [[0, 1], [2], [3]] output_shape [1, 64, 2, 64]
          : tensor<64x2x64xf16> into tensor<1x64x2x64xf16>
      %desc_2 = ktdp.construct_memory_view %c10000,
          sizes: [32, 64, 2, 64], strides: [8192, 128, 64, 1]
          {coordinate_set = #set_desc2, memory_space = #ktdp.memory_space<global>}
          : memref<32x64x2x64xf16>
      %c0_2 = arith.constant 0 : index
      %desc_2_at = ktdp.construct_access_tile %desc_2[%i2, %c0_2, %c0_2, %c0_2]
          {access_tile_order = #map4, access_tile_set = #set_desc2}
          : memref<32x64x2x64xf16> -> !ktdp.access_tile<1x64x2x64xindex>
      ktdp.store %expanded, %desc_2_at : tensor<1x64x2x64xf16>, !ktdp.access_tile<1x64x2x64xindex>

      scf.yield %iab_mv : memref<32xindex, "IAB">
    }
    return
  }
}
