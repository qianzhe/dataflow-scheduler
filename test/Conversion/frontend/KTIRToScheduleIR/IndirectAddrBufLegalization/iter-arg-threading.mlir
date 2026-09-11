// RUN: dataflow-scheduler-opt --indirect-addr-buf-legalization %s | FileCheck %s

// Iter-arg threading: verifies that the ind_addr_buf memref view is correctly
// threaded as an scf.for iter-arg after sub-step 2b, with a sentinel initial
// value outside both loops and the real view yielded from the scf.if branch.
// This uses a 2×32 IAB (same shape as basic-gather), so both sub-step 2a
// (window loop) and sub-step 2b (entry loop) fire.

// CHECK-LABEL: func.func @iter_arg_threading
// Sub-step 2a: outer window loop, annotated parallel.
// CHECK:      scf.for %[[I1:.+]] = %c0{{.*}} to %c2 step %c1{{.*}}
// Sub-step 2b: sentinel constructed before the inner loop, iter-arg threading.
// CHECK:        %[[SENTINEL:.+]] = ktdp_lowering.construct_memory_view %c0, sizes: [32],
// CHECK-SAME:     memory_space = "IAB"
// CHECK:        scf.for %[[I2:.+]] = %c0{{.*}} to %c32 step %c1
// CHECK-SAME:       iter_args(%{{.*}} = %[[SENTINEL]]) -> (memref<32xindex, "IAB">)
// CHECK:          %[[EQ0:.+]] = arith.cmpi eq, %[[I2]], %c0
// CHECK:          %[[IABMV:.+]] = scf.if %[[EQ0]] -> (memref<32xindex, "IAB">) {
// CHECK:            ktdp.construct_access_tile {{.*}}{{\[}}%[[I1]], {{.*}}{{\]}}
// CHECK-SAME:         -> !ktdp.access_tile<1x32xindex>
// CHECK:            ktdp.load {{.*}} <1x32xindex> -> tensor<1x32xindex>
// CHECK:            tensor.collapse_shape {{.*}} tensor<1x32xindex> into tensor<32xindex>
// CHECK:            ktdp_lowering.construct_memory_view %c0, sizes: [32],
// CHECK-SAME:         memory_space = "IAB"
// CHECK:            ktdp.store {{.*}} tensor<32xindex>, <32xindex>
// CHECK:            scf.yield
// CHECK:          } else {
// CHECK:            scf.yield
// CHECK:          }
// CHECK:          ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME:       base_ptr = %[[IABMV]][%[[I2]]]
// CHECK-SAME:       -> !ktdp.access_tile<2x64xindex>
// CHECK:          ktdp.load {{.*}} <2x64xindex> -> tensor<2x64xf16>
// CHECK:          linalg.generic
// CHECK:          ktdp.construct_access_tile {{.*}}{{\[}}%[[I1]], %[[I2]], {{.*}}{{\]}}
// CHECK-SAME:       -> !ktdp.access_tile<1x1x2x64xindex>
// CHECK:          ktdp.store {{.*}} tensor<1x1x2x64xf16>, <1x1x2x64xindex>
// CHECK:          scf.yield %[[IABMV]]
// CHECK:      } {loop_type = #ktdf.loop_type<parallel_loop>}

#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set3 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set4 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

module @local_schedule_1 {
  ktdf_arch.device @test_device {
    memory { kind = "IAB", ktdf_arch.features = { ktdf_arch.feature.indirect_address_buffer = { num_entries = 32 } } }
  }
  func.func @iter_arg_threading(%arg5: index, %arg6: index, %arg7: index, %arg8: index)
      attributes {grid = [1 : index]} {
    %c0     = arith.constant 0     : index
    %c10000 = arith.constant 10000 : index
    %addr_buf_base = arith.constant 2000 : index

    %addr_buf = ktdp.construct_memory_view %addr_buf_base,
        sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>
    %addr_buf_at = ktdp.construct_access_tile %addr_buf[%c0, %c0]
        {access_tile_order = #map1, access_tile_set = #set3}
        : memref<2x32xindex, #ktdp.memory_space<global>>
        -> !ktdp.access_tile<2x32xindex>
    %addr_tensor_stick = ktdp.load %addr_buf_at
        : !ktdp.access_tile<2x32xindex> -> tensor<2x32xindex>

    // IAB memory view (2×32 entries, pre-legalization)
    %iab_mv = ktdp_lowering.construct_memory_view %c0,
        sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set3, memory_space = "IAB"}
        : memref<2x32xindex, "IAB">
    %iab_at = ktdp.construct_access_tile %iab_mv[%c0, %c0]
        {access_tile_order = #map1, access_tile_set = #set3}
        : memref<2x32xindex, "IAB"> -> !ktdp.access_tile<2x32xindex>
    ktdp.store %addr_tensor_stick, %iab_at
        : tensor<2x32xindex>, !ktdp.access_tile<2x32xindex>

    %desc_1 = ktdp.construct_memory_view %c0,
        sizes: [64, 2, 64], strides: [64, 4096, 1]
        {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>}
        : memref<64x2x64xf16>

    %tile = ktdp_lowering.construct_indirect_access_tile
        intermediate_variables(%arg5, %arg6, %arg7, %arg8)
        base_ptr = %iab_mv[%arg5, %arg6]
        %desc_1[(%c0), (%arg7), (%arg8)]
        {variables_space_order = #map,
         variables_space_set = #set4}
        : memref<64x2x64xf16>, memref<2x32xindex, "IAB">
        -> !ktdp.access_tile<2x32x2x64xindex>

    %result = ktdp.load %tile
        : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>

    %empty = tensor.empty() : tensor<2x32x2x64xf16>
    %computed = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%result : tensor<2x32x2x64xf16>)
        outs(%empty : tensor<2x32x2x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %cf10 = arith.constant 10.000000e+00 : f16
      %added = arith.addf %in, %cf10 : f16
      linalg.yield %added : f16
    } -> tensor<2x32x2x64xf16>

    %desc_2 = ktdp.construct_memory_view %c10000,
        sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1]
        {coordinate_set = #set4, memory_space = #ktdp.memory_space<global>}
        : memref<2x32x2x64xf16>
    %desc_2_at = ktdp.construct_access_tile %desc_2[%c0, %c0, %c0, %c0]
        {access_tile_order = #map, access_tile_set = #set4}
        : memref<2x32x2x64xf16> -> !ktdp.access_tile<2x32x2x64xindex>
    ktdp.store %computed, %desc_2_at
        : tensor<2x32x2x64xf16>, !ktdp.access_tile<2x32x2x64xindex>
    return
  }
}
