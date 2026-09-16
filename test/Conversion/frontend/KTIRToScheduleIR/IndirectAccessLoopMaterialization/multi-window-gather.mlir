// RUN: dataflow-scheduler-opt --mlir-print-local-scope --indirect-access-loop-materialization %s | FileCheck %s

// Input mirrors the output of IndirectAddrBufLegalization run on its own
// multi-window-gather.mlir test case: the IAB is 3-D (3x2x32), so that pass
// materializes two window loops (%i1 0..3, %i2 0..2) wrapping the entry loop
// (%i3, 0..32), whose ind_addr_buf fill is guarded by scf.if (%i3 == 0)
// against a hoisted, loop-invariant IAB memref view (no iter-arg). Two
// remaining direct-subscript intermediate variables index %desc_1 directly:
// %arg4 (dim 1, stride 4096) and %arg5 (dim 2, stride 1, full 64-element
// coverage).
//
// %arg4 fails dense-packing: size[2]*stride[2] = 64*1 = 64 != 4096, so this
// pass must materialize a loop for it (%i4, 0..2), leaving %arg5 retained
// (it passes both dense-packing, being innermost with stride 1, and full
// coverage, since its trip count equals %desc_1's dim-2 size of 64).
//
// The two outer window loops (%i1, %i2) carry no state and are untouched by
// this pass. The entry loop's relocated ind_addr_buf fill chain (guarded by
// scf.if) is independent of %i4, but this pass must still splice the entire
// entry-loop body — scf.if included — into %i4 for it to be perfectly
// nested.

// CHECK-LABEL: func.func @local_schedule_1
// CHECK:       [[IAB_VIEW:%.+]] = ktdp_lowering.construct_memory_view %c0, sizes: [32],
// CHECK-SAME:      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
// CHECK-SAME:      memory_space = "IAB"
// CHECK:       scf.for [[I1:%arg[0-9]+]] = {{.*}} to %c3 step
// CHECK:         scf.for [[I2:%arg[0-9]+]] = {{.*}} to %c2 step
// CHECK:           scf.for [[I3:%arg[0-9]+]] = %c0{{.*}} to %c32 step %c1
// CHECK:             scf.for [[I4:%arg[0-9]+]] = %c0{{.*}} to %c2{{.*}} step %c1
// CHECK:               [[EQ0:%.+]] = arith.cmpi eq, [[I3]], %c0
// CHECK:               scf.if [[EQ0]] {
// CHECK:                 ktdp.construct_access_tile {{.*}}{{\[}}[[I1]], [[I2]], {{.*}}{{\]}}
// unchanged: carried over verbatim from the relocated scf.if guard (both
// window dims pinned by IndirectAddrBufLegalization, entry dim spans the
// full row).
// CHECK-SAME:                  access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-SAME:                  access_tile_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 31 >= 0)>
// CHECK-SAME:                  -> !ktdp.access_tile<1x1x32xindex>
// CHECK:                 ktdp.load {{.*}} <1x1x32xindex> -> tensor<1x1x32xindex>
// CHECK:                 tensor.collapse_shape {{.*}} {{\[\[}}0, 1, 2{{\]\]}}
// CHECK-SAME:                tensor<1x1x32xindex> into tensor<32xindex>
// CHECK:                 ktdp.construct_access_tile [[IAB_VIEW]][%c0
// CHECK-SAME:                  access_tile_order = affine_map<(d0) -> (d0)>
// CHECK-SAME:                  access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
// CHECK:                 ktdp.store {{.*}} tensor<32xindex>, <32xindex>
// CHECK:               }
// CHECK-NOT:           else
// CHECK:               ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME:                intermediate_variables([[ARG5:%arg[0-9]+]])
// CHECK-SAME:                base_ptr = [[IAB_VIEW]][[[I3]]]
// CHECK-SAME:                {{\[}}%c0, %c0 + [[I4]], [[ARG5]]{{\]}}
// dropped: [[I4]]'s absorbed dim (dim 1, trip count 2) vanishes; only
// [[ARG5]]'s dim (dim 2, full 64-element coverage) survives.
// CHECK-SAME:                variables_space_order = affine_map<(d0) -> (d0)>
// CHECK-SAME:                variables_space_set = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
// CHECK-SAME:                -> !ktdp.access_tile<64xindex>
// CHECK:               ktdp.load {{.*}} <64xindex> -> tensor<64xf16>
// CHECK:               linalg.generic
// CHECK-SAME:              iterator_types = ["parallel"]
// CHECK:               tensor.expand_shape {{.*}} tensor<64xf16> into tensor<1x1x1x1x64xf16>
// CHECK:               ktdp.construct_access_tile {{.*}}{{\[}}[[I1]], [[I2]], [[I3]], [[I4]], %c0{{.*}}{{\]}}
// pinned: window+entry dims stay pinned as carried over; [[I4]]'s dim is
// newly pinned to a single point, and the direct dim keeps its full
// 64-element range.
// CHECK-SAME:                access_tile_order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-SAME:                access_tile_set = affine_set<(d0, d1, d2, d3, d4) : (d3 == 0, d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d4 >= 0, -d4 + 63 >= 0)>
// CHECK-SAME:                -> !ktdp.access_tile<1x1x1x1x64xindex>
// CHECK:               ktdp.store {{.*}} tensor<1x1x1x1x64xf16>, <1x1x1x1x64xindex>
// CHECK:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:         } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:       } {loop_type = #ktdf.loop_type<parallel_loop>}

#set_ab       = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 31 >= 0)>
#set_ab_pin2  = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 31 >= 0)>
#set_iab      = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#set_desc1    = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set_vars     = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set_desc2     = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 31 >= 0, d3 >= 0, -d3 + 1 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#set_desc2_pin = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 1 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

module @local_schedule_1 {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @local_schedule_1() attributes {grid = [1 : index]} {
    %c0     = arith.constant 0     : index
    %c10000 = arith.constant 10000 : index
    %addr_buf_base = arith.constant 2000 : index
    %addr_buf = ktdp.construct_memory_view %addr_buf_base,
        sizes: [3, 2, 32], strides: [64, 32, 1]
        {coordinate_set = #set_ab, memory_space = #ktdp.memory_space<global>}
        : memref<3x2x32xindex, #ktdp.memory_space<global>>

    // The ind_addr_buf view is a pure descriptor for a fixed hardware region,
    // hoisted above the entire (two window loops + one entry loop) nest
    // rather than threaded as an iter-arg.
    %iab_mv = ktdp_lowering.construct_memory_view %c0,
        sizes: [32], strides: [1]
        {coordinate_set = #set_iab, memory_space = "IAB"} : memref<32xindex, "IAB">

    %c0_i1 = arith.constant 0 : index
    %c3    = arith.constant 3 : index
    %c1_i1 = arith.constant 1 : index
    scf.for %i1 = %c0_i1 to %c3 step %c1_i1 {
      %c0_i2 = arith.constant 0 : index
      %c2    = arith.constant 2 : index
      %c1_i2 = arith.constant 1 : index
      scf.for %i2 = %c0_i2 to %c2 step %c1_i2 {
        %c0_i3 = arith.constant 0  : index
        %c32   = arith.constant 32 : index
        %c1_i3 = arith.constant 1  : index
        scf.for %i3 = %c0_i3 to %c32 step %c1_i3 {
          %eq0 = arith.cmpi eq, %i3, %c0_i3 : index
          scf.if %eq0 {
            %addr_at = ktdp.construct_access_tile %addr_buf[%i1, %i2, %c0]
                {access_tile_order = #map3, access_tile_set = #set_ab_pin2}
                : memref<3x2x32xindex, #ktdp.memory_space<global>> -> !ktdp.access_tile<1x1x32xindex>
            %addr_stick_3d = ktdp.load %addr_at : !ktdp.access_tile<1x1x32xindex> -> tensor<1x1x32xindex>
            %addr_stick = tensor.collapse_shape %addr_stick_3d [[0, 1, 2]]
                : tensor<1x1x32xindex> into tensor<32xindex>
            %iab_at = ktdp.construct_access_tile %iab_mv[%c0]
                {access_tile_order = #map1, access_tile_set = #set_iab}
                : memref<32xindex, "IAB"> -> !ktdp.access_tile<32xindex>
            ktdp.store %addr_stick, %iab_at : tensor<32xindex>, !ktdp.access_tile<32xindex>
          }

          %desc_1 = ktdp.construct_memory_view %c0,
              sizes: [64, 2, 64], strides: [64, 4096, 1]
              {coordinate_set = #set_desc1, memory_space = #ktdp.memory_space<global>}
              : memref<64x2x64xf16>

          %tmp1 = ktdp_lowering.construct_indirect_access_tile
              intermediate_variables(%arg4, %arg5)
              base_ptr = %iab_mv[%i3]
              %desc_1[%c0, (%c0 + %arg4), %arg5]
              {variables_space_order = #map2, variables_space_set = #set_vars}
              : memref<64x2x64xf16>, memref<32xindex, "IAB"> -> !ktdp.access_tile<2x64xindex>
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

          %expanded = tensor.expand_shape %computed [[0, 1, 2, 3], [4]] output_shape [1, 1, 1, 2, 64]
              : tensor<2x64xf16> into tensor<1x1x1x2x64xf16>
          %desc_2 = ktdp.construct_memory_view %c10000,
              sizes: [3, 2, 32, 2, 64], strides: [8192, 4096, 128, 64, 1]
              {coordinate_set = #set_desc2, memory_space = #ktdp.memory_space<global>}
              : memref<3x2x32x2x64xf16>
          %desc_2_at = ktdp.construct_access_tile %desc_2[%i1, %i2, %i3, %c0, %c0]
              {access_tile_order = #map5, access_tile_set = #set_desc2_pin}
              : memref<3x2x32x2x64xf16> -> !ktdp.access_tile<1x1x1x2x64xindex>
          ktdp.store %expanded, %desc_2_at : tensor<1x1x1x2x64xf16>, !ktdp.access_tile<1x1x1x2x64xindex>
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
