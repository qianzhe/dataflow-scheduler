// RUN: dataflow-scheduler-opt --indirect-addr-buf-legalization %s | FileCheck %s

// Gather case: the addr_buf fill precedes the indirect access tile.
// Input is the @local_schedule_1 module emitted by IndirectComputeGroupSplit
// (Step 1 output).  At this point the full 2×32 address tensor is loaded into
// the ind_addr_buf in one shot; Steps 2+3 (IndirectAddrBufLegalization) will
// split it into per-row (window) and per-entry loops.
//
// After Step 2 the outer scf.for (window loop %i1) must wrap the access tile,
// narrowing the IAB fill from 2×32 to a single 32-entry row slice.
// After Step 3 an inner scf.for (per-entry loop %i2) further narrows to one
// entry per iteration, guarded by scf.if (%i2 == 0) for the fill.
//
// Sub-step 2a (window loop materialization) is verified below.
// Sub-step 2b (per-entry loop + scf.if guard) is not yet implemented.

// CHECK-LABEL: func.func @local_schedule_1
// CHECK:         scf.for [[IV:%arg[0-9]+]] = {{.*}} to %c2 step
// CHECK:           ktdp.construct_access_tile {{.*}}{{\[}}[[IV]], {{.*}}{{\]}} {{.*}} -> !ktdp.access_tile<1x32xindex>
// CHECK:           ktdp.load {{.*}} <1x32xindex> -> tensor<1x32xindex>
// CHECK:           ktdp_lowering.construct_memory_view {{.*}} sizes: [32],
// CHECK-SAME:        memory_space = "IAB"
// CHECK:           tensor.collapse_shape {{.*}} {{\[\[}}0, 1{{\]\]}}
// CHECK-SAME:        tensor<1x32xindex> into tensor<32xindex>
// CHECK:           ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME:        -> !ktdp.access_tile<32x2x64xindex>
// CHECK:           ktdp.load {{.*}} <32x2x64xindex> -> tensor<32x2x64xf16>
// CHECK:           linalg.generic
// CHECK-SAME:        iterator_types = ["parallel", "parallel", "parallel"]
// CHECK:           tensor.expand_shape {{.*}} {{\[\[}}0, 1{{.*}}tensor<32x2x64xf16> into tensor<1x32x2x64xf16>
// CHECK:           ktdp.construct_access_tile {{.*}}{{\[}}[[IV]], {{.*}}{{\]}} {{.*}} -> !ktdp.access_tile<1x32x2x64xindex>
// CHECK:           ktdp.store {{.*}} tensor<1x32x2x64xf16>, <1x32x2x64xindex>
// CHECK:         } {loop_type = #ktdf.loop_type<parallel_loop>}

#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// #set2 bounds: addr tensor rows and columns (full 2×32 window)
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
// #set3: used as the ind_addr_buf / addr-tensor access tile set (2×32 full window)
#set3 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set4 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

module @local_schedule_1 {
  ktdf_arch.device @test_device {
    memory { kind = "IAB", ktdf_arch.features = { ktdf_arch.feature.indirect_address_buffer = { num_entries = 32 } } }
  }
  func.func @local_schedule_1() attributes {grid = [1 : index]} {
    // At this stage the full 2×32 address tensor is loaded into ind_addr_buf
    // in one shot.  Steps 2+3 will split this into per-row (window) and
    // per-entry loops.
    %c0     = arith.constant 0     : index
    %c10000 = arith.constant 10000 : index
    // Compile-time constant address from the memory tracker; both
    // @local_schedule_idx_to_addr and @local_schedule_1 independently
    // reconstruct the addr_buf view from this address — no allocation or
    // memref is passed between them.
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

    %iab_mv = ktdp_lowering.construct_memory_view %c0,
        sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set3, memory_space = "IAB"}
        : memref<2x32xindex, "IAB">
    %iab_at = ktdp.construct_access_tile %iab_mv[%c0, %c0]
        {access_tile_order = #map1, access_tile_set = #set3}
        : memref<2x32xindex, "IAB"> -> !ktdp.access_tile<2x32xindex>
    ktdp.store %addr_tensor_stick, %iab_at
        : tensor<2x32xindex>, !ktdp.access_tile<2x32xindex>

    // Base offset is %c0: the original desc_1 base (%c1000) was folded into
    // each addr_tensor entry by @local_schedule_idx_to_addr.
    %desc_1 = ktdp.construct_memory_view %c0,
        sizes: [64, 2, 64], strides: [64, 4096, 1]
        {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>}
        : memref<64x2x64xf16>

    // All subscripts into %desc_1 are direct; indirection is resolved via the
    // ind_addr_buf.  Result shape 2×32×2×64 reflects the full IAB window ×
    // the two direct dimensions.
    %tmp1 = ktdp_lowering.construct_indirect_access_tile
        intermediate_variables(%arg5, %arg6, %arg7, %arg8)
        base_ptr = %iab_mv[%arg5, %arg6]
        %desc_1[(%c0), (%c0 + %arg7), (%arg8)]
        {variables_space_order = #map, variables_space_set = #set4}
        : memref<64x2x64xf16>, memref<2x32xindex, "IAB"> -> !ktdp.access_tile<2x32x2x64xindex>
    %tmp1_0 = ktdp.load %tmp1
        : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>

    // Element-wise compute on the full 2×32×2×64 tile (4 parallel dims).
    %12 = tensor.empty() : tensor<2x32x2x64xf16>
    %13 = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%tmp1_0 : tensor<2x32x2x64xf16>)
        outs(%12 : tensor<2x32x2x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %cf10 = arith.constant 10.000000e+00 : f16
      %14 = arith.addf %in, %cf10 : f16
      linalg.yield %14 : f16
    } -> tensor<2x32x2x64xf16>

    %desc_2 = ktdp.construct_memory_view %c10000,
        sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1]
        {coordinate_set = #set4, memory_space = #ktdp.memory_space<global>}
        : memref<2x32x2x64xf16>
    %0 = ktdp.construct_access_tile %desc_2[%c0, %c0, %c0, %c0]
        {access_tile_order = #map, access_tile_set = #set4}
        : memref<2x32x2x64xf16> -> !ktdp.access_tile<2x32x2x64xindex>
    ktdp.store %13, %0
        : tensor<2x32x2x64xf16>, !ktdp.access_tile<2x32x2x64xindex>
    return
  }
}
