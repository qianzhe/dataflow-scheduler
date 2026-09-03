// RUN: dataflow-scheduler-opt --indirect-addr-buf-legalization %s | FileCheck %s

// Scatter case: the source-data load and compute precede the addr_buf fill,
// so the block-move boundary for sub-step 2a starts from the source tile op
// (outermost dependent block) rather than from the IAB fill.
// Input is the @local_schedule_1 module emitted by IndirectComputeGroupSplit
// (Step 1 output).  At this point the full 2×32 address tensor is loaded into
// the ind_addr_buf in one shot; Steps 2+3 (IndirectAddrBufLegalization) will
// split it into per-row (window) and per-entry loops.
//
// After Step 2 the outer scf.for (window loop %i1) must wrap all ops from
// the source tile through the indirect store, narrowing the IAB fill from
// 2×32 to a single 32-entry row slice.
// After Step 3 an inner scf.for (per-entry loop %i2) further narrows to one
// entry per iteration, guarded by scf.if (%i2 == 0) for the fill.
//
// Sub-step 2a (window loop materialization) is verified below.
// Sub-step 2b (per-entry loop + scf.if guard) is not yet implemented.

// CHECK-LABEL: func.func @local_schedule_1
// CHECK:         scf.for {{.*}} to %c2 step
// CHECK:           linalg.generic
// CHECK-SAME:        iterator_types = ["parallel", "parallel", "parallel"]
// CHECK:           ktdp_lowering.construct_memory_view {{.*}} sizes: [32],
// CHECK-SAME:        memory_space = "IAB"
// CHECK:           ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME:        -> !ktdp.access_tile<32x2x64xindex>
// CHECK:           ktdp.store {{.*}} tensor<32x2x64xf16>, <32x2x64xindex>
// CHECK:         } {loop_type = #ktdf.loop_type<parallel_loop>}

#set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

module @local_schedule_1 {
  ktdf_arch.device @test_device {
    memory { kind = "IAB", ktdf_arch.features = { ktdf_arch.feature.indirect_address_buffer = { num_entries = 32 } } }
  }
  func.func @local_schedule_1() attributes {grid = [1 : index]} {
    %c0    = arith.constant 0    : index
    %c1000 = arith.constant 1000 : index
    // Compile-time constant address from the memory tracker; both
    // @local_schedule_idx_to_addr and @local_schedule_1 independently
    // reconstruct the addr_buf view from this address — no allocation or
    // memref is passed between them.
    %addr_buf_base = arith.constant 2000 : index
    %addr_buf = ktdp.construct_memory_view %addr_buf_base,
        sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>

    // Direct source load and element-wise compute come before the IAB fill.
    %desc_src = ktdp.construct_memory_view %c1000,
        sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1]
        {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>}
        : memref<2x32x2x64xf16>
    %src_tile = ktdp.construct_access_tile %desc_src[%c0, %c0, %c0, %c0]
        {access_tile_order = #map, access_tile_set = #set2}
        : memref<2x32x2x64xf16> -> !ktdp.access_tile<2x32x2x64xindex>
    %src = ktdp.load %src_tile
        : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>

    %empty = tensor.empty() : tensor<2x32x2x64xf16>
    %result = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%src : tensor<2x32x2x64xf16>)
        outs(%empty : tensor<2x32x2x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %cf10 = arith.constant 10.000000e+00 : f16
      %sum = arith.addf %in, %cf10 : f16
      linalg.yield %sum : f16
    } -> tensor<2x32x2x64xf16>

    // IAB fill: addr_buf (global) → iab_mv (IAB).
    // In scatter the fill comes after the compute and before the indirect store.
    %iab_mv = ktdp_lowering.construct_memory_view %c0,
        sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set, memory_space = "IAB"}
        : memref<2x32xindex, "IAB">
    %addr_tile = ktdp.construct_access_tile %addr_buf[%c0, %c0]
        {access_tile_order = #map1, access_tile_set = #set}
        : memref<2x32xindex, #ktdp.memory_space<global>>
        -> !ktdp.access_tile<2x32xindex>
    %addr_vals = ktdp.load %addr_tile
        : !ktdp.access_tile<2x32xindex> -> tensor<2x32xindex>
    %iab_tile = ktdp.construct_access_tile %iab_mv[%c0, %c0]
        {access_tile_order = #map1, access_tile_set = #set}
        : memref<2x32xindex, "IAB"> -> !ktdp.access_tile<2x32xindex>
    ktdp.store %addr_vals, %iab_tile
        : tensor<2x32xindex>, !ktdp.access_tile<2x32xindex>

    // Base offset is %c0: the original desc_dst base (%c10000) was folded
    // into each addr_tensor entry by @local_schedule_idx_to_addr.
    %desc_dst = ktdp.construct_memory_view %c0,
        sizes: [64, 2, 64], strides: [64, 4096, 1]
        {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>}
        : memref<64x2x64xf16>

    // All subscripts into %desc_dst are direct; indirection is resolved via
    // the ind_addr_buf.  Result shape 2×32×2×64 reflects the full IAB window
    // × the two direct dimensions.
    //   flat = 0 + ind_addr_buf[arg5,arg6] + c0*stride[0]
    //              + arg7*stride[1] + arg8*stride[2]
    //        = (c10000 + idx[arg5,arg6]*stride[0]) + arg7*stride[1]
    //              + arg8*stride[2]  ✓
    %dst_tile = ktdp_lowering.construct_indirect_access_tile
        intermediate_variables(%arg5, %arg6, %arg7, %arg8)
        base_ptr = %iab_mv[%arg5, %arg6]
        %desc_dst[(%c0), (%c0 + %arg7), (%arg8)]
        {variables_space_order = #map, variables_space_set = #set2}
        : memref<64x2x64xf16>, memref<2x32xindex, "IAB"> -> !ktdp.access_tile<2x32x2x64xindex>
    ktdp.store %result, %dst_tile
        : tensor<2x32x2x64xf16>, !ktdp.access_tile<2x32x2x64xindex>
    return
  }
}
