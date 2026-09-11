// RUN: dataflow-scheduler-opt --indirect-addr-buf-legalization %s | FileCheck %s

// W=2 scatter case: the IAB is 3-D (3×2×32), so two window loops are needed.
// The source-data load and compute precede the addr_buf fill (scatter order).
// After sub-step 2a:
//   - Outer scf.for %i1 (0..2) absorbs dim 0 (size 3).
//   - Inner scf.for %i2 (0..1) absorbs dim 1 (size 2).
//   - addr_buf AT on k=0: [%i1, %c0, %c0], access_tile_set pins d0→1,
//     result !ktdp.access_tile<1x2x32xindex>, collapse_shape [[0,1,2]] →
//     tensor<32xindex>.
//   - addr_buf AT on k=1: [%i1, %i2, %c0], access_tile_set pins d0 and d1→1,
//     result !ktdp.access_tile<1x1x32xindex>, collapse_shape [[0,1,2]] →
//     tensor<32xindex> (replaces the k=0 collapse).
//   - IAB memref finally rank-1: memref<32xindex, "IAB">.
// After sub-step 2b, a third loop (%i3) absorbs the remaining per-entry
// dimension (0..31), guarded by scf.if (%i3 == 0), with the IAB memref
// threaded as an iter-arg. The source-tile load and compute (independent of
// the IAB) end up after the scf.if guard, still inside the entry loop —
// indirect access tile result finally 2×64.

// CHECK-LABEL: func.func @local_schedule_1
// CHECK:         scf.for [[IV0:%arg[0-9]+]] = {{.*}} to %c3 step
// CHECK:           scf.for [[IV1:%arg[0-9]+]] = {{.*}} to %c2 step
// Sub-step 2b: sentinel outside the inner (entry) loop, iter-arg threading.
// CHECK:             [[SENTINEL:%.+]] = ktdp_lowering.construct_memory_view %c0, sizes: [32],
// CHECK-SAME:          memory_space = "IAB"
// CHECK:             scf.for [[IV2:%arg[0-9]+]] = %c0{{.*}} to %c32 step %c1
// CHECK-SAME:            iter_args(%{{.*}} = [[SENTINEL]]) -> (memref<32xindex, "IAB">)
// CHECK:               [[EQ0:%.+]] = arith.cmpi eq, [[IV2]], %c0
// CHECK:               [[IABMV:%.+]] = scf.if [[EQ0]] -> (memref<32xindex, "IAB">) {
// CHECK:                 ktdp.construct_access_tile {{.*}}{{\[}}[[IV0]], [[IV1]], {{.*}}{{\]}} {{.*}} -> !ktdp.access_tile<1x1x32xindex>
// CHECK:                 ktdp.load {{.*}} <1x1x32xindex> -> tensor<1x1x32xindex>
// CHECK:                 tensor.collapse_shape {{.*}} {{\[\[}}0, 1, 2{{\]\]}}
// CHECK-SAME:              tensor<1x1x32xindex> into tensor<32xindex>
// CHECK:                 ktdp_lowering.construct_memory_view %c0, sizes: [32],
// CHECK-SAME:              memory_space = "IAB"
// CHECK:                 ktdp.store {{.*}} tensor<32xindex>, <32xindex>
// CHECK:                 scf.yield
// CHECK:               } else {
// CHECK:                 scf.yield
// CHECK:               }
// Source-tile load + compute: independent of the IAB, so it ends up after
// the scf.if guard rather than before it.
// CHECK:               ktdp.construct_access_tile {{.*}}{{\[}}[[IV0]], [[IV1]], [[IV2]], {{.*}}{{\]}} {{.*}} -> !ktdp.access_tile<1x1x1x2x64xindex>
// CHECK:               ktdp.load {{.*}} <1x1x1x2x64xindex> -> tensor<1x1x1x2x64xf16>
// CHECK:               tensor.collapse_shape {{.*}} {{\[\[}}0, 1, 2, 3{{.*}}tensor<1x1x1x2x64xf16> into tensor<2x64xf16>
// CHECK:               linalg.generic
// CHECK-SAME:            iterator_types = ["parallel", "parallel"]
// CHECK:               ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME:              base_ptr = [[IABMV]][[[IV2]]]
// CHECK-SAME:              -> !ktdp.access_tile<2x64xindex>
// CHECK:               ktdp.store {{.*}} tensor<2x64xf16>, <2x64xindex>
// CHECK:               scf.yield [[IABMV]]
// CHECK:             }
// CHECK:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:         } {loop_type = #ktdf.loop_type<parallel_loop>}

// Affine sets for a 3-D IAB (3×2×32).
// #set_ab: coordinate set for the 3×2×32 global addr_buf.
#set_ab = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 31 >= 0)>
// #set_iab: same bounds, used as IAB coord set and access_tile_set.
#set_iab = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 31 >= 0)>
// #set_data: coordinate set for the indirect data descriptor (64×2×64).
#set_data = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// #set_src: source tile shape (3×2×32×2×64).
#set_src = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 31 >= 0, d3 >= 0, -d3 + 1 >= 0, d4 >= 0, -d4 + 63 >= 0)>
// #set_vars: variables_space_set for 5 intermediate vars (%a5..%a9):
//   d0:0..2 (IAB dim0), d1:0..1 (IAB dim1), d2:0..31 (IAB dim2=per-entry),
//   d3:0..1 (direct dim1 of data), d4:0..63 (direct dim2 of data).
#set_vars = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 31 >= 0, d3 >= 0, -d3 + 1 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#map3  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5  = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

module @local_schedule_1 {
  ktdf_arch.device @test_device {
    memory { kind = "IAB", ktdf_arch.features = { ktdf_arch.feature.indirect_address_buffer = { num_entries = 32 } } }
  }
  func.func @local_schedule_1() attributes {grid = [1 : index]} {
    %c0     = arith.constant 0     : index
    %c1000  = arith.constant 1000  : index
    %addr_buf_base = arith.constant 2000 : index

    // Source data: 3×2×32×2×64 (full window × two direct dims).
    %desc_src = ktdp.construct_memory_view %c1000,
        sizes: [3, 2, 32, 2, 64], strides: [8192, 4096, 128, 64, 1]
        {coordinate_set = #set_src, memory_space = #ktdp.memory_space<global>}
        : memref<3x2x32x2x64xf16>
    %src_tile = ktdp.construct_access_tile %desc_src[%c0, %c0, %c0, %c0, %c0]
        {access_tile_order = #map5, access_tile_set = #set_src}
        : memref<3x2x32x2x64xf16> -> !ktdp.access_tile<3x2x32x2x64xindex>
    %src = ktdp.load %src_tile
        : !ktdp.access_tile<3x2x32x2x64xindex> -> tensor<3x2x32x2x64xf16>

    // Element-wise compute (scatter: runs before addr_buf fill).
    %empty = tensor.empty() : tensor<3x2x32x2x64xf16>
    %result = linalg.generic {
        indexing_maps = [#map5, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
        ins(%src : tensor<3x2x32x2x64xf16>)
        outs(%empty : tensor<3x2x32x2x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %cf10 = arith.constant 10.000000e+00 : f16
      %sum = arith.addf %in, %cf10 : f16
      linalg.yield %sum : f16
    } -> tensor<3x2x32x2x64xf16>

    // Global addr_buf: 3×2×32 address table.
    %addr_buf = ktdp.construct_memory_view %addr_buf_base,
        sizes: [3, 2, 32], strides: [64, 32, 1]
        {coordinate_set = #set_ab, memory_space = #ktdp.memory_space<global>}
        : memref<3x2x32xindex, #ktdp.memory_space<global>>
    %addr_buf_at = ktdp.construct_access_tile %addr_buf[%c0, %c0, %c0]
        {access_tile_order = #map3, access_tile_set = #set_iab}
        : memref<3x2x32xindex, #ktdp.memory_space<global>>
        -> !ktdp.access_tile<3x2x32xindex>
    %addr_tensor = ktdp.load %addr_buf_at
        : !ktdp.access_tile<3x2x32xindex> -> tensor<3x2x32xindex>

    // IAB fill: addr_buf (global) → iab_mv (IAB).
    %iab_mv = ktdp_lowering.construct_memory_view %c0,
        sizes: [3, 2, 32], strides: [64, 32, 1]
        {coordinate_set = #set_iab, memory_space = "IAB"}
        : memref<3x2x32xindex, "IAB">
    %iab_at = ktdp.construct_access_tile %iab_mv[%c0, %c0, %c0]
        {access_tile_order = #map3, access_tile_set = #set_iab}
        : memref<3x2x32xindex, "IAB"> -> !ktdp.access_tile<3x2x32xindex>
    ktdp.store %addr_tensor, %iab_at
        : tensor<3x2x32xindex>, !ktdp.access_tile<3x2x32xindex>

    // Destination descriptor (indirect).
    %desc_dst = ktdp.construct_memory_view %c0,
        sizes: [64, 2, 64], strides: [64, 4096, 1]
        {coordinate_set = #set_data, memory_space = #ktdp.memory_space<global>}
        : memref<64x2x64xf16>

    // Indirect access tile: 5 intermediate vars (3 IAB + 2 direct).
    %dst_tile = ktdp_lowering.construct_indirect_access_tile
        intermediate_variables(%arg5, %arg6, %arg7, %arg8, %arg9)
        base_ptr = %iab_mv[%arg5, %arg6, %arg7]
        %desc_dst[(%c0), (%c0 + %arg8), (%arg9)]
        {variables_space_order = #map5, variables_space_set = #set_vars}
        : memref<64x2x64xf16>, memref<3x2x32xindex, "IAB">
        -> !ktdp.access_tile<3x2x32x2x64xindex>
    ktdp.store %result, %dst_tile
        : tensor<3x2x32x2x64xf16>, !ktdp.access_tile<3x2x32x2x64xindex>
    return
  }
}
