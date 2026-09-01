// RUN: dataflow-scheduler-opt --indirect-addr-buf-legalization %s | FileCheck %s

// When no ktdp_lowering.construct_indirect_access_tile is present the pass
// must be a no-op: the IR must pass through completely unchanged.

// CHECK-LABEL: func.func @no_indirect
// CHECK-NOT:   scf.for
func.func @no_indirect() {
  %c0 = arith.constant 0 : index
  %c10000 = arith.constant 10000 : index
  return
}
