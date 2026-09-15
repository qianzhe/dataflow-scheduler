add_dataflow_scheduler_library(DataflowSchedulerTransformUtils
    CustomLinalgTiling.cpp
    Hoisting.cpp
    IndirectAccessTileNarrowing.cpp
    PipelineTreeLegalizer.cpp
    RegionClonePrune.cpp
    SCFTilingUtils.cpp
    Utils.cpp

  EXPORT_NAME TransformUtils

  LINK_LIBS PUBLIC
    KTIR::KtdpDialect
    DataflowScheduler::Analysis
    DataflowScheduler::KTDPLoweringDialect
    DataflowSchedulerDialects::DataflowDialect
    DataflowSchedulerDialects::KTDFDialect
    DataflowSchedulerDialects::KTDFTransforms

  LINK_MLIR_LIBS PUBLIC
    MLIRAffineDialect
    MLIRArithDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRLinalgUtils
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRSCFUtils
    MLIRTensorDialect
    MLIRLoopLikeInterface
    MLIRSideEffectInterfaces
    MLIRViewLikeInterface
)
