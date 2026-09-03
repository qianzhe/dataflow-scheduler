add_subdirectory(Utils)

add_dataflow_scheduler_library(DataflowSchedulerTransforms
    PathExpansion/Materializer.cpp
    PathExpansion/Planner.cpp
    PathExpansion/PlannerFactories.cpp
    AddressAssignment.cpp
    IndirectAddrBufLegalization.cpp
    AffineMinCanonicalization.cpp
    ApplyDevicePatterns.cpp
    DoubleBuffering.cpp
    EnsureDeviceDeclaration.cpp
    HoistConstantStorage.cpp
    HoistInvariants.cpp
    MaterializeRegisters.cpp
    NormalizeGridTo1D.cpp
    NormalizeSCFForLoops.cpp
    ParallelizeLoopsAcrossInstances.cpp
    PathExpansion.cpp
    ScalarBroadcastLegalization.cpp
    StripMineSCFForLoops.cpp
    TileSCFForLoops.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Transforms

  GROUP DIALECT
  EXPORT_NAME Transforms
  
  DEPENDS
    DataflowSchedulerTransformsPassesIncGen

  LINK_LIBS PUBLIC
    KTIR::KtdpDialect
    DataflowScheduler::Analysis
    DataflowScheduler::ArchViews
    DataflowScheduler::KTDFAnalysis
    DataflowSchedulerDialects::KTDFDialect
    DataflowSchedulerDialects::KTDFArchAnalysis
    DataflowSchedulerDialects::KTDFArchTransforms
    DataflowScheduler::KTDFTransforms
    DataflowScheduler::KTDFUtils
    DataflowScheduler::KTDPLoweringDialect
    DataflowScheduler::TransformUtils
    DataflowScheduler::Utils

  LINK_MLIR_LIBS
    MLIRIR
    MLIRArithDialect
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRPass
    MLIRTransformUtils
    MLIRValueBoundsOpInterface
    MLIRAlignmentAttrInterface
)
