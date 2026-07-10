cmake_minimum_required(VERSION 3.18...3.31)
project(sgtlearn_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(SGTLEARN_BUILD_TESTS "Build C++ tests" ON)

include(FetchContent)


# region armadillo
find_package(Armadillo QUIET)

if (NOT Armadillo_FOUND)
    # Shared libarmadillo breaks auditwheel repair in manylinux wheels; static
    # wrapper links OpenBLAS at build time and drops the runtime .so dependency.
    set(STATIC_LIB ON CACHE BOOL "" FORCE)
    set(OPENBLAS_PROVIDES_LAPACK ON CACHE BOOL "" FORCE)
    set(BUILD_SMOKE_TEST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(armadillo
            GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
            GIT_TAG 14.2.x
    )
    FetchContent_MakeAvailable(armadillo)
endif ()

if (NOT TARGET armadillo::armadillo)
    if (TARGET armadillo)
        # FetchContent path: alias the real target so its include dirs
        # and link interface propagate to consumers.
        add_library(armadillo::armadillo ALIAS armadillo)
    else ()
        # find_package path: synthesise the namespaced target from legacy vars.
        add_library(armadillo::armadillo INTERFACE IMPORTED)
        target_include_directories(armadillo::armadillo INTERFACE
                ${ARMADILLO_INCLUDE_DIRS})
        target_link_libraries(armadillo::armadillo INTERFACE ${ARMADILLO_LIBRARIES})
    endif ()
endif ()

# endregion

# region --- 1. Core Library ---
add_library(sgtlearn_core STATIC
        src/algorithms/frontiers.h
        src/algorithms/missing_values.h
        src/Discretizers/univariate/UnivariateClassificationDiscretizer.h
        src/Domain/UnivariateSplitCandidate.h
        src/Domain/FeatureInfo.h
        src/Domain/LearningCriterion.h
        src/Domain/LearningFactories.h
        src/Splitters/univariate/Splitter.h
        src/Splitters/univariate/Splitter.tpp
        src/Splitters/univariate/ClassificationSplitter.h
        src/Splitters/univariate/EntropySplitter.h
        src/Splitters/univariate/GiniSplitter.h
        src/algorithms/TreeBuilder.h
        src/algorithms/TreeBuilder.tpp
        src/Discretizers/InnerDiscretizerBase.h
        src/Discretizers/DiscretizerInputKind.h
        src/Discretizers/factories/DiscretizerFactories.h
        src/Discretizers/factories/DiscretizerFactories.cpp
        src/Discretizers/univariate/UnivariateDiscretizer.h
        src/Discretizers/univariate/UnivariateDiscretizer.tpp
        src/Discretizers/ClassificationDiscretizer.h
        src/Discretizers/RegressionDiscretizer.h
        src/Domain/CategoricalSplitCandidate.h
        src/Discretizers/categorical/CategoricalDiscretizer.h
        src/Discretizers/categorical/CategoricalDiscretizer.tpp
        src/Discretizers/categorical/CategoricalClassificationDiscretizer.h
        src/Discretizers/categorical/CategoricalClassificationDiscretizer.cpp
        src/Discretizers/categorical/CategoricalRegressionDiscretizer.h
        src/Discretizers/categorical/CategoricalRegressionDiscretizer.cpp
        src/Splitters/categorical/CategoricalSplitter.h
        src/Splitters/categorical/CategoricalClassificationSplitter.h
        src/Splitters/categorical/CategoricalClassificationSplitter.cpp
        src/Splitters/categorical/CategoricalRegressionSplitter.h
        src/Splitters/categorical/CategoricalRegressionSplitter.cpp
        src/Discretizers/univariate/UnivariateClassificationDiscretizer.cpp
        src/Splitters/univariate/SquaredErrorSplitter.h
        src/Splitters/univariate/SquaredErrorSplitter.cpp
        src/Discretizers/univariate/UnivariateRegressionDiscretizer.cpp
        src/Discretizers/univariate/UnivariateRegressionDiscretizer.h
        src/Discretizers/univariate/GainHessianUnivariateDiscretizer.h
        src/algorithms/WaveletTreeMAE.h
        src/algorithms/WaveletTreeMAE.cpp
        src/Discretizers/univariate/GainHessianUnivariateDiscretizer.cpp
        src/Splitters/univariate/AbsoluteErrorSplitter.h
        src/Splitters/univariate/AbsoluteErrorSplitter.cpp
        src/Splitters/univariate/GainHessianSplitter.h
        src/Splitters/univariate/GainHessianSplitter.cpp
        src/Splitters/factories/SplitterFactory.h
        src/Splitters/factories/SplitterFactory.cpp
        src/algorithms/CoordinateDescent.h
        src/algorithms/TAO/TaoAdapter.h
        src/algorithms/TAO/TaoObjective.h
        src/algorithms/TAO/TaoObjective.cpp
        src/algorithms/TAO/ShapeGeneralizedTaoAdapter.h
        src/algorithms/TAO/ShapeGeneralizedTaoAdapter.cpp
        src/algorithms/TAO/ClassificationTaoAdapter.h
        src/algorithms/TAO/ClassificationTaoAdapter.cpp
        src/algorithms/TAO/RegressionTaoAdapter.h
        src/algorithms/TAO/RegressionTaoAdapter.cpp
        src/algorithms/TAO/TreeAlternatingOptimization.h
        src/algorithms/TAO/TreeAlternatingOptimization.cpp
        src/algorithms/FeatureBagging.h
        src/algorithms/KMeansUtils.h
        src/algorithms/ShapeGeneralizedTreeParams.h
        src/Estimators/ShapeFunctions/ShapeFunctionNode.h
        src/Estimators/ShapeFunctions/ShapeFunctionNode.cpp
        src/algorithms/ShapeBranchingTypes.h
        src/algorithms/BinPartitionAssignments.h
        src/Estimators/ShapeFunctions/ShapeFunctionSplitSearch.h
        src/Estimators/ShapeFunctions/ShapeFunctionSplitSearch.cpp
        src/Estimators/ShapeGeneralizedTree.h
        src/Estimators/ShapeGeneralizedTree.cpp
        src/Estimators/ClassificationShapeGeneralizedTree.h
        src/Estimators/ClassificationShapeGeneralizedTree.cpp
        src/Estimators/RegressionShapeGeneralizedTree.h
        src/Estimators/RegressionShapeGeneralizedTree.cpp
        src/BranchAssignmentObjectives/BranchAssignment.h
        src/BranchAssignmentObjectives/AbsoluteErrorBranchAssignment.h
        src/BranchAssignmentObjectives/AbsoluteErrorBranchAssignment.cpp
        src/BranchAssignmentObjectives/LeafAggregateProcessor.h
        src/BranchAssignmentObjectives/LeafAggregationBranchAssignment.h
        src/BranchAssignmentObjectives/LeafAggregationBranchAssignment.cpp
        src/BranchAssignmentObjectives/BranchAssignmentVariants.h
        src/BranchAssignmentObjectives/BranchAssignmentFactory.h
        src/BranchAssignmentObjectives/BranchAssignmentFactory.cpp
        src/Criterion.cpp
        src/Criterion.h
)

# PUBLIC headers: accessible to anyone who links to this library
target_include_directories(sgtlearn_core PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
)
# PUBLIC: sgtlearn_core's headers #include <armadillo>, so consumers need it too.
target_link_libraries(sgtlearn_core PUBLIC armadillo::armadillo)
set_target_properties(sgtlearn_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
# endregion

# region --- 2. Python Module (pybind11 + CARMA) ---
set(PYBIND11_FINDPYTHON ON)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# Resolve the interpreter pybind11/skbuild selected (Python_EXECUTABLE is set by
# FindPython; Python3_EXECUTABLE can be unset when hints come from the cache).
if (DEFINED Python_EXECUTABLE AND Python_EXECUTABLE)
    set(SGTLEARN_PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
elseif (DEFINED Python3_EXECUTABLE AND Python3_EXECUTABLE)
    set(SGTLEARN_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
else ()
    set(SGTLEARN_PYTHON_EXECUTABLE "${_Python3_EXECUTABLE}")
endif ()

execute_process(
        COMMAND "${SGTLEARN_PYTHON_EXECUTABLE}" -m pybind11_stubgen --help
        RESULT_VARIABLE SGTLEARN_PYBIND11_STUBGEN_RC
        OUTPUT_QUIET
        ERROR_QUIET
)
if (SGTLEARN_PYBIND11_STUBGEN_RC EQUAL 0)
    set(SGTLEARN_HAVE_PYBIND11_STUBGEN TRUE)
else ()
    set(SGTLEARN_HAVE_PYBIND11_STUBGEN FALSE)
    message(STATUS "pybind11_stubgen not found for ${SGTLEARN_PYTHON_EXECUTABLE}; skipping .pyi generation (listed in [build-system] requires)")
endif ()

set(CARMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CARMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(carma
        GIT_REPOSITORY https://github.com/RUrlus/carma.git
        GIT_TAG v0.8.0
)
FetchContent_MakeAvailable(carma)

# Auto-discover all binding modules from bindings/*.cpp
# Each file becomes its own module: _native.cpp -> import _native
file(GLOB BINDING_SOURCES CONFIGURE_DEPENDS "bindings/*.cpp")

foreach (BINDING_SRC ${BINDING_SOURCES})
    get_filename_component(MODULE_NAME ${BINDING_SRC} NAME_WE)

    pybind11_add_module(${MODULE_NAME} MODULE)
    target_sources(${MODULE_NAME} PRIVATE ${BINDING_SRC})
    target_link_libraries(${MODULE_NAME} PRIVATE sgtlearn_core carma::carma)
    install(TARGETS ${MODULE_NAME} DESTINATION .)

    if (SGTLEARN_HAVE_PYBIND11_STUBGEN)
        # Run stubgen via a Python wrapper so any failure (e.g. dynamic-linker
        # mismatches when importing the freshly-built .so) doesn't abort the
        # install. Subprocess return code is swallowed; missing .pyi is then
        # tolerated by the OPTIONAL install below.
        add_custom_command(
                TARGET ${MODULE_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=$<TARGET_FILE_DIR:${MODULE_NAME}>:$ENV{PYTHONPATH}"
                "${SGTLEARN_PYTHON_EXECUTABLE}" -c
                "import subprocess, sys; subprocess.call([sys.executable, '-m', 'pybind11_stubgen', '${MODULE_NAME}', '-o', r'$<TARGET_FILE_DIR:${MODULE_NAME}>'])"
                COMMENT "Generating python stubs for ${MODULE_NAME} (non-fatal)..."
                VERBATIM
        )
        install(FILES "$<TARGET_FILE_DIR:${MODULE_NAME}>/${MODULE_NAME}.pyi" DESTINATION . OPTIONAL)
    endif ()
endforeach ()

# endregion

# region --- 3. Unit Tests (Catch2) ---
if (SGTLEARN_BUILD_TESTS)
    FetchContent_Declare(Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.5.2
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(cpp_tests
            tests/test_wavelet_tree_mae.cpp
            tests/test_splitters.cpp
            tests/test_branch_assignment.cpp
    )

    target_link_libraries(cpp_tests PRIVATE
            sgtlearn_core
            Catch2::Catch2WithMain
    )

    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
    include(Catch)
    include(CTest)
    enable_testing()
    catch_discover_tests(cpp_tests)

endif ()

# endregion