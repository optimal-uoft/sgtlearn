cmake_minimum_required(VERSION 3.18...3.31)
project(sgtlearn_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(SGTLEARN_BUILD_TESTS "Build C++ tests" ON)

include(FetchContent)


# region armadillo
find_package(Armadillo QUIET)

if (NOT Armadillo_FOUND)
    FetchContent_Declare(armadillo
            GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
            GIT_TAG 14.2.x
    )
    FetchContent_MakeAvailable(armadillo)
endif ()

if (NOT TARGET armadillo::armadillo)
    add_library(armadillo::armadillo INTERFACE IMPORTED)
    target_include_directories(armadillo::armadillo INTERFACE
            ${ARMADILLO_INCLUDE_DIRS})
    target_link_libraries(armadillo::armadillo INTERFACE ${ARMADILLO_LIBRARIES})
endif ()

# endregion

# region --- 1. Core Library ---
add_library(sgtlearn_core STATIC
        src/algorithms/frontiers.h
        include/UnivariateClassificationDiscretizer.h
        src/Domain/SplitCandidate.h
        src/Splitters/Splitter.h
        src/Splitters/Splitter.tpp
        src/Splitters/ClassificationSplitter.h
        src/Splitters/EntropySplitter.h
        src/Splitters/GiniSplitter.h
        src/algorithms/TreeBuilder.h
        src/algorithms/TreeBuilder.tpp
        src/Discretizers/UnivariateDiscretizer.h
        src/Discretizers/UnivariateDiscretizer.tpp
        src/Discretizers/UnivariateClassificationDiscretizer.cpp
        src/Splitters/SquaredErrorSplitter.h
        src/Splitters/SquaredErrorSplitter.cpp
        src/Discretizers/UnivariateRegressionDiscretizer.cpp
        include/UnivariateRegressionDiscretizer.h
        include/GainHessianUnivariateDiscretizer.h
        src/algorithms/WaveletTreeMAE.h
        src/algorithms/WaveletTreeMAE.cpp
        src/Discretizers/GainHessianUnivariateDiscretizer.cpp
        src/Splitters/AbsoluteErrorSplitter.h
        src/Splitters/AbsoluteErrorSplitter.cpp
        src/Splitters/GainHessianSplitter.h
        src/Splitters/GainHessianSplitter.cpp
)

# PUBLIC headers: accessible to anyone who links to this library
target_include_directories(sgtlearn_core PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
        PRIVATE
        ${ARMADILLO_INCLUDE_DIRS}
)
target_link_libraries(sgtlearn_core ${ARMADILLO_LIBRARIES})
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

    add_custom_command(
            TARGET ${MODULE_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=$<TARGET_FILE_DIR:${MODULE_NAME}>:$ENV{PYTHONPATH}"
            ${Python3_EXECUTABLE} -m pybind11_stubgen ${MODULE_NAME}
            -o "$<TARGET_FILE_DIR:${MODULE_NAME}>"
            COMMENT "Generating python stubs for ${MODULE_NAME}..."
            VERBATIM
    )
    install(FILES "$<TARGET_FILE_DIR:${MODULE_NAME}>/${MODULE_NAME}.pyi" DESTINATION .)
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
    )

    target_include_directories(cpp_tests PRIVATE
            ${ARMADILLO_INCLUDE_DIRS}
    )

    target_link_libraries(cpp_tests PRIVATE
            sgtlearn_core
            Catch2::Catch2WithMain
            ${ARMADILLO_LIBRARIES}
    )

    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
    include(Catch)
    include(CTest)
    enable_testing()
    catch_discover_tests(cpp_tests)

endif ()

# endregion