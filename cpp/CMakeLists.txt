cmake_minimum_required(VERSION 3.18...3.31)
project(sgtlearn_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(SGTLEARN_BUILD_TESTS "Build C++ tests" ON)

include(FetchContent)

# region mlpack

find_package(mlpack QUIET)

if (NOT mlpack_FOUND)
    FetchContent_Declare(mlpack
            GIT_REPOSITORY https://github.com/mlpack/mlpack.git
            GIT_TAG 4.7.0
    )
    FetchContent_MakeAvailable(mlpack)
    add_library(mlpack::mlpack INTERFACE IMPORTED)
    target_include_directories(mlpack::mlpack INTERFACE
            "${mlpack_SOURCE_DIR}/src"
    )
endif ()

# endregion

# region armadillo
find_package(Armadillo QUIET)

if (NOT Armadillo_FOUND)
    FetchContent_Declare(armadillo
            GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
            GIT_TAG 14.2.x
    )
    FetchContent_MakeAvailable(armadillo)
endif ()

# endregion

# region --- 1. Core Library ---
add_library(sgtlearn_core STATIC
        include/sgtlearn/adder.hpp
        src/adder.cpp
        include/Discretizers.h
        include/frontiers.h
        src/UnivariateDiscretizer.cpp
)

# PUBLIC headers: accessible to anyone who links to this library
target_include_directories(sgtlearn_core PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        PRIVATE
        ${MLPACK_INCLUDE_DIRS}
        ${ARMADILLO_INCLUDE_DIRS}
        src
)
target_link_libraries(sgtlearn_core mlpack::mlpack  ${ARMADILLO_LIBRARIES})
set_target_properties(sgtlearn_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# endregion

# region --- 2. Python Module (pybind11) ---
set(PYBIND11_FINDPYTHON ON)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# Auto-discover all binding modules from bindings/*.cpp
# Each file becomes its own module: _native.cpp -> import _native
file(GLOB BINDING_SOURCES CONFIGURE_DEPENDS "bindings/*.cpp")

foreach (BINDING_SRC ${BINDING_SOURCES})
    get_filename_component(MODULE_NAME ${BINDING_SRC} NAME_WE)

    pybind11_add_module(${MODULE_NAME} MODULE)
    target_sources(${MODULE_NAME} PRIVATE ${BINDING_SRC})
    target_link_libraries(${MODULE_NAME} PRIVATE sgtlearn_core)
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
            ../tests/cpp/test_adder.cpp
            ../tests/cpp/test_univariateDiscretizer.cpp
    )

    target_include_directories(cpp_tests PRIVATE
            ${MLPACK_INCLUDE_DIRS}
            ${ARMADILLO_INCLUDE_DIRS}
    )

    target_link_libraries(cpp_tests PRIVATE
            sgtlearn_core
            Catch2::Catch2WithMain
            mlpack::mlpack
            ${ARMADILLO_LIBRARIES}
    )

    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
    include(Catch)
    include(CTest)
    enable_testing()
    catch_discover_tests(cpp_tests)

endif ()

# endregion