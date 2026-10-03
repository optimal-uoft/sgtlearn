# C++ Build Guide

This directory contains the C++ core library and pybind11 module definitions used by the Python package build.

## Layout Conventions

- `src/`: headers (`.h`, `.tpp`) and sources (`.cpp`) for `sgtlearn_core`; `src/` is its public include directory.
- `bindings/`: pybind11 module sources; each `*.cpp` file becomes one extension module target.
- `tests/`: C++ test files used by `cpp_tests`.

| Directory | Content Type | Visibility | Target |
|:--|:--|:--|:--|
| `src/` | Headers (`.h`, `.tpp`) and logic (`.cpp`) | Public | `sgtlearn_core` |
| `bindings/` | Python wrappers (`.cpp`) | Private | `<module_name>` |
| `tests/` | Unit tests (`.cpp`) | Private | `cpp_tests` |

## Targets Defined in `CMakeLists.txt`

- `sgtlearn_core` (static library): base C++ logic.
- `<module_name>` (pybind11 module): one module per file in `bindings/`.
- `cpp_tests` (Catch2 executable): optional C++ tests.

### File Addition Conventions

- `sgtlearn_core`: add new source/header files explicitly in `add_library(...)`.
- `bindings/`: module targets are auto-discovered with `file(GLOB ... "bindings/*.cpp")`.
- `cpp_tests`: add test files explicitly in `add_executable(cpp_tests ...)`.

## Stub Generation Behavior

For each pybind11 module target:

1. the extension module (`.so`) is built;
2. if `pybind11_stubgen` was importable at configure time, it runs as a non-fatal post-build step;
3. the generated `.pyi` is written to the same output directory as that module.

The install rules place the `.so` and, when it was generated, the matching `.pyi` together in the same install destination.

## Common Commands

Configure:

```bash
cmake -S cpp -B build -DSGTLEARN_BUILD_TESTS=ON
```

Build:

```bash
cmake --build build
```

Run C++ tests:

```bash
ctest --test-dir build --output-on-failure
```

Skip C++ test target:

```bash
cmake -S cpp -B build -DSGTLEARN_BUILD_TESTS=OFF
```

Most developers should use `pip install .` from the repository root, which drives this CMake configuration via scikit-build-core (with `-DSGTLEARN_BUILD_TESTS=OFF`).

## CLion Notes

1. For core/test files, use "New > C++ Source File" and add to the intended target.
2. For binding files, create sources under `bindings/` and do not manually attach them to a target.
3. If a new binding file is not picked up immediately, trigger a CMake reload.



