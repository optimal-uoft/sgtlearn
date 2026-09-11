# Candidate build used for correctness and performance

Source revision: `0d7d8f6d35859e4958564e65a3ac3592592e047f`, detached checkout `/tmp/sgtlearn-candidate`. The noneditable wheel is installed into `/tmp/sgtlearn-candidate-env`, separately from the immutable baseline. `imports.json` records actual package/native paths and hashes; `build-provenance.txt` records the clean source revision, compiler/core flags, hardware, library linkage and dependency source revisions. `dependencies.txt` contains the same 28 exact package versions as the baseline.

Reconstruction commands below assume fresh checkout/environment/build paths. Do not execute them against the captured environments while measurements are running. Run from the repository root:

```sh
git worktree add --detach /tmp/sgtlearn-candidate 0d7d8f6d35859e4958564e65a3ac3592592e047f
uv venv /tmp/sgtlearn-candidate-env --python 3.14.5
uv pip install --python /tmp/sgtlearn-candidate-env/bin/python -r benchmarks/results/outer-growth-candidate-build/dependencies.txt
CMAKE_BUILD_PARALLEL_LEVEL=4 uv pip install \
  --python /tmp/sgtlearn-candidate-env/bin/python \
  --no-build-isolation --no-deps /tmp/sgtlearn-candidate \
  --config-settings build-dir=/tmp/sgtlearn-candidate/build \
  --config-settings cmake.args=-DSGTLEARN_BUILD_TESTS=ON \
  --config-settings cmake.define.FETCHCONTENT_SOURCE_DIR_ARMADILLO=/tmp/sgt52-build/_deps/armadillo-src \
  --config-settings cmake.define.FETCHCONTENT_SOURCE_DIR_PYBIND11=/tmp/sgt52-build/_deps/pybind11-src \
  --config-settings cmake.define.FETCHCONTENT_SOURCE_DIR_CARMA=/tmp/sgt52-build/_deps/carma-src \
  --config-settings cmake.define.FETCHCONTENT_SOURCE_DIR_CATCH2=/tmp/sgt52-build/_deps/catch2-src
```

The source-only overrides reuse verified dependency checkouts, never compiled binaries. Their exact revisions are in `build-provenance.txt`; when those directories are unavailable, omit the four overrides and let the revision-pinned CMake declarations fetch the same sources. `cmake.args` explicitly overrides the package's default tests-OFF setting. An initial configure using only `cmake.define.SGTLEARN_BUILD_TESTS=ON` was overridden by that default; the corrected configuration enabled tests without changing the matching Release core flags. All three native modules were built in the fresh candidate build and installed from its wheel.

The tested native binary is `/tmp/sgtlearn-candidate/build/cpp_tests`. CTest ran from this same build directory. Python tests imported the isolated wheel while run from a neutral working directory with importlib import mode; the root task recorded 471 passed, 2 skipped, 31 expected MAE warnings and 44/44 native cases. Correctness validation completed before timed measurement, and no source/build changes occurred during the common, confirmation or branching experiments.

The Python environment contains pybind11 3.1.0 for build tooling, matching baseline; native CMake FetchContent uses pybind11 v2.13.6 (`a2e59f0e7065404b44dfe92a28aca47ba1378dc4`), also matching baseline. These are separate recorded dependencies, not an unreported native version mismatch.
