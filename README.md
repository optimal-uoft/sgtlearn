# SGTLearn
![sgt visualization](assets/SGT_Viz.png)

`sgtlearn` is a Python package for learning [Shape Generalized Trees (SGTs)](https://neurips.cc/virtual/2025/loc/san-diego/poster/115950).

- 🌳 **Shape Generalized Trees (SGTs):** A class of decision trees where each node applies a learnable, axis-aligned shape function to a feature for non-linear and interpretable splits.
- 👁 **Interpretability:** Each node's shape function can be visualized directly.
- ⚡ **ShapeCART Algorithm:** An efficient induction method for learning SGTs from data.
- 🔀 **Extensions:**
  - **Shape²GT (S²GT):** Bivariate shape functions for richer splits.
  - **SGT<sub>K</sub>:** Multi-way branching generalization.
  - **Shape²CART & ShapeCART<sub>K</sub>:** Algorithms for learning S²GTs and SGT<sub>K</sub>s.

## Developer Setup

Use a **project-local virtual environment** (`.venv`) so Python, pytest, and
scikit-learn stay isolated and reproducible:

```bash
# Requires Python >= 3.11
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```
The editable install builds the C++ extensions via scikit-build-core and
installs the `sgtlearn` package plus native modules into `.venv`.

Alternatively, one can use `uv` 

```bash
uv sync --all-extras
source .venv/bin/activate
```

> **Anaconda users:** Do not bootstrap the venv from an Anaconda Python.
> Anaconda ships a `libstdc++.so.6` that lags the symbol versions produced by
> recent system compilers (gcc ≥ 13), so the install succeeds but
> `import sgtlearn` fails with `ImportError: GLIBCXX_3.4.NN not found`. Use a
> non-Anaconda Python — e.g. `uv venv --python 3.12 .venv` (downloads a
> hermetic CPython), `pyenv`, or your distro's `python3`.


For a **non-editable** install into the active environment only:

```bash
pip install .
pip install ".[dev]"   # dev extras (pytest, scikit-learn) only if needed
```

## Build Workflow (scikit-build + CMake)

`pip install .` drives this build path:

1. `pyproject.toml` selects `scikit_build_core.build` as the backend.
2. CMake is configured from `cpp/CMakeLists.txt`.
3. Each file in `cpp/bindings/*.cpp` becomes one pybind11 module target.
4. After each module is built, `pybind11-stubgen` generates a matching `.pyi`.
5. The `.pyi` is generated and installed in the same location as the module `.so`.

## C++ Folder Conventions

- `cpp/include/sgtlearn/`: public headers for the core C++ API.
- `cpp/src/`: internal C++ implementation for the core library.
- `cpp/bindings/`: pybind11 binding entrypoints; one `.cpp` file maps to one Python extension module.
- `cpp/tests/`: C++ unit tests consumed by the `cpp_tests` executable target.

## CMake Targets

- `sgtlearn_core` (static library): shared C++ logic used by Python modules and tests.
- `<module_name>` (pybind11 module, one per file in `cpp/bindings/`): compiled extension modules installed into the package.
- `cpp_tests` (Catch2 executable): optional C++ test target, controlled by:
  - `-DSGTLEARN_BUILD_TESTS=ON` (build C++ tests)
  - `-DSGTLEARN_BUILD_TESTS=OFF` (default for `pip install`; the CMake option itself defaults to `ON`, but `pyproject.toml` overrides this so wheels don't ship test binaries)

### Overriding CMake options from `pip`

Example (build C++ tests for one install):

```bash
pip install . --config-settings=cmake.args="-DSGTLEARN_BUILD_TESTS=ON"
```


## Quick Start

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sgtlearn

X, y = make_classification(
    n_samples=500,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = sgtlearn.SGTClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome. Please feel free to submit a pull request.

## Citation

If you use this package in your research, please cite:

```text
@inproceedings{upadhyaempowering,
  title={Empowering Decision Trees via Shape Function Branching},
  author={Upadhya, Nakul and Cohen, Eldan},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

Additionally, check out our other works on our [lab website](https://optimal.mie.utoronto.ca/).
