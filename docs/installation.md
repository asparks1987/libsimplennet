# Installation

## Python

Install from a local checkout:

```bash
python -m pip install -e ".[dev]"
```

Build a wheel:

```bash
python -m pip wheel . --no-deps
```

The Python package uses scikit-build-core, CMake, and pybind11 to compile the native extension.

## C++

Build the native core:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The public headers are installed from `include/simplennet`.

## Docs

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```
