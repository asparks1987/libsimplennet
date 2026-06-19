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
cmake -S . -B build-native
cmake --build build-native --config Release
ctest --test-dir build-native -C Release --output-on-failure
```

The public headers are installed from `include/simplennet`.

The build produces:

- `simplennet`: the language-neutral shared C ABI library.
- `simplennet_native`: the Java JNI bridge, when CMake can find JNI.
- `simplennet_core`: the C++ implementation library.

## Other Languages

Any runtime that can call a C-compatible shared library can bind to `include/simplennet/c_api.h`. Use numeric arrays for tabular data or string arrays for serialized JSON/text inputs.

## Docs

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```
