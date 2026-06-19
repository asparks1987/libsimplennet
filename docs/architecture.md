# Architecture

libsimplennet is organized around one shared native core.

## Native core

The C++17 core implements a small dense MLP for supervised tabular prediction. It supports:

- Integer and float regression
- Categorical classification
- Deterministic initialization with a seed
- Full-batch gradient descent
- Portable `.snet` save/load

## C ABI

The C ABI in `include/simplennet/c_api.h` is the stable boundary for wrappers in Go, C#, Rust, JavaScript native addons, Ruby, Swift, and other runtimes that can call a C-compatible shared library. CMake builds this boundary as the `simplennet` shared library.

- Opaque predictor handles
- Plain row-major numeric arrays
- String arrays for serialized text/JSON inputs
- String arrays for categorical labels
- Integer return codes
- `snet_last_error()` for diagnostics

## Python binding

The Python package keeps the friendly `SimplePredictor` API and delegates training, prediction, save, and load to the pybind11 native extension.

## Java and Go bindings

Java and Go wrap the shared native DLL and expose the same model semantics as Python and C++.

- Java uses JNI and `System.loadLibrary`.
- Go uses the exported C ABI from `simplennet.dll`.
- Both can train, predict, save, and load the same `.snet` model directories.

## `.snet` format

A saved model is a directory containing:

- `model.json`: architecture, output type, feature names, class labels, training config, and `snet_format_version`
- `weights.bin`: little-endian binary weights and biases

Format version `1` is the first portable model format.
