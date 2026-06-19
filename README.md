# libsimplennet

**Plug a small neural network into any project and get quick typed predictions.**

libsimplennet is a lightweight multi-language prediction SDK. Import it, choose the output type you want, train it with examples from your app, and call `predict`. It is built for practical use cases where you need a simple supervised network without dragging a research stack into the project.

The same native C++ core powers Python, C++, Java, and Go, so models can be saved once as `.snet` directories and reused across supported runtimes.

## Why libsimplennet?

- **Simple mental model:** choose `int`, `float`, or `category`, then `fit` and `predict`.
- **Fast to embed:** designed for tools, services, scripts, desktop apps, and local workflows.
- **Multi-language by design:** C++ core, Python wrapper, Java JNI wrapper, and Go native-DLL wrapper.
- **Portable model files:** save trained networks as `.snet` directories with metadata and weights.
- **No heavy ML ceremony:** a small dense neural network for quick supervised tabular predictions.

## Good fits

Use libsimplennet when your project needs quick predictions from structured examples:

- Predict integer outputs such as counts, scores, levels, or ranks.
- Predict floating-point outputs such as price, duration, demand, confidence, or utilization.
- Predict category labels for routing, classification, decisions, or UI behavior.
- Train a local model and reload it later without rebuilding the network each run.

It is intentionally small. For large-scale deep learning, GPU training, images, audio, or huge model pipelines, use a full ML framework. For app-level prediction, libsimplennet keeps the path short.

## Python quickstart

Install from this checkout:

```bash
python -m pip install -e ".[dev]"
```

Train and predict an integer:

```python
from simplennet import SimplePredictor

predictor = SimplePredictor(
    output_type=int,
    epochs=30,
    learning_rate=0.001,
    hidden_units=(4,),
)

predictor.fit([[0], [1], [2], [3]], [0, 2, 4, 6])
print(predictor.predict([[4]]))
```

Save and load the trained network:

```python
predictor.save("double.snet")

loaded = SimplePredictor.load("double.snet")
print(loaded.predict([[5]]))
```

## C++ quickstart

```cpp
#include "simplennet/simple_predictor.hpp"

#include <iostream>

int main() {
    simplennet::TrainingOptions options;
    options.epochs = 30;
    options.learning_rate = 0.001;
    options.hidden_layers = {4};

    simplennet::SimplePredictor predictor(simplennet::OutputType::Int, options);
    predictor.fit({{0}, {1}, {2}, {3}}, {0, 2, 4, 6});

    std::cout << predictor.predict_ints({{4}}).front() << "\n";
}
```

Build and test the native core:

```bash
cmake -S . -B build-native
cmake --build build-native --config Release
ctest --test-dir build-native -C Release --output-on-failure
```

## Java quickstart

```java
SimplePredictor predictor = new SimplePredictor(OutputType.INT);
predictor.fit(new double[][]{{0}, {1}, {2}, {3}}, new double[]{0, 2, 4, 6});
int[] values = predictor.predictInts(new double[][]{{4}});
```

Run with the native DLL on the library path:

```bash
java -Djava.library.path=build-native/Release -cp java/target/classes ai.simplennet.SmokeTest
```

## Go quickstart

```go
predictor, err := simplennet.NewSimplePredictor(simplennet.OutputInt)
if err != nil {
    panic(err)
}
defer predictor.Close()

_ = predictor.Fit([][]float64{{0}, {1}, {2}, {3}}, []float64{0, 2, 4, 6})
values, _ := predictor.PredictInts([][]float64{{4}})
```

Set the native DLL path before running:

```powershell
$env:SIMPLENET_NATIVE_LIBRARY = "D:\path\to\build-native\Release\simplennet_native.dll"
go test ./...
```

## Website and docs

A standalone product website lives in `docs/website/index.html`. Open it directly in a browser.

The MkDocs content also lives in `docs/`:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

## Model format

`save()` writes a `.snet` directory:

- `model.json` stores architecture, output type, labels, feature names, training config, and `snet_format_version`.
- `weights.bin` stores little-endian weights and biases.

The format starts at `snet_format_version: 1`.

## Project layout

- `include/simplennet`: public C++ and C ABI headers
- `src/cpp`: shared native neural-network core
- `src/simplennet`: Python API facade
- `bindings/java`: JNI bridge
- `java`: Java wrapper and smoke test
- `go`: Go wrapper and tests
- `docs`: website and documentation

## Development

Run Python tests:

```bash
python -m pytest
```

Run C++ tests:

```bash
cmake -S . -B build-native
cmake --build build-native --config Release
ctest --test-dir build-native -C Release --output-on-failure
```

Build a Python wheel:

```bash
python -m pip wheel . --no-deps
```

Legacy imports through `libSimpleNNet` remain available, but new projects should use:

```python
from simplennet import SimplePredictor
```
