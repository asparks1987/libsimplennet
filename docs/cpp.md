# C++ Quickstart

The C++ SDK is the native v1 runtime surface.

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

Build the core and run tests:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The public headers are in `include/simplennet`.

For languages outside the first-party wrappers, bind to `include/simplennet/c_api.h` and the `simplennet` shared library. The C ABI includes numeric and text/JSON fit/predict functions, so custom runtimes can share the same `.snet` model format.
