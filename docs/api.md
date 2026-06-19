# API Reference

## Python

```python
SimplePredictor(
    output_type=None,
    *,
    epochs=50,
    hidden_units=(32, 16),
    learning_rate=0.01,
    random_state=42,
)
```

Methods:

- `fit(X, y)`: trains the native model.
- `predict(X)`: returns one value for one row or a list for many rows.
- `save(path)`: writes a `.snet` directory.
- `SimplePredictor.load(path)`: loads a `.snet` directory.

## C++

```cpp
simplennet::SimplePredictor predictor(simplennet::OutputType::Int, options);
predictor.fit(inputs, targets);
auto values = predictor.predict_ints(inputs);
predictor.save("model.snet");
auto loaded = simplennet::SimplePredictor::load("model.snet");
```

## C ABI

The C ABI is declared in `include/simplennet/c_api.h` and uses opaque handles:

- `snet_create`
- `snet_destroy`
- `snet_fit_numeric`
- `snet_fit_labels`
- `snet_predict_numeric`
- `snet_save`
- `snet_load`
- `snet_last_error`
