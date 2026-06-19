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
- `fit_text(inputs, y)`: trains from portable serialized text or JSON inputs.
- `predict_text(inputs)`: predicts from serialized text or JSON inputs.
- `save(path)`: writes a `.snet` directory.
- `SimplePredictor.load(path)`: loads a `.snet` directory.

## C++

```cpp
simplennet::SimplePredictor predictor(simplennet::OutputType::Int, options);
predictor.fit(inputs, targets);
auto values = predictor.predict_ints(inputs);
predictor.fit_text({R"({"kind":"cart","items":1})"}, {2});
auto text_values = predictor.predict_text_ints({R"({"kind":"cart","items":2})"});
predictor.save("model.snet");
auto loaded = simplennet::SimplePredictor::load("model.snet");
```

## C ABI

The C ABI is declared in `include/simplennet/c_api.h` and uses opaque handles:

- `snet_create`
- `snet_destroy`
- `snet_fit_numeric`
- `snet_fit_labels`
- `snet_fit_text_numeric_bits`
- `snet_fit_text_labels_bits`
- `snet_predict_numeric`
- `snet_predict_text_numeric`
- `snet_predict_text_ints`
- `snet_predict_text_class_indices`
- `snet_save`
- `snet_load`
- `snet_last_error`
