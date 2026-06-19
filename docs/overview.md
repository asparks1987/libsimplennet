# libsimplennet

libsimplennet is a simple neural-network SDK for projects that need quick predictions without a heavy machine-learning stack.

Import it, choose the output type, train from examples, and call `predict`. Pass numeric features directly, or serialize arbitrary app data to JSON/text and use the portable text API.

## The Pitch

Most apps do not need a giant training pipeline just to learn a useful pattern from structured data. libsimplennet gives you a small dense network that can be embedded into ordinary projects and called from the languages teams already use.

Runtime support:

- Python
- C++
- Java
- Go

Output types:

- `int`
- `float`
- `category`

## Quick Example

```python
from simplennet import SimplePredictor

predictor = SimplePredictor(output_type=int, epochs=30, learning_rate=0.001, hidden_units=(4,))
predictor.fit([[0], [1], [2], [3]], [0, 2, 4, 6])

print(predictor.predict([[4]]))
```

## Any App Data

For custom objects, event payloads, DTOs, game state, sensor data, carts, or user profiles, serialize the value to a stable string and train with `fit_text`.

```python
events = [
    '{"kind":"cart","items":1}',
    '{"kind":"cart","items":2}',
    '{"kind":"cart","items":3}',
]

predictor = SimplePredictor(output_type=int, epochs=30, learning_rate=0.001)
predictor.fit_text(events, [2, 4, 6])

print(predictor.predict_text(['{"kind":"cart","items":4}']))
```

The text/JSON path is available from Python, C++, Java, Go, and the C ABI.

## Why Use It?

- Plug prediction into an app without building a full ML platform.
- Train on simple tabular examples.
- Learn from serialized JSON/text payloads for arbitrary app data.
- Return native typed values your code already understands.
- Save trained models as portable `.snet` directories.
- Reuse the same native core across supported language wrappers.

For the standalone GitHub Pages product website, open `docs/index.html`.
