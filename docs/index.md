# libsimplennet

libsimplennet is a simple neural-network SDK for projects that need quick predictions without a heavy machine-learning stack.

Import it, choose the output type, train from examples, and call `predict`.

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

## Why Use It?

- Plug prediction into an app without building a full ML platform.
- Train on simple tabular examples.
- Return native typed values your code already understands.
- Save trained models as portable `.snet` directories.
- Reuse the same native core across supported language wrappers.

For the standalone product website, open `docs/website/index.html`.
