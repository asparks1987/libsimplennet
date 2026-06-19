# User Guide

## Train

Create a predictor with an output type and call `fit`.

```python
from simplennet import SimplePredictor

predictor = SimplePredictor(output_type=float, epochs=30, learning_rate=0.000001, hidden_units=(4,))
predictor.fit([[600], [900], [1200]], [1200.0, 1800.0, 2400.0])
```

## Predict

```python
single = predictor.predict([[1000]])
many = predictor.predict([[1000], [1300]])
```

One input row returns one value. Multiple input rows return a list.

## Save

```python
predictor.save("rent.snet")
```

## Load

```python
loaded = SimplePredictor.load("rent.snet")
print(loaded.predict([[1000]]))
```

The same `.snet` directory is shared by the native core and all language wrappers.
