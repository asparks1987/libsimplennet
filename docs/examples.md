# Examples

## Python integer output

```python
from simplennet import SimplePredictor

predictor = SimplePredictor(output_type=int, epochs=30, learning_rate=0.001, hidden_units=(4,))
predictor.fit([[0], [1], [2], [3]], [0, 2, 4, 6])

print(predictor.predict([[4]]))
```

## Python categorical output

```python
from simplennet import SimplePredictor

predictor = SimplePredictor(output_type="category", epochs=30, learning_rate=0.05, hidden_units=(4,))
predictor.fit([[0.1, 0.9], [0.8, 0.2]], ["cold", "hot"])

print(predictor.predict([[0.85, 0.15]]))
```

## C++

```cpp
simplennet::TrainingOptions options;
options.epochs = 30;
options.learning_rate = 0.001;
options.hidden_layers = {4};

simplennet::SimplePredictor predictor(simplennet::OutputType::Int, options);
predictor.fit({{0}, {1}, {2}, {3}}, {0, 2, 4, 6});
```
