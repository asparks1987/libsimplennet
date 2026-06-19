from simplennet import SimplePredictor


X = [
    {"rooms": 1, "sqft": 600},
    {"rooms": 2, "sqft": 900},
    {"rooms": 3, "sqft": 1200},
]
y = [1200.0, 1800.0, 2400.0]

predictor = SimplePredictor(output_type=float, epochs=100)
predictor.fit(X, y)

print(predictor.predict({"rooms": 2, "sqft": 950}))
