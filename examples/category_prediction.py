from simplennet import SimplePredictor


X = [[0.1, 0.9], [0.8, 0.2], [0.2, 0.7], [0.9, 0.1]]
y = ["cold", "hot", "cold", "hot"]

predictor = SimplePredictor(output_type="category", epochs=50)
predictor.fit(X, y)

print(predictor.predict([[0.85, 0.15]]))
