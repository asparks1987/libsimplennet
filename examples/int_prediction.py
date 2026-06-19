from simplennet import SimplePredictor


X = [[0], [1], [2], [3], [4], [5]]
y = [0, 2, 4, 6, 8, 10]

predictor = SimplePredictor(output_type=int, epochs=100)
predictor.fit(X, y)

print(predictor.predict([[6]]))
