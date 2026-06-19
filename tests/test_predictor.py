import importlib.util

import pytest

from simplennet import SimplePredictor


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("simplennet._native") is None,
    reason="native extension has not been built",
)


def test_integer_prediction_returns_int():
    predictor = SimplePredictor(output_type=int, epochs=5, hidden_units=(4,), learning_rate=0.001)
    predictor.fit([[0], [1], [2], [3]], [0, 2, 4, 6])

    result = predictor.predict([[4]])

    assert isinstance(result, int)


def test_float_prediction_returns_float():
    predictor = SimplePredictor(output_type=float, epochs=5, hidden_units=(4,), learning_rate=0.000001)
    predictor.fit([[600], [900], [1200]], [1200.0, 1800.0, 2400.0])

    result = predictor.predict([[1000]])

    assert isinstance(result, float)


def test_category_prediction_returns_known_label():
    predictor = SimplePredictor(output_type="category", epochs=5, hidden_units=(4,), learning_rate=0.05)
    predictor.fit([[0.1, 0.9], [0.8, 0.2], [0.2, 0.7], [0.9, 0.1]], ["cold", "hot", "cold", "hot"])

    result = predictor.predict([[0.85, 0.15]])

    assert result in {"cold", "hot"}


def test_save_load_round_trip_preserves_prediction(tmp_path):
    predictor = SimplePredictor(output_type=int, epochs=5, hidden_units=(4,), learning_rate=0.001)
    predictor.fit([[0], [1], [2], [3]], [0, 2, 4, 6])
    before = predictor.predict([[4]])

    predictor.save(tmp_path / "double-predictor.snet")
    loaded = SimplePredictor.load(tmp_path / "double-predictor.snet")
    after = loaded.predict([[4]])

    assert isinstance(after, int)
    assert after == before
    assert (tmp_path / "double-predictor.snet" / "model.json").exists()
    assert (tmp_path / "double-predictor.snet" / "weights.bin").exists()
