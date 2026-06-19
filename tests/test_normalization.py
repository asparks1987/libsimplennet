import pytest

from simplennet import SimplePredictor


def test_normalization_accepts_lists():
    predictor = SimplePredictor()

    list_array = predictor._normalize_X([[1, 2], [3, 4]], fitting=True, y_length=2)
    one_row = predictor._normalize_X([5, 6], fitting=False)

    assert list_array == [[1.0, 2.0], [3.0, 4.0]]
    assert one_row == [[5.0, 6.0]]


def test_normalization_accepts_numpy_arrays_when_available():
    np = pytest.importorskip("numpy")
    predictor = SimplePredictor()

    array = predictor._normalize_X(np.asarray([[1, 2], [3, 4]]), fitting=True, y_length=2)

    assert array == [[1.0, 2.0], [3.0, 4.0]]


def test_normalization_accepts_row_dicts_and_preserves_feature_order():
    predictor = SimplePredictor()

    predictor._normalize_X([{"a": 1, "b": 2}, {"a": 3, "b": 4}], fitting=True, y_length=2)
    row = predictor._normalize_X({"b": 20, "a": 10}, fitting=False)

    assert predictor.feature_names_ == ["a", "b"]
    assert row == [[10.0, 20.0]]


def test_normalization_accepts_pandas_dataframes_when_available():
    pd = pytest.importorskip("pandas")
    predictor = SimplePredictor()

    frame = pd.DataFrame({"left": [1, 2], "right": [3, 4]})
    array = predictor._normalize_X(frame, fitting=True, y_length=2)

    assert predictor.feature_names_ == ["left", "right"]
    assert array == [[1.0, 3.0], [2.0, 4.0]]


def test_normalization_rejects_wrong_feature_count():
    predictor = SimplePredictor()
    predictor._normalize_X([[1, 2]], fitting=True, y_length=1)

    with pytest.raises(ValueError, match="Expected 2 features"):
        predictor._normalize_X([[1, 2, 3]], fitting=False)
