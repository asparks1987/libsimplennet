import importlib
import warnings


def test_import_public_api():
    import simplennet
    from simplennet import OutputType, SimplePredictor

    assert simplennet.__version__
    assert OutputType("int").kind == "int"
    assert SimplePredictor(output_type=int).output_type.kind == "int"


def test_legacy_import_path_still_works():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = importlib.import_module("libSimpleNNet")

    assert hasattr(legacy, "SimplePredictor")
    assert hasattr(legacy, "preprocess_data_array")
