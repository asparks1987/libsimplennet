"""Python facade for the native libsimplennet core."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from .encoding import UniversalEncoder

try:
    from . import _native
except ImportError:  # pragma: no cover - exercised only before extension build
    _native = None

OutputKind = Literal["int", "float", "category"]


@dataclass(frozen=True)
class OutputType:
    """Declares how predictions should be converted back to Python values."""

    kind: OutputKind

    @classmethod
    def from_value(cls, value: "OutputType | type | str | None") -> "OutputType | None":
        if value is None:
            return None
        if isinstance(value, OutputType):
            return value
        if value is int or value in {"int", "integer"}:
            return cls("int")
        if value is float or value in {"float", "double"}:
            return cls("float")
        if value is str or value in {"str", "string", "category", "label", "categorical"}:
            return cls("category")
        raise ValueError("output_type must be int, float, str, 'int', 'float', or 'category'")


class SimplePredictor:
    """Typed supervised predictor backed by the native C++ libsimplennet core."""

    def __init__(
        self,
        output_type: OutputType | type | str | None = None,
        *,
        epochs: int = 50,
        hidden_units: Sequence[int] = (32, 16),
        validation_split: float = 0.0,
        learning_rate: float = 0.01,
        random_state: int | None = 42,
        verbose: int = 0,
        feature_width: int = 64,
    ) -> None:
        self.output_type = OutputType.from_value(output_type)
        self.epochs = int(epochs)
        self.hidden_units = tuple(int(unit) for unit in hidden_units)
        self.validation_split = validation_split
        self.learning_rate = float(learning_rate)
        self.random_state = 42 if random_state is None else int(random_state)
        self.verbose = verbose
        self.feature_width = int(feature_width)

        self._model: Any | None = None
        self.feature_names_: list[str] | None = None
        self.n_features_: int | None = None
        self.classes_: list[str] | None = None
        self.input_encoding_: str = "numeric"
        self._encoder: UniversalEncoder | None = None

    def fit(self, X: Any, y: Sequence[Any], **fit_kwargs: Any) -> "SimplePredictor":
        """Train the predictor and return ``self``."""

        self._require_native()
        targets = list(y)
        if not targets:
            raise ValueError("y must contain at least one target value")

        matrix = self._normalize_X(X, fitting=True, y_length=len(targets))
        if len(matrix) != len(targets):
            raise ValueError("X and y must contain the same number of samples")

        if self.output_type is None:
            self.output_type = self._infer_output_type(targets)

        epochs = int(fit_kwargs.pop("epochs", self.epochs))
        learning_rate = float(fit_kwargs.pop("learning_rate", self.learning_rate))
        if fit_kwargs:
            unsupported = ", ".join(sorted(fit_kwargs))
            raise TypeError(f"Unsupported fit option(s): {unsupported}")

        self._model = _native.NativePredictor(
            self.output_type.kind,
            epochs,
            learning_rate,
            list(self.hidden_units),
            self.random_state,
        )
        if self.feature_names_ is not None:
            self._model.feature_names = self.feature_names_

        if self.output_type.kind == "category":
            self._model.fit_labels(matrix, [str(target) for target in targets])
            self.classes_ = list(self._model.class_labels)
        else:
            self._model.fit_numeric(matrix, [float(target) for target in targets])
            self.classes_ = None
        return self

    def predict(self, X: Any) -> Any:
        """Predict one value for a single row or a list of values for multiple rows."""

        if self._model is None:
            raise RuntimeError("SimplePredictor must be fit or loaded before predict is called")
        assert self.output_type is not None

        matrix = self._normalize_X(X, fitting=False)
        if self.output_type.kind == "category":
            values = list(self._model.predict_labels(matrix))
        elif self.output_type.kind == "int":
            values = [int(value) for value in self._model.predict_ints(matrix)]
        else:
            values = [float(value) for value in self._model.predict_numbers(matrix)]
        return values[0] if len(values) == 1 else values

    def fit_text(self, inputs: Sequence[str], y: Sequence[Any], *, width: int | None = None, **fit_kwargs: Any) -> "SimplePredictor":
        """Train from serialized text or JSON inputs using the portable native encoder."""

        self._require_native()
        input_rows = [str(value) for value in inputs]
        targets = list(y)
        if not input_rows:
            raise ValueError("inputs must contain at least one value")
        if len(input_rows) != len(targets):
            raise ValueError("inputs and y must contain the same number of samples")

        if self.output_type is None:
            self.output_type = self._infer_output_type(targets)

        epochs = int(fit_kwargs.pop("epochs", self.epochs))
        learning_rate = float(fit_kwargs.pop("learning_rate", self.learning_rate))
        if fit_kwargs:
            unsupported = ", ".join(sorted(fit_kwargs))
            raise TypeError(f"Unsupported fit option(s): {unsupported}")

        text_width = self.feature_width if width is None else int(width)
        self._model = _native.NativePredictor(
            self.output_type.kind,
            epochs,
            learning_rate,
            list(self.hidden_units),
            self.random_state,
        )

        if self.output_type.kind == "category":
            self._model.fit_text_labels(input_rows, [str(target) for target in targets], text_width)
            self.classes_ = list(self._model.class_labels)
        else:
            self._model.fit_text(input_rows, [float(target) for target in targets], text_width)
            self.classes_ = None

        self.feature_width = text_width
        self.n_features_ = text_width
        self.feature_names_ = None
        self.input_encoding_ = "text"
        self._encoder = None
        return self

    def predict_text(self, inputs: Sequence[str]) -> Any:
        """Predict from serialized text or JSON inputs using the portable native encoder."""

        if self._model is None:
            raise RuntimeError("SimplePredictor must be fit or loaded before predict_text is called")
        assert self.output_type is not None
        input_rows = [str(value) for value in inputs]
        if not input_rows:
            raise ValueError("inputs must contain at least one value")

        if self.output_type.kind == "category":
            values = list(self._model.predict_text_labels(input_rows))
        elif self.output_type.kind == "int":
            values = [int(value) for value in self._model.predict_text_ints(input_rows)]
        else:
            values = [float(value) for value in self._model.predict_text_numbers(input_rows)]
        return values[0] if len(values) == 1 else values

    def save(self, path: str | Path) -> Path:
        """Save the predictor as a portable ``.snet`` directory."""

        if self._model is None:
            raise RuntimeError("SimplePredictor must be fit before it can be saved")
        target = Path(path)
        self._model.save(target)
        (target / "python_metadata.json").write_text(json.dumps(self._python_metadata(), indent=2), encoding="utf-8")
        if self._encoder is not None:
            self._encoder.save(target / "encoder.json")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "SimplePredictor":
        """Load a predictor saved with :meth:`save`."""

        cls._require_native()
        native_model = _native.NativePredictor.load(Path(path))
        predictor = cls(output_type=native_model.output_type)
        predictor._model = native_model
        predictor.output_type = OutputType.from_value(native_model.output_type)
        predictor.feature_names_ = list(native_model.feature_names)
        predictor.n_features_ = None
        predictor.classes_ = list(native_model.class_labels)
        metadata_path = Path(path) / "python_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            predictor.feature_width = int(metadata.get("feature_width", predictor.feature_width))
            predictor.n_features_ = metadata.get("n_features")
            predictor.input_encoding_ = metadata.get("input_encoding", "numeric")
            if predictor.input_encoding_ == "universal":
                predictor._encoder = UniversalEncoder.load(Path(path) / "encoder.json")
        return predictor

    @staticmethod
    def native_available() -> bool:
        """Return whether the native extension is importable."""

        return _native is not None

    @staticmethod
    def _require_native() -> None:
        if _native is None:
            raise ImportError(
                "libsimplennet's native extension is not built. Install the package with "
                "`python -m pip install -e .` or build a wheel before training/predicting."
            )

    def _normalize_X(self, X: Any, *, fitting: bool, y_length: int | None = None) -> list[list[float]]:
        feature_names: list[str] | None = None
        universal_source = X

        if _is_pandas_dataframe(X):
            feature_names = [str(column) for column in X.columns]
            raw = X.to_numpy().tolist()
            universal_source = X.to_dict(orient="records")
        elif isinstance(X, dict):
            feature_names = self._dict_feature_names(X.keys(), fitting=fitting)
            raw = [[X[name] for name in feature_names]]
            universal_source = X
        elif _is_sequence_of_dicts(X):
            rows = list(X)
            feature_names = self._dict_feature_names(rows[0].keys(), fitting=fitting)
            raw = [[row[name] for name in feature_names] for row in rows]
            universal_source = rows
        elif hasattr(X, "tolist"):
            raw = X.tolist()
        else:
            raw = X

        try:
            matrix = self._coerce_matrix(raw, fitting=fitting, y_length=y_length)
            encoding = "numeric"
        except (TypeError, ValueError):
            if not fitting and self.input_encoding_ != "universal":
                raise
            matrix = self._encode_universal(universal_source, fitting=fitting, y_length=y_length)
            encoding = "universal"

        if fitting:
            self.n_features_ = len(matrix[0])
            self.input_encoding_ = encoding
            if feature_names is not None:
                self.feature_names_ = feature_names
        elif self.n_features_ is not None and len(matrix[0]) != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, received {len(matrix[0])}")

        return matrix

    def _encode_universal(self, raw: Any, *, fitting: bool, y_length: int | None) -> list[list[float]]:
        if fitting or self._encoder is None:
            self._encoder = UniversalEncoder(width=self.feature_width)

        rows = self._row_values(raw, fitting=fitting, y_length=y_length)
        return self._encoder.transform_many(rows)

    def _coerce_matrix(self, raw: Any, *, fitting: bool, y_length: int | None) -> list[list[float]]:
        if _is_scalar(raw):
            return [[float(raw)]]

        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raw = list(raw)

        rows = list(raw)
        if not rows:
            raise ValueError("X must contain at least one row")

        if all(_is_scalar(value) for value in rows):
            values = [float(value) for value in rows]
            if fitting and y_length is not None and y_length == len(values):
                return [[value] for value in values]
            if not fitting and self.n_features_ == 1 and len(values) != 1:
                return [[value] for value in values]
            return [values]

        matrix = [[float(value) for value in row] for row in rows]
        width = len(matrix[0])
        if width == 0:
            raise ValueError("X rows must contain at least one feature")
        if any(len(row) != width for row in matrix):
            raise ValueError("all X rows must have the same number of features")
        return matrix

    def _row_values(self, raw: Any, *, fitting: bool, y_length: int | None) -> list[Any]:
        if _is_scalar(raw) or isinstance(raw, (str, bytes, dict)):
            return [raw]
        if not isinstance(raw, Sequence):
            return [raw]
        rows = list(raw)
        if not rows:
            raise ValueError("X must contain at least one row")
        if fitting and y_length is not None and y_length == len(rows):
            return rows
        if not fitting and self.n_features_ == self.feature_width and len(rows) != self.feature_width:
            return rows
        return [raw]

    def _dict_feature_names(self, keys: Iterable[Any], *, fitting: bool) -> list[str]:
        incoming = [str(key) for key in keys]
        if fitting or self.feature_names_ is None:
            return incoming
        missing = [name for name in self.feature_names_ if name not in incoming]
        if missing:
            raise ValueError(f"Missing feature(s): {', '.join(missing)}")
        return self.feature_names_

    def _infer_output_type(self, targets: list[Any]) -> OutputType:
        if all(isinstance(value, int) and not isinstance(value, bool) for value in targets):
            return OutputType("int")
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in targets):
            return OutputType("float")
        return OutputType("category")

    def _python_metadata(self) -> dict[str, Any]:
        return {
            "input_encoding": self.input_encoding_,
            "feature_width": self.feature_width,
            "n_features": self.n_features_,
            "feature_names": self.feature_names_,
        }


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_pandas_dataframe(value: Any) -> bool:
    return value.__class__.__name__ == "DataFrame" and hasattr(value, "to_numpy") and hasattr(value, "columns")


def _is_sequence_of_dicts(value: Any) -> bool:
    if isinstance(value, (dict, str, bytes)) or not isinstance(value, Sequence):
        return False
    return len(value) > 0 and all(isinstance(row, dict) for row in value)
