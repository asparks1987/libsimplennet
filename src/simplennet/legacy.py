"""Compatibility helpers for the historical ``libSimpleNNet`` API."""

from __future__ import annotations

import warnings
from typing import Any

from . import OutputType, SimplePredictor

warnings.warn(
    "libSimpleNNet is a legacy compatibility module. Import `simplennet` and "
    "`SimplePredictor` for new projects.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "OutputType",
    "SimplePredictor",
    "preprocess_data",
    "preprocess_data_array",
    "create_model",
    "train_model",
    "save_model",
    "load_model_from_file",
    "ProgressCallback",
]


def preprocess_data(df: Any, target_col: str):
    """Legacy helper that converts an ``input`` column and target column to arrays."""

    import numpy as np

    if "input" not in df:
        raise ValueError("Expected a dataframe with an 'input' column")
    if target_col not in df:
        raise ValueError(f"Expected target column {target_col!r}")

    working = df.copy()
    if not isinstance(working["input"].iloc[0], np.ndarray):
        working["input"] = working["input"].apply(lambda value: np.asarray([float(item) for item in str(value).split(",")]))

    targets = working[target_col].tolist()
    labels: list[Any] = []
    for value in targets:
        if value not in labels:
            labels.append(value)
    encoded = np.asarray([labels.index(value) for value in targets])
    X = np.stack(working["input"].values)
    return X, encoded


def preprocess_data_array(input_array: Any):
    """Legacy helper that converts comma-separated input to a NumPy array."""

    import numpy as np

    if isinstance(input_array, np.ndarray):
        return input_array
    if isinstance(input_array, str):
        return np.asarray([float(value) for value in input_array.split(",")])
    return np.asarray(input_array, dtype="float32")


def create_model(input_shape: tuple[int, ...]):
    """Create the legacy LSTM model."""

    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(50, activation="relu", input_shape=input_shape),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train_model(X: Any, y: Any, callback: Any = None, epochs: int = 1):
    """Train the legacy LSTM model and return ``(model, history)``."""

    import numpy as np
    from sklearn.model_selection import train_test_split

    X = np.asarray(X)
    y = np.asarray(y)
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if len(X_train.shape) == 2:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    if len(X_test.shape) == 2:
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = create_model((X_train.shape[1], X_train.shape[2]))
    callbacks = [callback] if callback is not None else None
    history = model.fit(X_train, y_train, epochs=epochs, verbose=0, validation_data=(X_test, y_test), callbacks=callbacks)
    return model, history


def save_model(model: Any, file_path: str) -> None:
    """Save a Keras model to disk."""

    model.save(file_path)


def load_model_from_file(file_path: str):
    """Load a Keras model from disk."""

    import tensorflow as tf

    return tf.keras.models.load_model(file_path)


class ProgressCallback:
    """Legacy callback factory for GUI progress updates."""

    def __new__(cls, gui: Any):
        import tensorflow as tf

        class _ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, gui: Any) -> None:
                super().__init__()
                self.gui = gui

            def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
                logs = logs or {}
                loss = logs.get("loss")
                self.gui.losses.append(loss)
                self.gui.update_loss_plot(self.gui.losses, epoch)
                self.gui.update_progress((epoch + 1) / int(self.gui.epoch_spinbox.get()) * 100)

        return _ProgressCallback(gui)
