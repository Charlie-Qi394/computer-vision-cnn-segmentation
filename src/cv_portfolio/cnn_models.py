"""Optional Keras CNN model definitions."""

from __future__ import annotations


def _keras():
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for model builders. Install with: pip install -e '.[deep-learning]'"
        ) from exc
    return keras


def build_cifar_baseline(input_shape: tuple[int, int, int] = (32, 32, 3), num_classes: int = 100):
    """Build a compact CIFAR-style CNN classifier."""
    keras = _keras()
    layers = keras.layers

    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="cifar_baseline_cnn")


def build_residual_cifar_model(input_shape: tuple[int, int, int] = (32, 32, 3), num_classes: int = 100):
    """Build a small residual CNN matching the portfolio experiment theme."""
    keras = _keras()
    layers = keras.layers

    def residual_block(x, filters: int):
        shortcut = x
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, padding="same")(shortcut)
        x = layers.Add()([x, shortcut])
        return layers.Activation("relu")(x)

    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = residual_block(x, 32)
    x = layers.MaxPooling2D()(x)
    x = residual_block(x, 64)
    x = layers.MaxPooling2D()(x)
    x = residual_block(x, 128)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="cifar_residual_cnn")
