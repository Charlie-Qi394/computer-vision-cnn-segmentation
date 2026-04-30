"""Optional Keras segmentation model definitions."""

from __future__ import annotations


def _keras():
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for segmentation builders. Install with: pip install -e '.[deep-learning]'"
        ) from exc
    return keras


def build_small_unet(input_shape: tuple[int, int, int] = (128, 128, 3), num_classes: int = 21):
    """Build a compact U-Net style segmentation model."""
    keras = _keras()
    layers = keras.layers

    def conv_block(x, filters: int):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        return x

    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)

    c1 = conv_block(x, 32)
    p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)
    bridge = conv_block(p2, 128)

    u2 = layers.UpSampling2D()(bridge)
    u2 = layers.Concatenate()([u2, c2])
    c3 = conv_block(u2, 64)
    u1 = layers.UpSampling2D()(c3)
    u1 = layers.Concatenate()([u1, c1])
    c4 = conv_block(u1, 32)

    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(c4)
    return keras.Model(inputs, outputs, name="small_unet_segmentation")


def build_fcn_baseline(input_shape: tuple[int, int, int] = (128, 128, 3), num_classes: int = 21):
    """Build a simple fully convolutional segmentation baseline."""
    keras = _keras()
    layers = keras.layers

    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.UpSampling2D(size=4, interpolation="bilinear")(x)
    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="fcn_segmentation_baseline")
