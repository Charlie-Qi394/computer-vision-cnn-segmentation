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


def build_fpn_aspp_attention_segmentation(
    input_shape: tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 21,
    fpn_filters: int = 128,
):
    """Build a compact FPN + ASPP + attention-gated segmentation model.

    This is a reusable, from-scratch portfolio implementation of the design
    explored in the coursework. It uses separable convolutions and controlled
    channel widths for computational efficiency; it does not perform pruning.
    """
    keras = _keras()
    layers = keras.layers

    def conv_block(x, filters: int):
        x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU()(x)

    def aspp_block(x, filters: int):
        branches = [
            layers.Conv2D(filters, 1, padding="same", use_bias=False)(x),
            layers.SeparableConv2D(filters, 3, dilation_rate=6, padding="same", use_bias=False)(x),
            layers.SeparableConv2D(filters, 3, dilation_rate=12, padding="same", use_bias=False)(x),
            layers.SeparableConv2D(filters, 3, dilation_rate=18, padding="same", use_bias=False)(x),
        ]
        branches = [layers.ReLU()(layers.BatchNormalization()(branch)) for branch in branches]

        pooled = layers.GlobalAveragePooling2D(keepdims=True)(x)
        pooled = layers.Conv2D(filters, 1, padding="same", use_bias=False)(pooled)
        pooled = layers.ReLU()(layers.BatchNormalization()(pooled))
        pooled = layers.UpSampling2D(
            size=(x.shape[1], x.shape[2]), interpolation="bilinear"
        )(pooled)

        merged = layers.Concatenate()(branches + [pooled])
        merged = layers.Conv2D(filters, 1, padding="same", use_bias=False)(merged)
        return layers.ReLU()(layers.BatchNormalization()(merged))

    def attention_gate(skip, decoder, filters: int):
        gate = layers.UpSampling2D(size=2, interpolation="bilinear")(decoder)
        gate = layers.Conv2D(filters, 1, padding="same")(gate)
        skip_projection = layers.Conv2D(filters, 1, padding="same")(skip)
        score = layers.ReLU()(layers.Add()([gate, skip_projection]))
        score = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(score)
        return layers.Multiply()([skip, score])

    def decoder_block(x, skip, filters: int):
        attended_skip = attention_gate(skip, x, max(filters // 2, 1))
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        x = layers.Concatenate()([x, attended_skip])
        return conv_block(x, filters)

    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)

    c1 = conv_block(x, 32)
    c2 = conv_block(layers.MaxPooling2D()(c1), 64)
    c3 = conv_block(layers.MaxPooling2D()(c2), 128)
    c4 = conv_block(layers.MaxPooling2D()(c3), 192)
    bottleneck = conv_block(layers.MaxPooling2D()(c4), 256)

    # Feature-pyramid lateral projections.
    p4 = layers.Conv2D(fpn_filters, 1, padding="same")(bottleneck)
    p3 = layers.Add()([
        layers.UpSampling2D(size=2, interpolation="bilinear")(p4),
        layers.Conv2D(fpn_filters, 1, padding="same")(c4),
    ])
    p2 = layers.Add()([
        layers.UpSampling2D(size=2, interpolation="bilinear")(p3),
        layers.Conv2D(fpn_filters, 1, padding="same")(c3),
    ])
    p1 = layers.Add()([
        layers.UpSampling2D(size=2, interpolation="bilinear")(p2),
        layers.Conv2D(fpn_filters, 1, padding="same")(c2),
    ])

    x = aspp_block(bottleneck, fpn_filters)
    x = decoder_block(x, p3, fpn_filters)
    x = decoder_block(x, p2, 96)
    x = decoder_block(x, p1, 64)
    x = decoder_block(x, c1, 32)
    outputs = layers.Conv2D(num_classes, 1, activation="softmax", name="segmentation_mask")(x)
    return keras.Model(inputs, outputs, name="fpn_aspp_attention_segmentation")
