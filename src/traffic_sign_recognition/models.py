"""Model definitions and augmentation."""

import tensorflow as tf


def augmentation(seed: int = 42):
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.08, seed=seed),
        tf.keras.layers.RandomTranslation(0.08, 0.08, seed=seed + 1),
        tf.keras.layers.RandomZoom(0.1, seed=seed + 2),
        tf.keras.layers.RandomContrast(0.15, seed=seed + 3),
    ], name="augmentation")


def build_custom_cnn(image_size=64, num_classes=43, seed=42):
    inputs = tf.keras.Input((image_size, image_size, 3))
    x = augmentation(seed)(inputs)
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="custom_cnn")


def build_mobilenet_v2(image_size=96, num_classes=43, pretrained=True, seed=42):
    inputs = tf.keras.Input((image_size, image_size, 3))
    x = augmentation(seed)(inputs)
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0)(x)
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet" if pretrained else None,
        input_shape=(image_size, image_size, 3), pooling="avg"
    )
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="mobilenet_v2")


def build_model(name, image_size=None, num_classes=43, pretrained=True, seed=42):
    if name == "custom_cnn":
        return build_custom_cnn(image_size or 64, num_classes, seed)
    if name == "mobilenet_v2":
        return build_mobilenet_v2(image_size or 96, num_classes, pretrained, seed)
    raise ValueError(f"Unknown model: {name}")
