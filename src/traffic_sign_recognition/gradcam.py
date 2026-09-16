"""Grad-CAM explanations for saved Keras classifiers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from .inference import predict_image, prepare_image


def _last_conv_layer(model: tf.keras.Model):
    def walk(layer):
        for child in getattr(layer, "layers", []):
            yield from walk(child)
        yield layer

    conv_types = {"Conv2D", "DepthwiseConv2D", "SeparableConv2D"}
    layers = [layer for layer in walk(model) if layer.__class__.__name__ in conv_types]
    if not layers:
        raise ValueError("The model does not contain a Conv2D layer for Grad-CAM")
    return layers[-1]


def explain(model: tf.keras.Model, image_path: str | Path, output_path: str | Path) -> dict:
    image = prepare_image(image_path, int(model.input_shape[1]))
    batch = tf.convert_to_tensor(image[None], dtype=tf.float32)
    nested_base = None
    for candidate in reversed(model.layers):
        if isinstance(candidate, tf.keras.Model):
            try:
                _last_conv_layer(candidate)
                nested_base = candidate
                break
            except ValueError:
                continue
    if nested_base is not None:
        conv_layer = _last_conv_layer(nested_base)
        base_index = model.layers.index(nested_base)
        with tf.GradientTape() as tape:
            features = batch
            for layer in model.layers[1:base_index]:
                features = layer(features, training=False)
            conv_outputs, features = tf.keras.Model(nested_base.inputs, [conv_layer.output, nested_base.output])(features)
            for layer in model.layers[base_index + 1:]:
                features = layer(features, training=False)
            predictions = features
            class_id = tf.argmax(predictions[0])
            score = predictions[:, class_id]
    else:
        conv_layer = _last_conv_layer(model)
        grad_model = tf.keras.Model(model.inputs, [conv_layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(batch)
            class_id = tf.argmax(predictions[0])
            score = predictions[:, class_id]
    gradients = tape.gradient(score, conv_outputs)
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    cam = tf.reduce_sum(conv_outputs * weights[:, None, None, :], axis=-1)[0]
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    heatmap = np.uint8(cam.numpy() * 255)
    heatmap = cv2.applyColorMap(cv2.resize(heatmap, (image.shape[1], image.shape[0])), cv2.COLORMAP_JET)
    original = cv2.cvtColor(np.uint8(image * 255), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original, 0.55, heatmap, 0.45, 0)
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
    result = predict_image(model, image_path)
    result.update({"cam_layer": conv_layer.name, "output_path": str(output_path)})
    return result
