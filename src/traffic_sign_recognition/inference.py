"""Reusable image prediction helpers for CLI, Grad-CAM, and the demo server."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


def prepare_image(image_path: str | Path, image_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    return np.asarray(image, dtype="float32") / 255.0


def predict_array(model: tf.keras.Model, image: np.ndarray, top_k: int = 5) -> dict:
    probabilities = model.predict(image[None], verbose=0)[0]
    order = np.argsort(probabilities)[::-1][:top_k]
    return {
        "class_id": int(order[0]),
        "confidence": float(probabilities[order[0]]),
        "top_k": [{"class_id": int(i), "confidence": float(probabilities[i])} for i in order],
    }


def predict_image(model: tf.keras.Model, image_path: str | Path, top_k: int = 5) -> dict:
    image = prepare_image(image_path, int(model.input_shape[1]))
    result = predict_array(model, image, top_k=top_k)
    result["image_size"] = int(model.input_shape[1])
    return result
