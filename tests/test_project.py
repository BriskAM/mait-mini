from pathlib import Path
import numpy as np
import tensorflow as tf

from traffic_sign_recognition.models import build_custom_cnn, build_mobilenet_v2
from traffic_sign_recognition.train import set_seed
from traffic_sign_recognition.inference import predict_array


def test_model_output_shapes():
    assert build_custom_cnn(32).output_shape == (None, 43)
    assert build_mobilenet_v2(32, pretrained=False).output_shape == (None, 43)


def test_seed_is_reproducible():
    set_seed(7); a = tf.random.uniform((4,)).numpy()
    set_seed(7); b = tf.random.uniform((4,)).numpy()
    np.testing.assert_allclose(a, b)


def test_prediction_format():
    model = build_custom_cnn(32)
    result = predict_array(model, np.zeros((32, 32, 3), dtype="float32"), top_k=3)
    assert 0 <= result["class_id"] < 43
    assert len(result["top_k"]) == 3
    assert all(0 <= row["confidence"] <= 1 for row in result["top_k"])
