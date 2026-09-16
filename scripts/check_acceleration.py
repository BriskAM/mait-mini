#!/usr/bin/env python3
"""Report whether TensorFlow can see an Apple Silicon GPU."""

import json
import platform
import tensorflow as tf


def main():
    payload = {"machine": platform.machine(), "tensorflow": tf.__version__, "physical_devices": [d.name for d in tf.config.list_physical_devices()], "gpu_devices": [d.name for d in tf.config.list_physical_devices("GPU")]}
    payload["gpu_available"] = bool(payload["gpu_devices"])
    print(json.dumps(payload, indent=2))
    if not payload["gpu_available"]:
        print("No supported GPU was found. On Apple Silicon, install requirements-macos.txt and check macOS GPU access.")


if __name__ == "__main__": main()
