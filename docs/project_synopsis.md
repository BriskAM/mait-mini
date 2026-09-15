# GTSRB Traffic-Sign Recognition — Project Synopsis

This project builds a reproducible TensorFlow/Keras pipeline for classifying the 43 German Traffic Sign Recognition Benchmark classes. It includes complete dataset download and checksum verification, EDA, leakage-aware validation, augmentation, a custom CNN, MobileNetV2 transfer learning, evaluation, CLI inference, a Colab notebook, a dashboard website, and formal DOCX documentation.

## Recorded full-data run

| Model | Test accuracy | Macro F1 |
|---|---:|---:|
| Custom CNN, 3 epochs | 51.75% | 36.44% |
| MobileNetV2, 1 epoch | 53.09% | 42.24% |

The runs are CPU-feasible demonstrations; matched longer GPU training is recommended for a final benchmark.

## Scope

Version one is classification and single-image inference only. Object detection, real-time video, and safety-critical deployment are out of scope.
