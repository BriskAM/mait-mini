# GTSRB Traffic-Sign Recognition — Project Synopsis

This project builds a reproducible TensorFlow/Keras pipeline for classifying the 43 German Traffic Sign Recognition Benchmark classes. It includes complete dataset download and checksum verification, EDA, leakage-aware validation, augmentation, a custom CNN, MobileNetV2 transfer learning, evaluation, CLI inference, a Colab notebook, a dashboard website, and formal DOCX documentation.

## Recorded full-data run

| Model | Test accuracy | Macro F1 |
|---|---:|---:|
| Custom CNN, 5 epochs, M1 Metal GPU | 56.90% | 56.19% |
| MobileNetV2, 5 epochs, M1 Metal GPU | 62.86% | 55.74% |

These are reproducible five-epoch baseline runs on Apple M1 Metal GPU; longer tuned training is recommended for a final benchmark.

## Scope

Version one is classification and single-image inference only. Object detection, real-time video, and safety-critical deployment are out of scope.
