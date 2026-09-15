# GTSRB Traffic-Sign Recognition

TensorFlow/Keras project for classifying the 43 German Traffic Sign Recognition Benchmark (GTSRB) classes. It includes a compact CNN baseline, a MobileNetV2 transfer-learning model, reproducible data preparation, evaluation reports, and a Google Colab notebook.

## Dataset

The complete GTSRB archive is downloaded from [Zenodo](https://zenodo.org/records/13741936) and verified against MD5 `7faafd0d78d831c7eb4d555641a83a62`. It is extracted under `data/raw/gtsrb/` and intentionally excluded from Git. GTSRB is a single-image classification dataset; this repository does not implement bounding-box detection.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
python scripts/download_dataset.py
```

## Training and evaluation

```bash
python -m traffic_sign_recognition.train --model custom_cnn
python -m traffic_sign_recognition.train --model mobilenet_v2
python -m traffic_sign_recognition.evaluate --model-path artifacts/models/mobilenet_v2/model.keras
python -m traffic_sign_recognition.predict --model-path artifacts/models/mobilenet_v2/model.keras --image path/to/sign.png
```

Useful options include `--image-size`, `--epochs`, `--batch-size`, `--learning-rate`, `--seed`, `--output-dir`, and `--no-pretrained`. A small CPU smoke run is `--epochs 1 --max-train-samples 256 --max-test-samples 128 --no-pretrained`.

Training outputs include the best `.keras` model, `history.json`, accuracy/loss plots, a confusion matrix, and per-class precision/recall/F1 metrics.

## Google Colab

Open `notebooks/traffic_sign_recognition_colab.ipynb` in Colab. The notebook installs the pinned dependencies, downloads and verifies the archive automatically, previews class balance and sample images, trains both models, evaluates them, and runs single-image inference. Enable a GPU runtime for full training.

## Citation and attribution

Please cite the original GTSRB paper: J. Stallkamp et al., “The German Traffic Sign Recognition Benchmark: A multi-class classification competition,” IJCNN 2011. The project uses the public dataset distributed by Ruhr-Universität Bochum and mirrored on Zenodo.
