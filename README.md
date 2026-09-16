# GTSRB Traffic-Sign Recognition

TensorFlow/Keras project for classifying the 43 German Traffic Sign Recognition Benchmark (GTSRB) classes. It includes a compact CNN baseline, a MobileNetV2 transfer-learning model, reproducible data preparation, evaluation reports, and a Google Colab notebook.

## Dataset

The complete GTSRB archive is downloaded from [Zenodo](https://zenodo.org/records/13741936) and verified against MD5 `7faafd0d78d831c7eb4d555641a83a62`. It is extracted under `data/raw/gtsrb/` and intentionally excluded from Git.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
python scripts/download_dataset.py
```

For the local Apple Silicon environment, install the Metal plugin after the base requirements:

```bash
uv pip install --python .venv/bin/python -r requirements-macos.txt
PYTHONPATH=src python scripts/check_acceleration.py
```

Training automatically uses `/GPU:0` when TensorFlow exposes it. Add `--require-gpu` when a run must fail rather than fall back to CPU. On the M1 host, the verified accelerated environment reports `GPU:0` and `Metal device set to: Apple M1`.

## Training and evaluation

```bash
python -m traffic_sign_recognition.train --model custom_cnn
python -m traffic_sign_recognition.train --model mobilenet_v2
python -m traffic_sign_recognition.evaluate --model-path artifacts/models/mobilenet_v2/model.keras
python -m traffic_sign_recognition.predict --model-path artifacts/models/mobilenet_v2/model.keras --image path/to/sign.png
```

Useful options include `--image-size`, `--epochs`, `--batch-size`, `--learning-rate`, `--seed`, `--output-dir`, and `--no-pretrained`. A small CPU smoke run is `--epochs 1 --max-train-samples 256 --max-test-samples 128 --no-pretrained`.

Training outputs include the best `.keras` model, `history.json`, accuracy/loss plots, a confusion matrix, and per-class precision/recall/F1 metrics.

## Explanations, error analysis, and experiments

Generate an explanation heatmap for a prediction:

```bash
PYTHONPATH=src python scripts/generate_gradcam.py --model-path artifacts/models/mobilenet_v2/model.keras --image path/to/sign.png
```

Summarize the most frequent class confusions after evaluation:

```bash
python scripts/error_analysis.py --evaluation-dir artifacts/models/mobilenet_v2/evaluation
```

Run a small learning-rate experiment grid (defaults are intentionally lightweight):

```bash
PYTHONPATH=src python scripts/run_experiments.py
```

Launch the local upload and webcam demo with Grad-CAM explanations:

```bash
PYTHONPATH=src python scripts/demo_server.py --model-path artifacts/models/mobilenet_v2/model.keras
```

Open `http://localhost:8000`. The hosted dashboard includes the same interface, while actual predictions require the local Keras server because model files are intentionally excluded from Git.

## Optional detection extension

The GTSRB models classify cropped signs. For full-scene localization, install the optional YOLO dependency and provide a YOLO-format traffic-sign detector checkpoint:

```bash
uv pip install -r requirements-detection.txt
python scripts/detect.py --weights path/to/sign_detector.pt --source path/to/image-or-video
python scripts/webcam_detect.py --weights path/to/sign_detector.pt
```

These scripts select PyTorch MPS when available. A detector checkpoint and bounding-box annotations are not part of the GTSRB classification archive, so they must be supplied separately.

## Full pipeline artifacts

Generate the EDA figures and website data after downloading the dataset:

```bash
python scripts/run_eda.py
python scripts/build_results.py
python -m http.server 8000 --directory site/dist
```

Then open `http://localhost:8000` to view the project dashboard. It presents the class distribution, representative images, image statistics, model comparison, learning curves, confusion matrices, and the reproducible workflow. Generated model reports and plots are kept under ignored `artifacts/`; the dashboard-ready figures and JSON are stored under `site/dist/`.

The project synopsis and detailed report are available as both Markdown source and formatted Word documents:

- `docs/project_synopsis.md` and `docs/traffic_sign_recognition_synopsis.docx`
- `docs/project_report.md` and `docs/traffic_sign_recognition_project_report.docx`
- PDF exports: `docs/traffic_sign_recognition_synopsis.pdf` and `docs/traffic_sign_recognition_project_report.pdf`
- Presenter guide: `docs/traffic_sign_recognition_presenter_guide.docx` and `docs/traffic_sign_recognition_presenter_guide.pdf`
- Verified literature survey and source-to-code mapping: `docs/literature_survey.md`

The DOCX and PDF deliverables use an IEEE-style two-column paper layout with Times New Roman typography, numbered sections, abstracts, index terms, compact tables, figures, results, limitations, and references.

The report records the completed full-data runs and their hardware in `site/dist/data/results.json`. Use matched GPU training for a final performance benchmark.

## Google Colab

Open `notebooks/traffic_sign_recognition_colab.ipynb` in Colab. The notebook installs the pinned dependencies, downloads and verifies the archive automatically, previews class balance and sample images, trains both models, evaluates them, and runs single-image inference. Enable a GPU runtime for full training.

## Project structure

```text
scripts/                         dataset, EDA, result, and document builders
src/traffic_sign_recognition/    reusable data, model, training, evaluation, prediction code
notebooks/                       end-to-end Google Colab notebook
site/dist/                       static project website
docs/                            synopsis and project report
tests/                           preprocessing, labels, models, and downloader tests
```

## Citation and attribution

Please cite the original GTSRB paper: J. Stallkamp et al., “The German Traffic Sign Recognition Benchmark: A multi-class classification competition,” IJCNN 2011. The project uses the public dataset distributed by Ruhr-Universität Bochum and mirrored on Zenodo.
