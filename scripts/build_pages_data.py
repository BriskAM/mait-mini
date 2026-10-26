#!/usr/bin/env python3
"""Build the data files for the GitHub Pages site (site/data, site/img) from a notebook results bundle.

Usage (needs numpy, pandas, Pillow, h5py):
    python scripts/build_pages_data.py --results path/to/unzipped/gtsrb_demo_results [--data data/raw/gtsrb] [--out site]

Inputs
  --results  the folder produced by unzipping gtsrb_demo_results.zip from the demo notebook
  --data     the extracted GTSRB dataset (only Train.csv / Test.csv are read, for the EDA charts)
Outputs (all under --out)
  data/cnn_weights.bin, data/cnn_manifest.json   the custom CNN, as raw float32 arrays read from the .keras file
  data/results.json                              every number the dashboard charts
  data/samples.json, data/samples.png            sample test images (64x64 sprite) with the Keras predictions recorded in the notebook
  img/*.png                                      selected notebook figures
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

SEED = 42
FIGURES = ["eda_one_per_class", "eda_track", "eda_brightness", "eda_resolution", "augmentation_preview",
           "gradcam_grid", "tsne_embeddings", "confident_errors", "robustness_examples"]

# Facts printed by the notebook run (cells 7, 9, 12, 14, 16, 19, 22, 25, 29); they are not in the results bundle.
FACTS = {
    "train_images": 39209, "test_images": 12630, "classes": 43, "tracks": 1307, "frames_per_track": 30,
    "largest_class": {"name": "Speed limit (50km/h)", "images": 2250}, "smallest_class": {"name": "Speed limit (20km/h)", "images": 210}, "imbalance": 10.7,
    "width_min": 25, "width_max": 243, "width_median": 43, "share_under_40px": 0.40, "brightness_mean": 82, "brightness_std": 48,
    "random_split_tracks_leaked": 1291, "random_split_tracks_total": 1307,
    "fit_images": 33570, "fit_tracks": 1119, "val_images": 5639, "val_tracks": 188,
    "cnn_params": 116203, "mobilenet_params": 2313067, "mobilenet_trainable_stage1": 55083, "mobilenet_trainable_stage2": 1834219,
    "tflite_mb": 0.24, "tflite_agreement": 1.0, "tflite_ms": 1.24, "keras_mb": 1.5,
}


def read_weights(keras_file: Path):
    """Return (float32 blob, manifest) for the custom CNN inside a Keras 3 .keras archive."""
    with zipfile.ZipFile(keras_file) as z:
        z.extract("model.weights.h5", keras_file.parent / "_tmp_weights")
        config = json.loads(z.read("config.json"))
    h5 = h5py.File(keras_file.parent / "_tmp_weights" / "model.weights.h5", "r")
    bn_eps = next(l["config"]["epsilon"] for l in config["config"]["layers"] if l["class_name"] == "BatchNormalization")
    order = [("conv1", "conv2d", 2), ("bn1", "batch_normalization", 4), ("conv2", "conv2d_1", 2), ("bn2", "batch_normalization_1", 4),
             ("conv3", "conv2d_2", 2), ("bn3", "batch_normalization_2", 4), ("dense1", "dense", 2), ("dense2", "dense_1", 2)]
    names = {"conv": ["w", "b"], "bn": ["gamma", "beta", "mean", "var"], "dense": ["w", "b"]}
    chunks, entries, offset = [], [], 0
    for tag, group, n in order:
        for i in range(n):
            arr = np.asarray(h5[f"layers/{group}/vars/{i}"], dtype="<f4")
            key = tag.rstrip("0123456789")
            entries.append({"name": f"{tag}.{names[key][i]}", "shape": list(arr.shape), "offset": offset})
            chunks.append(arr.ravel()); offset += arr.size
    shutil.rmtree(keras_file.parent / "_tmp_weights")
    blob = np.concatenate(chunks).astype("<f4")
    return blob, {"dtype": "float32", "count": int(blob.size), "bn_epsilon": bn_eps, "input": [64, 64, 3], "entries": entries}


def sample_set(res: Path, out: Path, names):
    """Pick gallery images from sample_test_images.npz and attach the Keras results recorded in test_predictions.csv."""
    npz = np.load(res / "sample_test_images.npz"); X, y = npz["X"], npz["y"]
    preds = pd.read_csv(res / "results" / "custom_cnn" / "test_predictions.csv")
    sel = np.random.default_rng(SEED).choice(len(preds), len(X), replace=False)       # same call as the notebook's export cell
    rows = preds.iloc[sel].reset_index(drop=True)
    assert (rows.true_class.values == y).all(), "sample mapping does not match the recorded predictions"
    rng = np.random.default_rng(SEED)
    chosen = []
    wrong = [i for i in range(len(X)) if not rows.correct[i]]
    for c in range(43):
        ok = [i for i in range(len(X)) if y[i] == c and rows.correct[i]]
        chosen += list(rng.choice(ok, min(2, len(ok)), replace=False)) if ok else []
    chosen += wrong
    cols = 16; tiles = len(chosen); sheet = Image.new("RGB", (cols * 64, ((tiles + cols - 1) // cols) * 64))
    meta = []
    for k, i in enumerate(chosen):
        sheet.paste(Image.fromarray(X[i]), ((k % cols) * 64, (k // cols) * 64))
        meta.append({"tile": k, "true": int(y[i]), "pred": int(rows.pred_class[i]), "conf": float(rows.confidence[i]), "hard": bool(not rows.correct[i])})
    sheet.save(out / "data" / "samples.png", optimize=True)
    (out / "data" / "samples.json").write_text(json.dumps({"cols": cols, "tile": 64, "items": meta}, separators=(",", ":")))
    return len(meta), len(wrong)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/raw/gtsrb"))
    ap.add_argument("--out", type=Path, default=Path("site"))
    a = ap.parse_args(); res, out = a.results, a.out
    (out / "data").mkdir(parents=True, exist_ok=True); (out / "img").mkdir(parents=True, exist_ok=True)
    names = json.loads((res / "class_names.json").read_text())

    blob, manifest = read_weights(res / "models" / "custom_cnn.keras")
    (out / "data" / "cnn_weights.bin").write_bytes(blob.tobytes())
    (out / "data" / "cnn_manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))

    # --- models -------------------------------------------------------------------------------------------------------
    models = {}
    for key, label in [("custom_cnn", "Custom CNN"), ("mobilenetv2_head_only", "MobileNetV2 (frozen)"), ("mobilenetv2_fine-tuned", "MobileNetV2 (fine-tuned)"), ("ensemble", "Ensemble")]:
        m = json.loads((res / "results" / key / "metrics.json").read_text())
        models[key] = {"label": label, "val_accuracy": m["validation_accuracy"], "test_accuracy": m["test_accuracy"], "test_macro_f1": m["test_macro_f1"],
                       "test_top3": m["test_top3"], "ece": m["ece"], "parameters": m["parameters"], "latency_ms": m["latency_ms_bs1"], "training_seconds": m["training_seconds"]}
    hist = lambda k: json.loads((res / "results" / k / "history.json").read_text())
    h_cnn, h_head, h_ft = hist("custom_cnn"), hist("mobilenetv2_head_only"), hist("mobilenetv2_fine-tuned")
    history = {"custom_cnn": {k: h_cnn[k] for k in ("accuracy", "val_accuracy", "loss", "val_loss")},
               "mobilenet": {k: h_head[k] + h_ft[k] for k in ("accuracy", "val_accuracy", "loss", "val_loss")}, "mobilenet_boundary": len(h_head["loss"])}

    ladder = [{"step": int(r["step"]), "model": r["model"], "track": r["track"], "val_accuracy": float(r["val_accuracy"]), "val_macro_f1": float(r["val_macro_f1"]), "parameters": int(r["parameters"])}
              for r in csv.DictReader(open(res / "progress_ladder.csv"))]

    # --- per class, confusion, errors --------------------------------------------------------------------------------
    cm = np.load(res / "confusion_matrix.npy")
    per = pd.read_csv(res / "per_class_metrics_selected_model.csv")
    per_class = [{"id": int(r["class"]), "name": r["name"], "precision": float(r["precision"]), "recall": float(r["recall"]), "f1": float(r["f1"]),
                  "support": int(r["support"]), "train_images": int(r["train_images"])} for _, r in per.iterrows()]
    off = cm.copy(); np.fill_diagonal(off, 0)
    top = [{"true": int(i), "pred": int(j), "count": int(off[i, j])} for i, j in zip(*np.unravel_index(np.argsort(off.ravel())[::-1][:12], off.shape))]

    # --- calibration and abstention (custom CNN, test set) -----------------------------------------------------------
    p = pd.read_csv(res / "results" / "custom_cnn" / "test_predictions.csv"); conf, ok = p.confidence.values, p.correct.values.astype(bool)
    edges = np.linspace(0, 1, 11); rel = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        rel.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "confidence": float(conf[m].mean()) if m.any() else None, "accuracy": float(ok[m].mean()) if m.any() else None})
    ths = np.linspace(0.30, 0.999, 120)
    coverage = [{"t": float(t), "coverage": float((conf >= t).mean()), "accuracy": float(ok[conf >= t].mean()) if (conf >= t).any() else None} for t in ths]
    hist_conf = {"edges": np.linspace(0, 1, 21).tolist(), "correct": np.histogram(conf[ok], bins=np.linspace(0, 1, 21))[0].tolist(), "wrong": np.histogram(conf[~ok], bins=np.linspace(0, 1, 21))[0].tolist()}

    robust = {}
    for r in csv.DictReader(open(res / "robustness_selected_model.csv")):
        key = list(r.keys())[0]
        robust[r[key]] = {"mild": float(r["mild"]), "medium": float(r["medium"]), "severe": float(r["severe"]), "clean": float(r["clean"])}

    # --- EDA (read from the dataset CSVs) ----------------------------------------------------------------------------
    tr, te = pd.read_csv(a.data / "Train.csv"), pd.read_csv(a.data / "Test.csv")
    eda = {"train_counts": tr.ClassId.value_counts().sort_index().tolist(), "test_counts": te.ClassId.value_counts().sort_index().tolist(),
           "width_hist": {"edges": list(range(20, 260, 10)), "counts": np.histogram(tr.Width, bins=list(range(20, 260, 10)))[0].tolist()}}

    n_samples, n_hard = sample_set(res, out, names)

    for f in FIGURES:                                       # downscale so the page stays light (the Grad-CAM grid alone is 2 MB)
        im = Image.open(res / "figures" / f"{f}.png").convert("RGB"); im.thumbnail((1500, 1500))
        im.save(out / "img" / f"{f}.webp", quality=88, method=6)

    results = {"class_names": names, "facts": FACTS, "models": models, "history": history, "ladder": ladder, "per_class": per_class,
               "confusion": cm.tolist(), "top_confusions": top, "reliability": rel, "coverage": coverage, "confidence_hist": hist_conf,
               "robustness": robust, "eda": eda, "selected": "custom_cnn", "mode": json.loads((res / "metrics.json").read_text())["mode"],
               "versions": {"tensorflow": json.loads((res / "metrics.json").read_text())["tensorflow"], "keras": json.loads((res / "metrics.json").read_text())["keras"]}}
    (out / "data" / "results.json").write_text(json.dumps(results, separators=(",", ":")))
    print(f"weights: {blob.size:,} floats | samples: {n_samples} ({n_hard} hard) | results.json {(out / 'data' / 'results.json').stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
