#!/usr/bin/env python3
"""Build the panel-demo Colab notebook (notebooks/traffic_sign_recognition_demo.ipynb).

The notebook is generated so its narrative and code stay reviewable as plain text.
Run:  python scripts/build_demo_notebook.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "traffic_sign_recognition_demo.ipynb"
SIGNNAMES = [r["SignName"] for r in csv.DictReader(open(ROOT / "data/raw/gtsrb/signnames.csv"))]
assert len(SIGNNAMES) == 43

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").replace("__SIGNNAMES__", repr(SIGNNAMES))})


# ───────────────────────────── 0. Title ─────────────────────────────
md(r"""
# Traffic-Sign Recognition on GTSRB
### A reproducible deep-learning pipeline — from raw data to a deployable, explainable classifier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BriskAM/mait-mini/blob/main/notebooks/traffic_sign_recognition_demo.ipynb)

**What this notebook does:** downloads and verifies the German Traffic Sign Recognition Benchmark (GTSRB), profiles it, trains and compares a custom CNN against MobileNetV2 transfer learning, evaluates on the *official* test split, explains the predictions with Grad-CAM, stress-tests robustness and calibration, and exports a validated model.

| # | Section | What it shows |
|---|---|---|
| 1 | Problem & workflow | Goal, scope, pipeline diagram |
| 2 | Setup | Reproducibility, hardware check |
| 3 | Data | Verified download, label & image-size audit |
| 4 | Exploratory analysis | Class imbalance, image quality, track structure |
| 5 | Validation design | **Why a grouped split** (leakage demo) |
| 6 | Preprocessing & augmentation | Cached tensors, augmentation preview |
| 7 | Models | Architectures and design rationale |
| 8 | Training | Live curves, two-stage transfer learning |
| 9 | Evaluation | Official test metrics, per-class F1, confusion analysis |
| 9A | Progress ladder | Baseline to final model, one change at a time |
| 10 | Explainability | Grad-CAM, embedding map |
| 11 | Reliability | Calibration, selective prediction, robustness |
| 12 | Inference demo | Top-5 predictions + heatmap for any image |
| 13 | Export & checks | Saved model, reload parity, TFLite |
| 14 | Wrap-up | Decision log, limitations, future work, Q&A |

""")

# ───────────────────────────── 1. Problem ─────────────────────────────
md(r"""
---
## 1 · Problem statement and workflow

**Task.** Given a cropped photograph of a traffic sign, predict which of the **43 GTSRB classes** it is (speed limits, warnings, mandatory and prohibitory signs, …).

**Why it matters.** Driver-assistance systems must read signs reliably under glare, shadow, blur and partial occlusion. Mistakes are asymmetric: confusing *“Speed limit 30”* with *“Speed limit 80”* is far worse than a harmless near-miss between two warning triangles. So we evaluate more than raw accuracy.

**Scope.** Classification of *cropped* signs only. Detection in full scenes, video, and any safety-critical deployment are out of scope (see §14).

**Evaluation protocol (fixed before looking at results):**
1. Train on the official training split, minus a *grouped* validation hold-out (§5).
2. Choose hyper-parameters and the final model using **validation** data only.
3. Report **official test** results once, at the end — accuracy *and* macro-F1 (because classes are imbalanced).
""")

code(r"""
# Pipeline overview (rendered here so the workflow is visible at a glance)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

steps = ["Download +\nMD5 verify", "EDA &\ndata audit", "Grouped\ntrain/val split", "Cache + augment\n(64×64)",
         "Train\nCNN / MobileNetV2", "Evaluate on\nofficial test", "Explain, calibrate,\nstress-test", "Export +\nparity checks"]
fig, ax = plt.subplots(figsize=(15, 2.6)); ax.axis("off"); ax.set_xlim(0, len(steps) * 1.9); ax.set_ylim(0, 1)
for i, s in enumerate(steps):
    x = i * 1.9 + 0.1
    ax.add_patch(FancyBboxPatch((x, 0.2), 1.55, 0.6, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#e8f4fd", ec="#1f77b4", lw=1.4))
    ax.text(x + 0.775, 0.5, s, ha="center", va="center", fontsize=9.5)
    if i < len(steps) - 1: ax.annotate("", xy=(x + 1.88, 0.5), xytext=(x + 1.57, 0.5), arrowprops=dict(arrowstyle="->", color="#555", lw=1.4))
plt.title("End-to-end workflow", fontsize=12, pad=2); plt.show()
""")

# ───────────────────────────── 2. Setup ─────────────────────────────
md(r"""
---
## 2 · Setup and reproducibility

**Decisions made here**
- A single `MODE` switch controls runtime. Everything downstream reads from one `CFG` dict, so *demo* and *full* run identical code with different budgets.
- All randomness is seeded (`SEED=42`) for Python, NumPy and TensorFlow so results are repeatable on the same hardware.
- The notebook is **self-contained**: no Google Drive, no GitHub clone, no local files. It also runs unchanged outside Colab (set `GTSRB_DIR` to reuse an existing copy of the dataset).
- The GPU check *warns* instead of crashing; without a GPU the notebook drops to `smoke` mode so it still completes.
""")

code(r"""
MODE = "full"  #@param ["smoke", "demo", "full"]
import os
MODE = os.environ.get("GTSRB_MODE", MODE)

import io, json, time, math, random, hashlib, zipfile, shutil, urllib.request, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
try:
    import cv2
except ImportError:
    %pip -q install opencv-python-headless
    import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED); keras.utils.set_random_seed(SEED)

gpus = tf.config.list_physical_devices("GPU")
for g in gpus: tf.config.experimental.set_memory_growth(g, True)
if not gpus and MODE != "smoke":
    print("WARNING: No GPU detected — switching to 'smoke' mode so the notebook still completes.\n   (Runtime ▸ Change runtime type ▸ T4 GPU, then rerun for the real thing.)")
    MODE = "smoke"

CFGS = {   # epochs are upper bounds; early stopping usually ends sooner
    "smoke": dict(max_train=2400, max_test=1200, cnn_epochs=2, ladder_epochs=1, head_epochs=1, ft_epochs=1, eda_sample=300, robust_n=300, tsne_n=600),
    "demo":  dict(max_train=None, max_test=None, cnn_epochs=12, ladder_epochs=10, head_epochs=6, ft_epochs=8, eda_sample=1500, robust_n=3000, tsne_n=3000),
    "full":  dict(max_train=None, max_test=None, cnn_epochs=30, ladder_epochs=15, head_epochs=12, ft_epochs=15, eda_sample=3000, robust_n=6000, tsne_n=4000),
}
CFG = CFGS[MODE]
IMG, MB_SIZE, BATCH, NUM_CLASSES = 64, 128, 128, 43   # cache resolution, MobileNet input resolution, batch size
CLASS_NAMES = __SIGNNAMES__

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
BLUE, ORANGE, GREEN, RED, PURPLE = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7b5ea7"

WORK = Path(os.environ.get("GTSRB_WORK", "/content/gtsrb_demo" if Path("/content").exists() else "./gtsrb_demo"))
ART = WORK / "artifacts"; FIG = ART / "figures"; FIG.mkdir(parents=True, exist_ok=True)
def savefig(name): plt.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
def short(c, n=26):
    s = CLASS_NAMES[c].replace("End of speed limit (80km/h)", "End of limit 80").replace("Speed limit (", "Limit ").replace("km/h)", "")
    return s if len(s) <= n else s[:n - 1] + "…"

print(f"MODE={MODE} | TensorFlow {tf.__version__} | Keras {keras.__version__} | GPU: {[g.name for g in gpus] or 'none'}")
print("Config:", CFG)
""")

# ───────────────────────────── 3. Data ─────────────────────────────
md(r"""
---
## 3 · Data acquisition and integrity

**Dataset.** GTSRB (Stallkamp et al., IJCNN 2011): real photographs taken from moving vehicles in Germany. The complete archive is mirrored on Zenodo.

**Decisions**
- **Verify the MD5 checksum** of the 320 MB archive before extracting. A truncated or tampered download would otherwise silently poison every result.
- **Stream** the hash instead of reading the whole file into memory.
- **Use the official train/test split** exactly as published, so results are comparable with the literature. The test set is never used for any decision.
- Assert the expected counts (39,209 train / 12,630 test / 43 classes) — fail fast if the data is not what we think it is.
""")

code(r"""
URL = "https://zenodo.org/api/records/13741936/files/data.zip/content"
EXPECTED_MD5 = "7faafd0d78d831c7eb4d555641a83a62"

def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""): h.update(block)
    return h.hexdigest()

def find_root(base):
    for cand in [base, *base.rglob("Train")]:
        cand = cand.parent if cand.name == "Train" else cand
        if (cand / "Train").is_dir() and (cand / "Test.csv").exists(): return cand
    raise FileNotFoundError(f"GTSRB Train/ and Test.csv not found under {base}")

if os.environ.get("GTSRB_DIR"):
    DATA = find_root(Path(os.environ["GTSRB_DIR"])); print("Using existing dataset:", DATA)
else:
    WORK.mkdir(parents=True, exist_ok=True)
    archive = WORK / "data.zip"
    if not archive.exists() or md5_of(archive) != EXPECTED_MD5:
        print("Downloading GTSRB (~320 MB)…")
        last = [0]
        def hook(blocks, bs, total):
            done = blocks * bs
            if total > 0 and done - last[0] > 25 << 20: last[0] = done; print(f"  {done / 1e6:5.0f} / {total / 1e6:.0f} MB")
        urllib.request.urlretrieve(URL, archive, hook)
    got = md5_of(archive); print("MD5:", got)
    assert got == EXPECTED_MD5, "Checksum mismatch — download is corrupt; delete data.zip and rerun."
    if not (WORK / "gtsrb" / "Train").exists():
        with zipfile.ZipFile(archive) as z: z.extractall(WORK / "gtsrb")
    DATA = find_root(WORK / "gtsrb")
print("Dataset verified at", DATA)
""")

code(r"""
# Build tidy label tables. Train.csv / Test.csv also provide each image's original width & height.
def tidy_meta(csv_path, key_prefix):
    meta = pd.read_csv(csv_path)
    meta["rel"] = meta["Path"].str.split(key_prefix + "/", n=1).str[1]
    return meta.set_index("rel")[["Width", "Height", "ClassId"]]

train_meta = tidy_meta(DATA / "Train.csv", "Train")
rows = []
for folder in sorted((DATA / "Train").iterdir()):
    if folder.is_dir() and folder.name.isdigit():
        for p in sorted(folder.glob("*.png")):
            _, track, frame = p.stem.split("_")              # file name = class_track_frame
            rows.append((str(p), int(folder.name), f"{folder.name}_{track}", int(frame), f"{folder.name}/{p.name}"))
train_df = pd.DataFrame(rows, columns=["path", "label", "group", "frame", "rel"])
train_df = train_df.join(train_meta[["Width", "Height"]], on="rel").drop(columns="rel")

test_csv = pd.read_csv(DATA / "Test.csv")
test_df = pd.DataFrame({"path": [str(DATA / "Test" / Path(p).name) for p in test_csv["Path"]], "label": test_csv["ClassId"].astype(int),
                        "Width": test_csv["Width"], "Height": test_csv["Height"]})

assert len(train_df) == 39209 and len(test_df) == 12630 and train_df.label.nunique() == NUM_CLASSES, "Unexpected dataset size"
assert train_df.path.map(os.path.exists).all() and test_df.path.map(os.path.exists).all(), "Missing image files"
print(f"train images: {len(train_df):,}   official test images: {len(test_df):,}   classes: {train_df.label.nunique()}")
print(f"distinct training *tracks* (physical sign instances): {train_df.group.nunique():,}")
train_df.head()
""")

# ───────────────────────────── 4. EDA ─────────────────────────────
md(r"""
---
## 4 · Exploratory data analysis

EDA is not decoration — each finding below changes a modelling decision.

| Question | Why we ask |
|---|---|
| Are classes balanced? | Decides the metric (accuracy vs macro-F1) and whether to re-weight |
| How big are the images? | Decides the input resolution |
| Are lighting conditions diverse? | Decides the augmentation strength |
| How are images related? | Decides the validation strategy (§5) |
""")

code(r"""
# 4.1 Class distribution — train vs official test
tr_counts = train_df.label.value_counts().sort_index(); te_counts = test_df.label.value_counts().sort_index()
fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
order = tr_counts.sort_values().index
axes[0].barh([short(c) for c in order], tr_counts[order].values, color=BLUE); axes[0].set_title("Training images per class")
axes[1].barh([short(c) for c in order], te_counts[order].values, color=ORANGE); axes[1].set_title("Official test images per class")
for a in axes: a.set_xlabel("Images"); a.tick_params(axis="y", labelsize=8)
plt.tight_layout(); savefig("eda_class_distribution"); plt.show()

ratio = tr_counts.max() / tr_counts.min()
print(f"Smallest class: {short(tr_counts.idxmin())} ({tr_counts.min()}) | largest: {short(tr_counts.idxmax())} ({tr_counts.max()}) | imbalance ≈ {ratio:.1f}×")
""")

md(r"""
**Finding — strong class imbalance (≈10×).** A model that is great on the common speed-limit signs and poor on rare ones could still post a high *accuracy*. We therefore report **macro-F1** (every class counts equally) next to accuracy, and inspect per-class results in §9. The test distribution mirrors training, so the imbalance is a property of the data, not a split artefact.
""")

code(r"""
# 4.2 One example per class (43 sign types)
fig, axes = plt.subplots(5, 9, figsize=(16, 9.5))
rng = np.random.default_rng(SEED)
for ax in axes.ravel(): ax.axis("off")
for c in range(NUM_CLASSES):
    p = train_df[train_df.label == c].path.iloc[rng.integers((train_df.label == c).sum())]
    axes.ravel()[c].imshow(Image.open(p)); axes.ravel()[c].set_title(f"{c}: {short(c, 22)}", fontsize=7.5)
plt.suptitle("One random training example per class", y=1.0); plt.tight_layout(); savefig("eda_one_per_class"); plt.show()
""")

code(r"""
# 4.3 Image resolution — taken from the CSV metadata, no need to open 39k files
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].hist(train_df.Width, bins=50, color=BLUE, alpha=.8, label="train"); axes[0].hist(test_df.Width, bins=50, color=ORANGE, alpha=.6, label="test")
axes[0].set(title="Width (px)", xlabel="pixels", ylabel="images"); axes[0].legend()
axes[1].hist2d(train_df.Width, train_df.Height, bins=40, cmap="viridis"); axes[1].set(title="Width × Height (train)", xlabel="width", ylabel="height")
axes[2].scatter(train_df.Width, train_df.Width / train_df.Height, s=2, alpha=.15, color=PURPLE); axes[2].set(title="Aspect ratio vs width", xlabel="width", ylabel="width / height")
plt.tight_layout(); savefig("eda_resolution"); plt.show()
print(f"Width range {train_df.Width.min()}–{train_df.Width.max()} px | median {int(train_df.Width.median())} px | "
      f"{(train_df.Width < 40).mean():.0%} of training images are narrower than 40 px")
print(f"Aspect ratio (w/h): median {np.median(train_df.Width / train_df.Height):.2f}, 5–95%: {np.percentile(train_df.Width / train_df.Height, [5, 95]).round(2)}")
""")

md(r"""
**Finding — signs are small and vary widely in size.** The median image is only a few dozen pixels wide, so there is little detail to preserve. That supports a **small cached resolution (64×64)**: higher resolution would just interpolate pixels that do not exist, cost memory, and slow training. Aspect ratios are close to 1, so a plain resize (no letterboxing) distorts little.
""")

code(r"""
# 4.4 Lighting diversity — brightness & contrast from a random sample
def quick_stats(p):
    a = np.asarray(Image.open(p).convert("L"), dtype=np.float32); return a.mean(), a.std()
sample = train_df.sample(CFG["eda_sample"], random_state=SEED)
with ThreadPoolExecutor(16) as ex: stats = np.array(list(ex.map(quick_stats, sample.path)))
sample = sample.assign(brightness=stats[:, 0], contrast=stats[:, 1])

fig, axes = plt.subplots(2, 5, figsize=(16, 6.2)); gs = axes[0, 0].get_gridspec()
for ax in axes[:, 0:2].ravel(): ax.remove()
ax_h = fig.add_subplot(gs[:, 0:2]); ax_h.hist(sample.brightness, bins=40, color=ORANGE, edgecolor="white")
ax_h.set(title="Mean brightness (grey-level) distribution", xlabel="mean pixel value (0–255)", ylabel="images")
dark = sample.nsmallest(3, "brightness"); bright = sample.nlargest(3, "brightness"); low_c = sample.nsmallest(3, "contrast")
for j, (title, frame) in enumerate([("Darkest", dark), ("Brightest", bright)]):
    for i, (_, r) in enumerate(frame.iterrows()):
        ax = axes[j, 2 + i]; ax.imshow(Image.open(r.path)); ax.axis("off"); ax.set_title(f"{title}\n{r.brightness:.0f}", fontsize=8)
plt.tight_layout(); savefig("eda_brightness"); plt.show()
print(f"Brightness mean {sample.brightness.mean():.0f} ± {sample.brightness.std():.0f}; darkest {sample.brightness.min():.0f}, brightest {sample.brightness.max():.0f}")
""")

md(r"""
**Finding — wide illumination range.** Some images are nearly black (dusk, shade), others washed out. The model must be tolerant of brightness/contrast change, so the augmentation pipeline (§6) includes a contrast jitter, and §11 explicitly stress-tests lighting.
""")

code(r"""
# 4.5 Data structure: each physical sign appears as a *track* of ~30 near-identical frames
g = train_df[train_df.label == 14].group.iloc[0]
track = train_df[train_df.group == g].sort_values("frame")
show = track.iloc[np.linspace(0, len(track) - 1, 10).astype(int)]
fig, axes = plt.subplots(1, 10, figsize=(16, 2.2))
for ax, (_, r) in zip(axes, show.iterrows()): ax.imshow(Image.open(r.path)); ax.axis("off"); ax.set_title(f"frame {r.frame}\n{r.Width}×{r.Height}", fontsize=7)
plt.suptitle(f"A single physical '{CLASS_NAMES[14]}' sign photographed as the car approaches (track {g}, {len(track)} frames)", y=1.12); savefig("eda_track"); plt.show()
sizes = train_df.groupby("group").size()
print(f"{len(sizes):,} tracks; frames per track: median {int(sizes.median())}, min {sizes.min()}, max {sizes.max()}")
""")

md(r"""
**Finding — images are *not* independent.** The 39,209 training images come from only ~1,300 physical signs, each photographed ~30 times as the car approaches. Consecutive frames of the same sign look almost identical. This is the most important data insight in the project: it determines how we must split for validation. →
""")

# ───────────────────────────── 5. Split ─────────────────────────────
md(r"""
---
## 5 · Validation design — preventing data leakage

**Problem.** If we split *images* at random, frames of the same physical sign land in both training and validation. The model then “recognises” a sign it effectively memorised and validation accuracy is inflated — it no longer predicts how the model behaves on a *new* sign.

**Decision.** Split by **track** (group = `class_track`). Within **each class** we hold out about one seventh of its tracks (at least one) for validation, using a seeded random generator. All frames of a sign go entirely to training *or* validation, and every one of the 43 classes is guaranteed to appear in validation. That is about 14 % of the data.

*Why per class?* Library group-splitters such as `GroupShuffleSplit` do not balance classes: in our first run it left five classes (0, 19, 21, 23, 29) with no validation images at all, which silently biased validation accuracy and macro-F1. A `StratifiedGroupKFold` attempt was a heuristic with no coverage guarantee and behaved differently across scikit-learn versions, so we do the per-class holdout explicitly. It is deterministic and does not depend on the library version.

The cell below proves the difference empirically instead of just asserting it.
""")

code(r"""
# Compare a naive random split with the grouped split
rand_tr, rand_va = train_test_split(train_df, test_size=1 / 7, random_state=SEED, stratify=train_df.label)
leaked = len(set(rand_tr.group) & set(rand_va.group))

rng = np.random.default_rng(SEED); val_groups = []
for _, tracks in train_df.groupby("label").group.unique().items():        # hold out ~1/7 of each class's tracks (>= 1)
    val_groups += list(rng.choice(tracks, max(1, round(len(tracks) / 7)), replace=False))
is_val = train_df.group.isin(val_groups).values
fit_df, val_df = train_df[~is_val].reset_index(drop=True), train_df[is_val].reset_index(drop=True)
assert not set(fit_df.group) & set(val_df.group), "Leakage: a track appears in both splits"
assert val_df.label.nunique() == NUM_CLASSES, "Validation is missing some classes"

summary = pd.DataFrame({
    "split": ["Random image split", "Grouped (used)"],
    "val images": [len(rand_va), len(val_df)],
    "tracks in BOTH train & val": [leaked, 0],
    "val tracks seen in training": [f"{leaked / rand_va.group.nunique():.0%}", "0%"],
})
display(summary.style.hide(axis="index"))
print(f"Grouped split → fit {len(fit_df):,} images / {fit_df.group.nunique()} tracks, validation {len(val_df):,} images / {val_df.group.nunique()} tracks; all 43 classes in validation: {val_df.label.nunique() == 43}")

if CFG["max_train"]:    # smoke mode only: small, class-covering subset
    fit_df = fit_df.sample(CFG["max_train"], random_state=SEED).reset_index(drop=True)
    val_df = val_df.sample(min(len(val_df), CFG["max_train"] // 4), random_state=SEED).reset_index(drop=True)
""")

md(r"""
**Result.** With a random split almost every validation track has already been seen during training; the grouped split reduces that to zero. Validation numbers from here on are therefore an honest estimate of generalisation to unseen signs. The **official test set** (different physical signs, provided by the benchmark) remains untouched until §9.
""")

# ───────────────────────────── 6. Preprocessing ─────────────────────────────
md(r"""
---
## 6 · Preprocessing and augmentation

**Preprocessing decisions**

| Decision | Choice | Reason |
|---|---|---|
| Resolution | **64×64**, cached once | Median sign is tiny (§4.3); upscaling adds no information |
| Cropping to the annotated ROI | **No** | Real detectors output boxes with margin; training on the full frame teaches tolerance to it |
| Storage | **`uint8` NumPy arrays in RAM** | Whole dataset ≈ 0.5 GB — removes per-epoch disk/decode cost and makes training GPU-bound |
| Normalisation | **Inside the model** (`Rescaling`) | Exported model accepts raw 0–255 pixels; no train/serve preprocessing mismatch |

**Augmentation decisions** (training only — Keras layers are inactive at inference)

| Transform | Setting | Reason |
|---|---|---|
| Rotation | ±14° | Camera tilt, slanted poles |
| Translation | ±5 % | Imperfect crops |
| Zoom | ±8 % | Different distances |
| Contrast | ±10 % | Lighting variation (§4.4) |
| **Horizontal flip** | **Deliberately NOT used** | Flipping changes meaning: *Turn left ahead* ↔ *Turn right ahead*, *Keep left* ↔ *Keep right* would receive the wrong label |
""")

code(r"""
def load_resized(path):
    with Image.open(path) as im: return np.asarray(im.convert("RGB").resize((IMG, IMG), Image.Resampling.BILINEAR))

def load_array(frame):
    with ThreadPoolExecutor(16) as ex: return np.stack(list(ex.map(load_resized, frame.path)))

t0 = time.time()
test_eval_df = test_df if not CFG["max_test"] else test_df.sample(CFG["max_test"], random_state=SEED).reset_index(drop=True)
X_fit, y_fit = load_array(fit_df), fit_df.label.values.astype("int32")
X_val, y_val = load_array(val_df), val_df.label.values.astype("int32")
X_test, y_test = load_array(test_eval_df), test_eval_df.label.values.astype("int32")
print(f"Cached in {time.time() - t0:.0f}s | fit {X_fit.shape} | val {X_val.shape} | test {X_test.shape} | dtype {X_fit.dtype} | "
      f"{(X_fit.nbytes + X_val.nbytes + X_test.nbytes) / 1e9:.2f} GB")
""")

code(r"""
def augmentation(seed=SEED):
    return keras.Sequential([layers.RandomRotation(0.04, seed=seed), layers.RandomTranslation(0.05, 0.05, seed=seed + 1),
                             layers.RandomZoom(0.08, seed=seed + 2), layers.RandomContrast(0.10, seed=seed + 3)], name="augmentation")

# Visual check: one original image and 7 random augmentations of it
aug_demo = augmentation(); example = X_fit[np.random.default_rng(3).integers(len(X_fit))]
fig, axes = plt.subplots(2, 8, figsize=(16, 4.4))
for ax in axes.ravel(): ax.axis("off")
axes[0, 0].imshow(example); axes[0, 0].set_title("original", fontsize=9)
for i in range(1, 16):
    axes.ravel()[i].imshow(np.clip(aug_demo(example[None].astype("float32"), training=True)[0].numpy(), 0, 255).astype("uint8")); axes.ravel()[i].set_title("augmented", fontsize=8)
plt.suptitle("Augmentation preview (training-time only)", y=1.02); plt.tight_layout(); savefig("augmentation_preview"); plt.show()

# Why no flipping: demonstrate what a flip does to a directional sign
c_left, c_right = 34, 33
fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))
axes[0].imshow(X_fit[y_fit == c_left][0] if (y_fit == c_left).any() else example); axes[0].set_title(f"'{CLASS_NAMES[c_left]}'", fontsize=9)
axes[1].imshow((X_fit[y_fit == c_left][0] if (y_fit == c_left).any() else example)[:, ::-1]); axes[1].set_title("…flipped", fontsize=9)
axes[2].imshow(X_fit[y_fit == c_right][0] if (y_fit == c_right).any() else example); axes[2].set_title(f"'{CLASS_NAMES[c_right]}'", fontsize=9)
for a in axes: a.axis("off")
plt.suptitle("A flipped ‘left’ sign looks like a ‘right’ sign but would keep its ‘left’ label", fontsize=10, y=1.08); plt.show()
""")

# ───────────────────────────── 7. Models ─────────────────────────────
md(r"""
---
## 7 · Model design

We compare two deliberately different approaches to expose the accuracy/cost trade-off rather than a single number.

### A. Custom CNN (trained from scratch)
Three Conv-BatchNorm-MaxPool-Dropout blocks (32→64→128 filters), global average pooling, a 128-unit dense layer, softmax over 43 classes.

| Choice | Reason |
|---|---|
| Small, ~0.1 M parameters | Signs are simple shapes at low resolution; a big network over-fits and is slow on embedded hardware |
| BatchNorm | Faster, more stable convergence |
| Global average pooling | Far fewer parameters than flatten→dense, less over-fitting, keeps spatial maps for Grad-CAM |
| Dropout 0.2 / 0.3 | Regularisation against memorising individual tracks |

### B. MobileNetV2 (ImageNet transfer learning)
A pretrained efficient backbone with a new 43-way head. Trained in **two stages**:
1. **Head only** (backbone frozen) — adapt the classifier without destroying pretrained features.
2. **Fine-tune** the top backbone layers at a **10× smaller learning rate**, with BatchNorm statistics kept frozen (the standard recipe for small-batch / small-domain fine-tuning).

Why two stages? A frozen ImageNet backbone sees natural photographs, not 64-pixel sign crops; a head alone has a *domain gap* it cannot close. Fine-tuning lets the later layers specialise.

Both models take **raw 0–255 pixels** and contain their own augmentation and normalisation, so they can be deployed as-is.
""")

code(r"""
def build_custom_cnn():
    inp = layers.Input((IMG, IMG, 3), name="image")                      # raw pixels 0–255
    x = augmentation()(inp)                                               # training-only
    x = layers.Rescaling(1 / 255.0)(x)
    for i, f in enumerate((32, 64, 128), 1):
        x = layers.Conv2D(f, 3, padding="same", activation="relu", name=f"conv{i}")(x)
        x = layers.BatchNormalization()(x); x = layers.MaxPooling2D()(x); x = layers.Dropout(0.2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", name="embedding")(x); x = layers.Dropout(0.3)(x)
    return keras.Model(inp, layers.Dense(NUM_CLASSES, activation="softmax", name="probs")(x), name="custom_cnn")

def build_mobilenet():
    inp = layers.Input((IMG, IMG, 3), name="image")
    x = augmentation()(inp)
    x = layers.Resizing(MB_SIZE, MB_SIZE, interpolation="bilinear")(x)   # upsample on-GPU to MobileNet's working size
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)                         # MobileNetV2 expects [-1, 1]
    try:
        base = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(MB_SIZE, MB_SIZE, 3), pooling="avg", name="backbone")
    except Exception as e:                                                # offline fallback keeps the notebook runnable
        print("WARNING: ImageNet weights unavailable (", type(e).__name__, ") → random init"); base = keras.applications.MobileNetV2(include_top=False, weights=None, input_shape=(MB_SIZE, MB_SIZE, 3), pooling="avg", name="backbone")
    base.trainable = False
    x = base(x, training=False); x = layers.Dropout(0.25)(x)
    return keras.Model(inp, layers.Dense(NUM_CLASSES, activation="softmax", name="probs")(x), name="mobilenet_v2")

cnn = build_custom_cnn(); mobilenet = build_mobilenet()
cnn.summary(line_length=100)
print("\nMobileNetV2 model (backbone frozen):")
mobilenet.summary(line_length=100)
""")

# ───────────────────────────── 8. Training ─────────────────────────────
md(r"""
---
## 8 · Training

**Common recipe**
- Adam optimiser (Kingma & Ba, 2015) with sparse categorical cross-entropy.
- **EarlyStopping** on validation accuracy with `restore_best_weights=True` → the epoch budget is a ceiling, and we always keep the best epoch.
- **ReduceLROnPlateau** → automatic learning-rate decay when validation loss stalls.
- Model selection uses **validation** metrics only.

Class weights were *not* applied: per-class results in §9 show whether the imbalance actually hurts; the reported macro-F1 guards against a misleadingly high accuracy.
""")

code(r"""
def fit(model, epochs, lr, patience, tag):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = [keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=patience, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=max(1, patience // 2), min_lr=1e-6)]
    t0 = time.time()
    h = model.fit(X_fit, y_fit, validation_data=(X_val, y_val), batch_size=BATCH, epochs=epochs, callbacks=cbs, verbose=2)
    print(f"[{tag}] {len(h.history['loss'])} epochs in {time.time() - t0:.0f}s | best val acc {max(h.history['val_accuracy']):.4f}")
    return h.history, time.time() - t0

def plot_history(hist, title, boundary=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for ax, k, name in zip(axes, ("accuracy", "loss"), ("Accuracy", "Loss")):
        ax.plot(hist[k], "o-", color=BLUE, label="train"); ax.plot(hist["val_" + k], "o-", color=ORANGE, label="validation")
        if boundary is not None: ax.axvline(boundary - .5, color="grey", ls="--"); ax.text(boundary - .4, ax.get_ylim()[0] + .02 * np.ptp(ax.get_ylim()), " fine-tuning →", color="grey", fontsize=9)
        ax.set(title=f"{title} — {name}", xlabel="epoch"); ax.legend()
    plt.tight_layout(); savefig("curves_" + title.lower().replace(" ", "_")); plt.show()

hist_cnn, time_cnn = fit(cnn, CFG["cnn_epochs"], 1e-3, patience=5, tag="Custom CNN")
plot_history(hist_cnn, "Custom CNN")
""")

code(r"""
# MobileNetV2 stage 1 — train the new head only
hist_head, time_head = fit(mobilenet, CFG["head_epochs"], 1e-3, patience=4, tag="MobileNetV2 · stage 1 (head)")
mobilenet.save_weights(WORK / "mobilenet_stage1.weights.h5")                 # snapshot for the head-only ablation
plot_history(hist_head, "MobileNetV2 stage 1")
""")

code(r"""
# MobileNetV2 stage 2 — fine-tune the top of the backbone at a 10× lower learning rate
FT_LAYERS = 50
backbone = mobilenet.get_layer("backbone"); backbone.trainable = True
for l in backbone.layers[:-FT_LAYERS]: l.trainable = False
for l in backbone.layers:
    if isinstance(l, layers.BatchNormalization): l.trainable = False        # keep ImageNet BN statistics fixed
print(f"Trainable parameters: {sum(int(np.prod(w.shape)) for w in mobilenet.trainable_weights):,} of {mobilenet.count_params():,}")

hist_ft, time_ft = fit(mobilenet, CFG["ft_epochs"], 1e-4, patience=4, tag="MobileNetV2 · stage 2 (fine-tune)")
hist_mb = {k: hist_head[k] + hist_ft[k] for k in hist_head}
plot_history(hist_mb, "MobileNetV2 two-stage", boundary=len(hist_head["loss"]))
""")

# ───────────────────────────── 9. Evaluation ─────────────────────────────
md(r"""
---
## 9 · Evaluation on the official test set

Metrics reported:
- **Accuracy** — overall fraction correct.
- **Macro-F1** — F1 averaged over classes with equal weight (the honest metric under imbalance).
- **Top-3 accuracy** — is the right answer among the 3 most likely?
- **ECE** (expected calibration error) — do confidence scores mean what they say? (§11)
- **Latency** and **size** — the deployment cost of each model. Latency is timed through single, un-compiled calls on the Colab GPU runtime, so it includes Python overhead; use it to compare the models with each other, not as an embedded-hardware figure.

To isolate *what each design decision bought us* we also score the head-only MobileNetV2.
""")

code(r"""
def ece_score(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e

def predict_probs(model, X): return model.predict(X, batch_size=256, verbose=0)

def score_from_probs(p, y):
    pred = p.argmax(1); conf = p.max(1)
    return dict(probs=p, pred=pred, conf=conf, accuracy=(pred == y).mean(), macro_f1=f1_score(y, pred, average="macro", labels=range(NUM_CLASSES), zero_division=0),
                top3=np.mean([yy in np.argsort(pp)[-3:] for yy, pp in zip(y, p)]), ece=ece_score(conf, pred == y))

def score(model, X, y): return score_from_probs(predict_probs(model, X), y)

def latency_ms(model, X, n=200):
    model.predict(X[:1], verbose=0); t = time.time()
    for i in range(n): model(X[i:i + 1], training=False)
    return (time.time() - t) / n * 1000

mobilenet_final_weights = mobilenet.get_weights()
mobilenet.load_weights(WORK / "mobilenet_stage1.weights.h5"); R_head = score(mobilenet, X_test, y_test); V_head = score(mobilenet, X_val, y_val)
mobilenet.set_weights(mobilenet_final_weights)

R = {"Custom CNN": score(cnn, X_test, y_test), "MobileNetV2 (head only)": R_head, "MobileNetV2 (fine-tuned)": score(mobilenet, X_test, y_test)}
V = {"Custom CNN": score(cnn, X_val, y_val), "MobileNetV2 (head only)": V_head, "MobileNetV2 (fine-tuned)": score(mobilenet, X_val, y_val)}
params = {"Custom CNN": cnn.count_params(), "MobileNetV2 (head only)": mobilenet.count_params(), "MobileNetV2 (fine-tuned)": mobilenet.count_params()}
lat = {"Custom CNN": latency_ms(cnn, X_test), "MobileNetV2 (fine-tuned)": latency_ms(mobilenet, X_test)}; lat["MobileNetV2 (head only)"] = lat["MobileNetV2 (fine-tuned)"]

table = pd.DataFrame({
    "Val accuracy": {k: V[k]["accuracy"] for k in R}, "Test accuracy": {k: R[k]["accuracy"] for k in R}, "Test macro-F1": {k: R[k]["macro_f1"] for k in R},
    "Test top-3": {k: R[k]["top3"] for k in R}, "ECE ↓": {k: R[k]["ece"] for k in R},
    "Params (M)": {k: params[k] / 1e6 for k in R}, "Latency (ms/img, bs=1)": lat}).round(4)
display(table.style.format({"Val accuracy": "{:.2%}", "Test accuracy": "{:.2%}", "Test macro-F1": "{:.2%}", "Test top-3": "{:.2%}", "ECE ↓": "{:.3f}", "Params (M)": "{:.2f}", "Latency (ms/img, bs=1)": "{:.1f}"})
        .highlight_max(subset=["Test accuracy", "Test macro-F1"], color="#d7f5d7").highlight_min(subset=["ECE ↓"], color="#d7f5d7"))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
names = list(R); xs = np.arange(len(names)); w = .38
axes[0].bar(xs - w / 2, [R[k]["accuracy"] for k in names], w, label="Accuracy", color=BLUE); axes[0].bar(xs + w / 2, [R[k]["macro_f1"] for k in names], w, label="Macro-F1", color=ORANGE)
axes[0].set_xticks(xs, [n.replace(" (", "\n(") for n in names], fontsize=9); axes[0].set(title="Official test performance", ylim=(0, 1.05)); axes[0].legend(loc="lower right")
for i, k in enumerate(names): axes[0].text(i - w / 2, R[k]["accuracy"] + .01, f"{R[k]['accuracy']:.1%}", ha="center", fontsize=8); axes[0].text(i + w / 2, R[k]["macro_f1"] + .01, f"{R[k]['macro_f1']:.1%}", ha="center", fontsize=8)
axes[1].scatter([lat[k] for k in names], [R[k]["accuracy"] for k in names], s=[params[k] / 1e4 + 40 for k in names], c=[BLUE, ORANGE, GREEN], alpha=.8)
for k in names: axes[1].annotate(k, (lat[k], R[k]["accuracy"]), textcoords="offset points", xytext=(6, 6), fontsize=8)
axes[1].set(title="Accuracy vs latency (bubble ∝ parameters)", xlabel="ms per image (batch 1)", ylabel="Test accuracy")
plt.tight_layout(); savefig("model_comparison"); plt.show()
table.to_csv(ART / "model_comparison.csv")
""")

code(r"""
# Model selection uses VALIDATION accuracy only (never the test set)
CANDIDATES = {"Custom CNN": cnn, "MobileNetV2 (fine-tuned)": mobilenet}
BEST_NAME = max(CANDIDATES, key=lambda k: V[k]["accuracy"]); BEST = CANDIDATES[BEST_NAME]; RB = R[BEST_NAME]
print(f"Selected model (by validation accuracy): {BEST_NAME}")
print(f"→ official test accuracy {RB['accuracy']:.2%}, macro-F1 {RB['macro_f1']:.2%}, top-3 {RB['top3']:.2%}")
""")

code(r"""
# 9.2 Per-class performance of the selected model
rep = classification_report(y_test, RB["pred"], labels=range(NUM_CLASSES), output_dict=True, zero_division=0)
per = pd.DataFrame({"class": range(NUM_CLASSES), "name": CLASS_NAMES, "precision": [rep[str(c)]["precision"] for c in range(NUM_CLASSES)],
                    "recall": [rep[str(c)]["recall"] for c in range(NUM_CLASSES)], "f1": [rep[str(c)]["f1-score"] for c in range(NUM_CLASSES)],
                    "support": [rep[str(c)]["support"] for c in range(NUM_CLASSES)]})
per["train_images"] = per["class"].map(tr_counts)
per.to_csv(ART / "per_class_metrics.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw=dict(width_ratios=[1.4, 1]))
ps = per.sort_values("f1"); axes[0].barh([short(c, 30) for c in ps["class"]], ps.f1, color=[RED if f < .9 else BLUE for f in ps.f1])
axes[0].axvline(rep["macro avg"]["f1-score"], color="k", ls="--", lw=1); axes[0].set(title=f"Per-class F1 — {BEST_NAME}  (dashed = macro avg)", xlabel="F1"); axes[0].tick_params(axis="y", labelsize=7.5)
axes[1].scatter(per.train_images, per.f1, c=[RED if f < .9 else BLUE for f in per.f1], s=50)
for _, r in per.nsmallest(5, "f1").iterrows(): axes[1].annotate(short(int(r["class"]), 20), (r.train_images, r.f1), fontsize=7, xytext=(5, 3), textcoords="offset points")
axes[1].set(title="Does less training data mean lower F1?", xlabel="training images in class", ylabel="F1")
plt.tight_layout(); savefig("per_class_f1"); plt.show()

rho = per.f1.corr(per.train_images, method="spearman")
print(f"Spearman correlation between class size and F1: {rho:+.2f}  (|ρ| < 0.3 ⇒ weak: imbalance alone does not explain the errors)")
print("\nHardest classes:"); display(per.nsmallest(8, "f1")[["class", "name", "precision", "recall", "f1", "support", "train_images"]].round(3).style.hide(axis="index"))
""")

code(r"""
# 9.3 Confusion analysis
cm = confusion_matrix(y_test, RB["pred"], labels=range(NUM_CLASSES)); np.save(ART / "confusion_matrix.npy", cm)
cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
fig, ax = plt.subplots(figsize=(10, 8.5)); im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set(title=f"Row-normalised confusion matrix — {BEST_NAME}", xlabel="Predicted class", ylabel="True class"); ax.grid(False); plt.colorbar(im, fraction=.046, label="share of true class")
plt.tight_layout(); savefig("confusion_matrix"); plt.show()

off = cm.copy(); np.fill_diagonal(off, 0)
pairs = pd.DataFrame([(i, j, off[i, j]) for i, j in zip(*np.nonzero(off))], columns=["true", "pred", "count"]).sort_values("count", ascending=False).head(10)
pairs["true_name"] = pairs.true.map(short); pairs["pred_name"] = pairs.pred.map(short)
display(pairs[["true", "true_name", "pred", "pred_name", "count"]].style.hide(axis="index"))

fig, ax = plt.subplots(figsize=(9, 4)); lab = [f"{short(t, 20)}  →  {short(p, 20)}" for t, p in zip(pairs.true, pairs.pred)][::-1]
ax.barh(lab, pairs["count"][::-1], color=PURPLE); ax.set(title="Ten most frequent confusions", xlabel="test images"); ax.tick_params(axis="y", labelsize=8); plt.tight_layout(); savefig("top_confusions"); plt.show()
""")

code(r"""
# Are the mistakes *semantically* close? Compare confusions inside vs across sign families.
FAMILY = {**{c: "Speed limit" for c in (0, 1, 2, 3, 4, 5, 7, 8)}, **{c: "Prohibitory" for c in (9, 10, 15, 16, 17)},
          **{c: "Danger / warning" for c in (11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)},
          **{c: "Mandatory" for c in (33, 34, 35, 36, 37, 38, 39, 40)}, **{c: "Other / end / priority" for c in (6, 12, 13, 14, 32, 41, 42)}}
err = RB["pred"] != y_test
same = np.mean([FAMILY[t] == FAMILY[p] for t, p in zip(y_test[err], RB["pred"][err])]) if err.any() else float("nan")
print(f"{err.sum()} errors out of {len(y_test)}; {same:.0%} of them stay inside the same sign family "
      f"(random guessing would give ≈{sum((sum(f == k for f in FAMILY.values()) / 43) ** 2 for k in set(FAMILY.values())):.0%}).")
print("Families:", {k: sum(f == k for f in FAMILY.values()) for k in sorted(set(FAMILY.values()))})
""")

code(r"""
# Misclassification gallery — the most *confident* mistakes are the ones that matter in practice
bad = np.where(err)[0]; bad = bad[np.argsort(-RB["conf"][bad])][:12]
fig, axes = plt.subplots(2, 6, figsize=(16, 6));
for ax in axes.ravel(): ax.axis("off")
for ax, i in zip(axes.ravel(), bad):
    ax.imshow(X_test[i]); ax.set_title(f"true: {short(y_test[i], 18)}\npred: {short(RB['pred'][i], 18)}\nconf {RB['conf'][i]:.0%}", fontsize=8, color=RED)
plt.suptitle("Most confident errors on the official test set", y=1.0); plt.tight_layout(); savefig("confident_errors"); plt.show()
""")

md(r"""
---
## 9A · Progress ladder: from a naive baseline to the final model

A single final number says little about *why* the design works. Here we rebuild the story as an **experiment ladder**: start from a trivial baseline and change **one thing per step**, measuring each step on the **validation set** (the test set stays untouched).

| Step | Change | Question it answers |
|---|---|---|
| 0 | Always predict the most common class | What does "knowing nothing" score? |
| 1 | Linear classifier on tiny pixel images | How far do raw pixels get us? |
| 2 | Add hidden layers (MLP) | Does non-linearity help without spatial structure? |
| 3 | Convolutions | Does exploiting spatial structure help? |
| 4 | Deeper CNN, BatchNorm, global pooling | Does a better-conditioned architecture help? |
| 5 | Augmentation and dropout (the final custom CNN of §8) | Does regularisation improve generalisation to unseen signs? |
| 6 | Pretrained MobileNetV2, frozen backbone | How does transfer learning do on its own? |
| 7 | Fine-tune the top backbone layers | Does adapting the features close the domain gap? |
| 8 | Average the two best models (ensemble) | Do the models make different mistakes? |

Steps 0-5 form the *from-scratch* track and steps 6-7 the *transfer-learning* track; step 8 combines both. Steps 5-7 reuse the models already trained in §8, so only steps 0-4 train here.
""")

code(r"""
# Train the early rungs of the ladder (same data, same validation set, same optimiser as the final models)
def ladder_model(kind):
    inp = layers.Input((IMG, IMG, 3)); x = layers.Rescaling(1 / 255.0)(inp)
    if kind == "linear":
        x = layers.Flatten()(layers.AveragePooling2D(4)(x))                       # 16x16 pixels
    elif kind == "mlp":
        x = layers.Flatten()(layers.AveragePooling2D(2)(x)); x = layers.Dense(256, activation="relu")(x); x = layers.Dense(128, activation="relu")(x)
    elif kind == "basic_cnn":
        for f in (32, 64): x = layers.MaxPooling2D()(layers.Conv2D(f, 3, activation="relu")(x))
        x = layers.Dense(128, activation="relu")(layers.Flatten()(x))
    elif kind == "cnn_bn":
        for f in (32, 64, 128): x = layers.MaxPooling2D()(layers.BatchNormalization()(layers.Conv2D(f, 3, padding="same", activation="relu")(x)))
        x = layers.Dense(128, activation="relu")(layers.GlobalAveragePooling2D()(x))
    return keras.Model(inp, layers.Dense(NUM_CLASSES, activation="softmax")(x), name=f"ladder_{kind}")

LADDER = [("linear", "Linear classifier on 16x16 pixels", CFG["ladder_epochs"]), ("mlp", "MLP on 32x32 pixels", CFG["ladder_epochs"]),
          ("basic_cnn", "Basic CNN (2 conv layers)", CFG["cnn_epochs"]), ("cnn_bn", "Deeper CNN + BatchNorm + global pooling", CFG["cnn_epochs"])]
ladder_scores = {}
for kind, label, ep in LADDER:
    m = ladder_model(kind); fit(m, ep, 1e-3, patience=5, tag=label)
    ladder_scores[label] = (score(m, X_val, y_val), m.count_params())
""")

code(r"""
# Ensemble: average the class probabilities of the two trained models (no extra training)
def add_ensemble(Xs, y, A, B): return score_from_probs((A["probs"] + B["probs"]) / 2, y)
R["Ensemble"] = add_ensemble(X_test, y_test, R["Custom CNN"], R["MobileNetV2 (fine-tuned)"])
V["Ensemble"] = add_ensemble(X_val, y_val, V["Custom CNN"], V["MobileNetV2 (fine-tuned)"])
params["Ensemble"] = params["Custom CNN"] + params["MobileNetV2 (fine-tuned)"]; lat["Ensemble"] = lat["Custom CNN"] + lat["MobileNetV2 (fine-tuned)"]

majority = np.bincount(y_fit, minlength=NUM_CLASSES).argmax()
base_acc = float((y_val == majority).mean()); base_f1 = f1_score(y_val, np.full_like(y_val, majority), average="macro", labels=range(NUM_CLASSES), zero_division=0)
rows = [(0, "Majority-class baseline", "baseline", "Always predict the most common class", base_acc, base_f1, 0)]
for n, (kind, label, _) in enumerate(LADDER, 1):
    sc, npar = ladder_scores[label]; rows.append((n, label, "from scratch", label, sc["accuracy"], sc["macro_f1"], npar))
rows += [(5, "Final custom CNN (+ augmentation, dropout)", "from scratch", "Adds augmentation and dropout (section 8)", V["Custom CNN"]["accuracy"], V["Custom CNN"]["macro_f1"], params["Custom CNN"]),
         (6, "MobileNetV2, frozen backbone", "transfer learning", "Pretrained features, new head only", V["MobileNetV2 (head only)"]["accuracy"], V["MobileNetV2 (head only)"]["macro_f1"], params["MobileNetV2 (head only)"]),
         (7, "MobileNetV2, fine-tuned", "transfer learning", "Adapt the top backbone layers at a lower LR", V["MobileNetV2 (fine-tuned)"]["accuracy"], V["MobileNetV2 (fine-tuned)"]["macro_f1"], params["MobileNetV2 (fine-tuned)"]),
         (8, "Ensemble (CNN + MobileNetV2)", "combined", "Average the two models' probabilities", V["Ensemble"]["accuracy"], V["Ensemble"]["macro_f1"], params["Ensemble"])]
ladder_df = pd.DataFrame(rows, columns=["step", "model", "track", "change", "val_accuracy", "val_macro_f1", "parameters"])
ladder_df["best_so_far"] = ladder_df.val_accuracy.cummax()
ladder_df["gain_vs_previous_best"] = ladder_df.best_so_far.diff().fillna(0)
ladder_df.to_csv(ART / "progress_ladder.csv", index=False)
display(ladder_df.drop(columns="best_so_far").style.hide(axis="index").format({"val_accuracy": "{:.2%}", "val_macro_f1": "{:.2%}", "parameters": "{:,}", "gain_vs_previous_best": "{:+.2%}"}))

TRACK_COLOR = {"baseline": "#8c8c8c", "from scratch": BLUE, "transfer learning": ORANGE, "combined": GREEN}
labels = ["0\nMajority\nclass", "1\nLinear\n16x16 px", "2\nMLP\n32x32 px", "3\nBasic CNN\n2 conv", "4\nDeeper CNN\n+ BatchNorm", "5\n+ Augment.\n+ dropout", "6\nMobileNetV2\nfrozen", "7\nMobileNetV2\nfine-tuned", "8\nEnsemble\nCNN + MNV2"]
fig, axes = plt.subplots(1, 2, figsize=(19, 6.5), gridspec_kw=dict(width_ratios=[1.55, 1]))
ax = axes[0]; ax.bar(ladder_df.step, ladder_df.val_accuracy * 100, color=[TRACK_COLOR[t] for t in ladder_df.track], width=.72)
ax.step(ladder_df.step, ladder_df.best_so_far * 100, where="mid", color="k", lw=2, label="best so far")
for r in ladder_df.itertuples(): ax.text(r.step, r.val_accuracy * 100 + 1, f"{r.val_accuracy:.1%}", ha="center", fontsize=9)
ax.set_xticks(ladder_df.step, labels, fontsize=8); ax.set(ylim=(0, 124), ylabel="Validation accuracy (%)", title="Progress ladder: validation accuracy at each step")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=t) for t, c in TRACK_COLOR.items()] + [plt.Line2D([], [], color="k", lw=2, label="best so far")], loc="upper center", ncol=5, fontsize=8, frameon=False)
ax = axes[1]; ax.bar(ladder_df.step, (1 - ladder_df.val_accuracy) * 100, color=[TRACK_COLOR[t] for t in ladder_df.track], width=.72); ax.set_yscale("log")
for r in ladder_df.itertuples(): ax.text(r.step, (1 - r.val_accuracy) * 100 * 1.12, f"{(1 - r.val_accuracy) * 100:.1f}%", ha="center", fontsize=9)
ax.set_xticks(ladder_df.step); ax.set(xlabel="step", ylabel="Validation error (%) - log scale", title="The same ladder as error rate")
plt.tight_layout(); savefig("progress_ladder"); plt.show()

b = ladder_df.iloc[ladder_df.val_accuracy.idxmax()]
print(f"Baseline {ladder_df.val_accuracy.iloc[0]:.1%} -> best step {int(b.step)} ({b.model}) {b.val_accuracy:.2%} on validation; "
      f"error reduced by a factor of {(1 - ladder_df.val_accuracy.iloc[0]) / max(1e-9, 1 - b.val_accuracy):.0f}.")
print(f"Ensemble on the official test set: accuracy {R['Ensemble']['accuracy']:.2%}, macro-F1 {R['Ensemble']['macro_f1']:.2%} "
      f"(best single model: {max(R['Custom CNN']['accuracy'], R['MobileNetV2 (fine-tuned)']['accuracy']):.2%})")
""")

# ───────────────────────────── 10. Explainability ─────────────────────────────
md(r"""
---
## 10 · Explainability

Accuracy does not prove the model is looking at the *right thing*. A classifier could, in principle, rely on background cues that happen to correlate with a class.

### Grad-CAM
Grad-CAM (Selvaraju et al., ICCV 2017) weights the last convolutional feature maps by the gradient of the predicted class score, producing a heat-map of *where the network looked*. Red = strong influence.

- **Custom CNN:** final conv block (16×16 map).
- **MobileNetV2:** last conv layer `out_relu` (4×4 map — coarse by design, so its heat-maps are blurrier).
""")

code(r"""
_cam_cache = {}
def _cam_parts(model):
    if model.name in _cam_cache: return _cam_cache[model.name]
    try:
        bb = model.get_layer("backbone"); conv = bb.get_layer("out_relu")
        parts = ("nested", keras.Model(bb.input, [conv.output, bb.output]), model.layers.index(bb))
    except ValueError:
        parts = ("flat", keras.Model(model.inputs, [model.get_layer("conv3").output, model.output]), None)
    _cam_cache[model.name] = parts; return parts

def gradcam(model, x_uint8, target=None):
    kind, cam_model, idx = _cam_parts(model); x = tf.convert_to_tensor(x_uint8[None].astype("float32"))
    with tf.GradientTape() as tape:
        if kind == "flat": conv_out, probs = cam_model(x, training=False)
        else:
            h = x
            for l in model.layers[1:idx]: h = l(h, training=False)
            conv_out, h = cam_model(h, training=False)
            for l in model.layers[idx + 1:]: h = l(h, training=False)
            probs = h
        cls = int(tf.argmax(probs[0])) if target is None else target
        score = probs[:, cls]
    grads = tape.gradient(score, conv_out)
    cam = tf.nn.relu(tf.reduce_sum(conv_out * tf.reduce_mean(grads, axis=(1, 2))[:, None, None, :], axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    return cv2.resize(cam, (x_uint8.shape[1], x_uint8.shape[0]), interpolation=cv2.INTER_CUBIC).clip(0, 1), cls, float(probs[0, cls])

def overlay(ax, img, cam): ax.imshow(img, interpolation="bilinear"); ax.imshow(cam, cmap="jet", alpha=.45, interpolation="bilinear"); ax.axis("off")

rng = np.random.default_rng(7)
correct_idx = rng.choice(np.where(~err)[0], 5, replace=False); wrong_idx = bad[:5] if len(bad) >= 5 else bad
idxs = list(correct_idx) + list(wrong_idx)
fig, axes = plt.subplots(3, len(idxs), figsize=(2.1 * len(idxs), 7.2))
for j, i in enumerate(idxs):
    axes[0, j].imshow(X_test[i], interpolation="bilinear"); axes[0, j].axis("off"); axes[0, j].set_title(("correct: " if j < 5 else "wrong: ") + short(y_test[i], 16), fontsize=8, color=GREEN if j < 5 else RED)
    for r, (nm, m) in enumerate([("Custom CNN", cnn), ("MobileNetV2", mobilenet)], start=1):
        cam, cls, pr = gradcam(m, X_test[i]); overlay(axes[r, j], X_test[i], cam); axes[r, j].set_title(f"{nm}\n→ {short(cls, 14)} {pr:.0%}", fontsize=7)
plt.suptitle("Grad-CAM: left 5 = correct predictions, right 5 = confident errors (of the selected model)", y=1.0); plt.tight_layout(); savefig("gradcam_grid"); plt.show()
""")

code(r"""
# Embedding map: what does the network's internal representation look like? (t-SNE of the penultimate layer)
from sklearn.manifold import TSNE
embedder = keras.Model(BEST.input, BEST.layers[-1].input)
n = min(CFG["tsne_n"], len(X_test)); sel = np.random.default_rng(SEED).choice(len(X_test), n, replace=False)
emb = embedder.predict(X_test[sel], batch_size=256, verbose=0)
xy = TSNE(n_components=2, perplexity=min(30, n // 4), init="pca", random_state=SEED).fit_transform(emb)
fams = sorted(set(FAMILY.values())); cols = dict(zip(fams, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]))
fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
for f in fams:
    m = np.array([FAMILY[c] == f for c in y_test[sel]]); axes[0].scatter(*xy[m].T, s=6, color=cols[f], label=f, alpha=.7)
axes[0].legend(markerscale=3, fontsize=8); axes[0].set_title(f"t-SNE of {BEST_NAME} embeddings — coloured by sign family")
axes[1].scatter(*xy.T, s=6, c=y_test[sel], cmap="nipy_spectral", alpha=.75); axes[1].set_title("…coloured by individual class (43 clusters)")
for a in axes: a.set_xticks([]); a.set_yticks([]); a.grid(False)
plt.tight_layout(); savefig("tsne_embeddings"); plt.show()
""")

# ───────────────────────────── 11. Reliability ─────────────────────────────
md(r"""
---
## 11 · Reliability: calibration, selective prediction and robustness

A production classifier must know **when it is unsure** and degrade **gracefully** on bad input. Three checks:

1. **Calibration** — when the model says 90 % confident, is it right ~90 % of the time? (reliability diagram, ECE)
2. **Selective prediction** — if we *abstain* below a confidence threshold and hand over to a human/other sensor, how much accuracy do we gain and what coverage do we give up?
3. **Robustness** — accuracy under controlled corruptions the real world produces: darkness, glare, noise, blur, occlusion, tilt.
""")

code(r"""
# 11.1 Calibration + selective prediction (selected model, official test set)
conf, correct = RB["conf"], RB["pred"] == y_test
edges = np.linspace(0, 1, 11); mids = (edges[:-1] + edges[1:]) / 2
acc_bin = [correct[(conf > lo) & (conf <= hi)].mean() if ((conf > lo) & (conf <= hi)).any() else np.nan for lo, hi in zip(edges[:-1], edges[1:])]
ths = np.linspace(0.3, 0.999, 80); cover = np.array([(conf >= t).mean() for t in ths]); acc_cov = np.array([correct[conf >= t].mean() if (conf >= t).any() else np.nan for t in ths])

fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
axes[0].hist(conf[correct], bins=30, alpha=.75, color=GREEN, label="correct"); axes[0].hist(conf[~correct], bins=30, alpha=.8, color=RED, label="wrong")
axes[0].set(title="Confidence of correct vs wrong predictions", xlabel="max softmax probability", ylabel="images"); axes[0].set_yscale("log"); axes[0].legend()
axes[1].plot([0, 1], [0, 1], "k--", label="perfect calibration"); axes[1].bar(mids, np.nan_to_num(acc_bin), width=.09, color=BLUE, alpha=.8, edgecolor="white", label="model")
axes[1].set(title=f"Reliability diagram  (ECE = {RB['ece']:.3f})", xlabel="confidence", ylabel="accuracy in bin"); axes[1].legend()
axes[2].plot(cover, acc_cov, color=PURPLE, lw=2); axes[2].set(title="Selective prediction: abstain when unsure", xlabel="coverage (share of images answered)", ylabel="accuracy on answered images")
axes[2].invert_xaxis()
plt.tight_layout(); savefig("calibration_selective"); plt.show()

for target in (0.99, 0.995, 0.999):
    ok = np.where(acc_cov >= target)[0]
    print(f"To reach ≥{target:.1%} accuracy: " + (f"threshold {ths[ok[0]]:.3f} → answers {cover[ok[0]]:.1%} of images, abstains on {1 - cover[ok[0]]:.1%}" if len(ok) else "not reachable at the thresholds tried"))
""")

code(r"""
# 11.2 Robustness to real-world corruptions (selected model)
def c_brightness(X, f): return np.clip(X.astype(np.float32) * f, 0, 255).astype(np.uint8)
def c_noise(X, s): return np.clip(X + np.random.default_rng(1).normal(0, s, X.shape), 0, 255).astype(np.uint8)
def c_blur(X, k): return np.stack([cv2.GaussianBlur(x, (k, k), 0) for x in X])
def c_rotate(X, d):
    M = cv2.getRotationMatrix2D((IMG / 2, IMG / 2), d, 1.0); return np.stack([cv2.warpAffine(x, M, (IMG, IMG), borderMode=cv2.BORDER_REPLICATE) for x in X])
def c_occlude(X, frac):
    side = int(IMG * math.sqrt(frac)); r = np.random.default_rng(2); out = X.copy()
    for x in out: i, j = r.integers(0, IMG - side + 1, 2); x[i:i + side, j:j + side] = 0
    return out

CORR = {"Darkening (×brightness)": (c_brightness, [0.7, 0.5, 0.3], "×{}"), "Glare (×brightness)": (c_brightness, [1.4, 1.8, 2.4], "×{}"),
        "Gaussian noise (σ)": (c_noise, [10, 25, 50], "σ={}"), "Blur (kernel)": (c_blur, [3, 5, 7], "k={}"),
        "Rotation (°)": (c_rotate, [10, 20, 35], "{}°"), "Occlusion (area share)": (c_occlude, [0.1, 0.2, 0.35], "{:.0%}")}
n = min(CFG["robust_n"], len(X_test)); Xs, ys = X_test[:n], y_test[:n]
clean = (predict_probs(BEST, Xs).argmax(1) == ys).mean(); rob = {}
for name, (fn, levels, _) in CORR.items(): rob[name] = [(predict_probs(BEST, fn(Xs, lv)).argmax(1) == ys).mean() for lv in levels]

fig, axes = plt.subplots(1, 6, figsize=(18, 3.6), sharey=True)
for ax, (name, (fn, levels, fmt)) in zip(axes, CORR.items()):
    ax.plot(range(4), [clean] + rob[name], "o-", color=BLUE, lw=2); ax.set_xticks(range(4), ["clean"] + [fmt.format(l) for l in levels], fontsize=8); ax.set_title(name, fontsize=9); ax.set_ylim(0, 1.02)
axes[0].set_ylabel("accuracy"); plt.suptitle(f"Robustness of {BEST_NAME} on {n:,} test images", y=1.04); plt.tight_layout(); savefig("robustness"); plt.show()

fig, axes = plt.subplots(len(CORR), 4, figsize=(7, 1.8 * len(CORR))); ex = Xs[:1]
for r, (name, (fn, levels, fmt)) in enumerate(CORR.items()):
    for c, im in enumerate([ex] + [fn(ex, lv) for lv in levels]): axes[r, c].imshow(im[0]); axes[r, c].axis("off")
    axes[r, 0].set_title(name, fontsize=7, loc="left")
plt.tight_layout(); savefig("robustness_examples"); plt.show()
pd.DataFrame(rob, index=["mild", "medium", "severe"]).T.assign(clean=clean).round(3)
""")

# ───────────────────────────── 12. Inference demo ─────────────────────────────
md(r"""
---
## 12 · Live inference demo

A reusable `predict_sign()` that takes **any image** (file path, URL-downloaded file, or NumPy array) and returns the top-5 classes with confidences plus a Grad-CAM heat-map. 
""")

code(r"""
def predict_sign(image, model=None, top_k=5, show=True):
    model = model or BEST
    img = np.asarray(Image.open(image).convert("RGB").resize((IMG, IMG), Image.Resampling.BILINEAR)) if isinstance(image, (str, Path)) else np.asarray(image)
    probs = model.predict(img[None].astype("float32"), verbose=0)[0]; top = np.argsort(probs)[::-1][:top_k]
    if show:
        cam, cls, _ = gradcam(model, img)
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), gridspec_kw=dict(width_ratios=[1, 1, 2.4]))
        axes[0].imshow(img, interpolation="bilinear"); axes[0].set_title("input"); axes[0].axis("off")
        overlay(axes[1], img, cam); axes[1].set_title("Grad-CAM")
        axes[2].barh([short(c, 34) for c in top][::-1], probs[top][::-1], color=[GREEN if i == 0 else BLUE for i in range(top_k)][::-1]); axes[2].set(xlim=(0, 1), title=f"Top-{top_k} predictions"); axes[2].grid(axis="y", alpha=0)
        plt.tight_layout(); plt.show()
    return [(int(c), CLASS_NAMES[c], float(probs[c])) for c in top]

demo_ids = np.random.default_rng(11).choice(len(X_test), 4, replace=False)
for i in demo_ids:
    print(f"Ground truth: [{y_test[i]}] {CLASS_NAMES[y_test[i]]}"); predict_sign(X_test[i])
""")

code(r"""
# Optional: try your own photo (Colab only). Tick the box and rerun this cell.
RUN_UPLOAD = False  #@param {type:"boolean"}
if RUN_UPLOAD:
    try:
        from google.colab import files
        for name in files.upload():
            print("→", name); print(predict_sign(name))
    except ImportError:
        print("File upload is only available inside Google Colab.")
""")

# ───────────────────────────── 13. Export ─────────────────────────────
md(r"""
---
## 13 · Export and production-readiness checks

Training a model is not the same as shipping one. Before calling it done we verify:

| Check | Why |
|---|---|
| Save **all three trained models** in the native `.keras` format, with per-model metrics, predictions, confusion matrices and histories | Lets the team re-test and re-analyse without retraining |
| **Reload and compare predictions** to the in-memory model | Catches serialisation bugs and non-deterministic layers |
| **Accepts raw 0–255 pixels** | No separate preprocessing code to drift out of sync |
| **TFLite conversion + parity** | Proves it can run on mobile/embedded hardware, and shows the size/accuracy cost |
""")

code(r"""
# 13.1 Save every trained model plus its full results, then verify reloading
MODELS_DIR, RES_DIR = ART / "models", ART / "results"; MODELS_DIR.mkdir(exist_ok=True); RES_DIR.mkdir(exist_ok=True)
slug = lambda n: n.lower().replace(" (", "_").replace(")", "").replace(" ", "_")          # "MobileNetV2 (fine-tuned)" -> "mobilenetv2_fine-tuned"

# the head-only MobileNetV2 is rebuilt from the stage-1 snapshot, saved, then the fine-tuned weights are restored
mobilenet.load_weights(WORK / "mobilenet_stage1.weights.h5"); mobilenet.save(MODELS_DIR / f"{slug('MobileNetV2 (head only)')}.keras"); mobilenet.set_weights(mobilenet_final_weights)
cnn.save(MODELS_DIR / f"{slug('Custom CNN')}.keras"); mobilenet.save(MODELS_DIR / f"{slug('MobileNetV2 (fine-tuned)')}.keras")
HISTORIES = {"Custom CNN": hist_cnn, "MobileNetV2 (head only)": hist_head, "MobileNetV2 (fine-tuned)": hist_ft}
TRAIN_SECONDS = {"Custom CNN": time_cnn, "MobileNetV2 (head only)": time_head, "MobileNetV2 (fine-tuned)": time_head + time_ft}

for name in R:
    d = RES_DIR / slug(name); d.mkdir(exist_ok=True); r = R[name]
    rep_m = classification_report(y_test, r["pred"], labels=range(NUM_CLASSES), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    (d / "classification_report.json").write_text(json.dumps(rep_m, indent=2))
    np.save(d / "confusion_matrix.npy", confusion_matrix(y_test, r["pred"], labels=range(NUM_CLASSES)))
    (d / "history.json").write_text(json.dumps({k: [float(x) for x in v] for k, v in HISTORIES.get(name, {}).items()}, indent=2))
    top3 = np.argsort(r["probs"], axis=1)[:, -3:][:, ::-1]
    pd.DataFrame({"test_index": test_eval_df.index, "image_path": test_eval_df.path.map(lambda p: Path(p).name), "true_class": y_test, "true_name": [CLASS_NAMES[c] for c in y_test],
                  "pred_class": r["pred"], "pred_name": [CLASS_NAMES[c] for c in r["pred"]], "confidence": r["conf"].round(5), "correct": r["pred"] == y_test,
                  "top3_classes": [" ".join(map(str, t)) for t in top3]}).to_csv(d / "test_predictions.csv", index=False)
    (d / "metrics.json").write_text(json.dumps({"model": name, "mode": MODE, "validation_accuracy": float(V[name]["accuracy"]), "test_accuracy": float(r["accuracy"]), "test_macro_f1": float(r["macro_f1"]),
        "test_top3": float(r["top3"]), "ece": float(r["ece"]), "parameters": int(params[name]), "latency_ms_bs1": float(lat[name]), "training_seconds": float(TRAIN_SECONDS.get(name, TRAIN_SECONDS["Custom CNN"] + TRAIN_SECONDS["MobileNetV2 (fine-tuned)"])), "test_images": int(len(y_test))}, indent=2))

try:                                                         # the ensemble as one deployable Keras model
    _in = layers.Input((IMG, IMG, 3)); ens_model = keras.Model(_in, layers.Average()([cnn(_in), mobilenet(_in)]), name="ensemble"); ens_model.save(MODELS_DIR / "ensemble.keras")
    _back = keras.models.load_model(MODELS_DIR / "ensemble.keras"); _d = np.abs(_back.predict(X_test[:256], verbose=0) - ens_model.predict(X_test[:256], verbose=0)).max()
    print(f"Ensemble saved; reload max |dp| = {_d:.2e}")
except Exception as e:
    print("Ensemble model export skipped:", type(e).__name__, str(e)[:200])

(ART / "class_names.json").write_text(json.dumps(CLASS_NAMES, indent=2))
table.to_csv(ART / "model_comparison.csv"); per.to_csv(ART / "per_class_metrics_selected_model.csv", index=False); pairs.to_csv(ART / "top_confusions_selected_model.csv", index=False)
pd.DataFrame(rob, index=["mild", "medium", "severe"]).T.assign(clean=clean).to_csv(ART / "robustness_selected_model.csv")
sel = np.random.default_rng(SEED).choice(len(X_test), min(1000, len(X_test)), replace=False)
np.savez_compressed(ART / "sample_test_images.npz", X=X_test[sel], y=y_test[sel])               # 64x64 uint8 images for quick external testing
metrics = {"mode": MODE, "selected_model": BEST_NAME, "selection_criterion": "validation accuracy", "seed": SEED, "input": f"{IMG}x{IMG}x3 raw 0-255 pixels",
           "official_test": {"accuracy": float(RB["accuracy"]), "macro_f1": float(RB["macro_f1"]), "top3": float(RB["top3"]), "ece": float(RB["ece"]), "n_images": int(len(y_test))},
           "comparison": json.loads(table.to_json(orient="index")), "config": CFG, "tensorflow": tf.__version__, "keras": keras.__version__, "gpu": [g.name for g in gpus]}
(ART / "metrics.json").write_text(json.dumps(metrics, indent=2))

MODEL_PATH = MODELS_DIR / f"{slug(BEST_NAME)}.keras"; shutil.copy(MODEL_PATH, ART / "best_model.keras")
for name, m in {"Custom CNN": cnn, "MobileNetV2 (fine-tuned)": mobilenet}.items():
    back = keras.models.load_model(MODELS_DIR / f"{slug(name)}.keras"); a, b = m.predict(X_test[:256], verbose=0), back.predict(X_test[:256], verbose=0)
    assert np.abs(a - b).max() < 1e-4 and (a.argmax(1) == b.argmax(1)).all(), f"Reloaded {name} disagrees with the original!"
    print(f"Reload parity OK - {name}: max |dp| = {np.abs(a - b).max():.2e} | {(MODELS_DIR / f'{slug(name)}.keras').stat().st_size / 1e6:.1f} MB")
""")

code(r"""
# 13.2 TFLite conversion (float16-weights) and parity check
try:
    conv = tf.lite.TFLiteConverter.from_keras_model(BEST); conv.optimizations = [tf.lite.Optimize.DEFAULT]; conv.target_spec.supported_types = [tf.float16]
    tflite = conv.convert(); (ART / "best_model.tflite").write_bytes(tflite)
    interp = tf.lite.Interpreter(model_content=tflite); interp.allocate_tensors(); inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    nchk = min(500, len(X_test)); preds = []; t0 = time.time()
    for x in X_test[:nchk]:
        interp.set_tensor(inp["index"], x[None].astype(inp["dtype"])); interp.invoke(); preds.append(interp.get_tensor(out["index"]).argmax())
    ms = (time.time() - t0) / nchk * 1000
    agree = np.mean(np.array(preds) == RB["pred"][:nchk]); tacc = np.mean(np.array(preds) == y_test[:nchk])
    print(f"TFLite: {len(tflite) / 1e6:.2f} MB (Keras file {MODEL_PATH.stat().st_size / 1e6:.1f} MB) | agrees with Keras on {agree:.1%} of {nchk} images | accuracy {tacc:.1%} | {ms:.2f} ms/img (single-threaded CPU)")
except Exception as e:
    print("TFLite conversion skipped:", type(e).__name__, str(e)[:200])
""")

code(r"""
# 13.3 Bundle models + results + figures + a README, and download it
(ART / "README.md").write_text(f'''# GTSRB demo export (mode: {MODE})

models/    trained Keras models (custom_cnn, mobilenetv2_head_only, mobilenetv2_fine-tuned, ensemble); best_model.keras = {BEST_NAME}
           (+ best_model.tflite when conversion succeeded)
results/   per model: metrics.json, classification_report.json, confusion_matrix.npy, history.json, test_predictions.csv
figures/   every figure produced by the notebook (PNG)
*.csv/json comparison table, progress_ladder.csv, per-class metrics, top confusions, robustness, class names, overall metrics
sample_test_images.npz   1000 official-test images (64x64 uint8, keys X and y) for quick external testing

## Load and predict
    import numpy as np, json, tensorflow as tf
    model = tf.keras.models.load_model("models/custom_cnn.keras")      # accepts raw 0-255 pixels, shape (N, 64, 64, 3)
    names = json.load(open("class_names.json")); d = np.load("sample_test_images.npz")
    pred = model.predict(d["X"].astype("float32")).argmax(1); print((pred == d["y"]).mean(), names[pred[0]])
''')
bundle = shutil.make_archive(str(WORK / "gtsrb_demo_results"), "zip", ART)
print("Bundle:", bundle, f"({Path(bundle).stat().st_size / 1e6:.1f} MB)"); print("Contents:", *sorted(str(p.relative_to(ART)) for p in ART.rglob("*") if p.is_file() and p.suffix != ".png"), sep="\n  ")
DOWNLOAD = True  #@param {type:"boolean"}
if DOWNLOAD:
    try:
        from google.colab import files; files.download(bundle)
    except ImportError: print("Download is only available inside Google Colab; the bundle is at", bundle)
""")

# ───────────────────────────── 14. Wrap-up ─────────────────────────────
md(r"""
---
## 14 · Wrap-up

### Results at a glance
The comparison table in §9 and the selected-model summary are the numbers for **this run** (re-generated every time the notebook executes).
""")

code(r"""
print(f"MODE: {MODE}   |   Selected: {BEST_NAME}")
print(f"Official test → accuracy {RB['accuracy']:.2%} · macro-F1 {RB['macro_f1']:.2%} · top-3 {RB['top3']:.2%} · ECE {RB['ece']:.3f}")
print(f"Weakest class: [{per.nsmallest(1, 'f1')['class'].iloc[0]}] {per.nsmallest(1, 'f1')['name'].iloc[0]}  (F1 {per.f1.min():.2f})")
print(f"MobileNetV2: head-only {R['MobileNetV2 (head only)']['accuracy']:.2%} → fine-tuned {R['MobileNetV2 (fine-tuned)']['accuracy']:.2%}  (value of fine-tuning: {R['MobileNetV2 (fine-tuned)']['accuracy'] - R['MobileNetV2 (head only)']['accuracy']:+.2%})")
table
""")

md(r"""
### Decision log

| # | Decision | Alternative considered | Why this one |
|---|---|---|---|
| 1 | Verify MD5 of the dataset archive | Trust the download | Silent corruption would invalidate everything downstream |
| 2 | Official train/test split, test used once | Custom re-split | Comparable with published results; no test-set tuning |
| 3 | **Grouped (per-track) validation split** | Random image split | Frames of one sign are near-duplicates; random split leaks (§5) |
| 4 | Report **macro-F1** + accuracy | Accuracy only | ≈10× class imbalance (§4.1) |
| 5 | 64×64 cached `uint8` arrays | Full-resolution streaming | Median sign is tiny; removes I/O bottleneck |
| 6 | No horizontal-flip augmentation | Flip augmentation | Would corrupt left/right-meaning classes |
| 7 | Normalisation + augmentation **inside** the model | External preprocessing | No train/serve skew; one deployable artefact |
| 8 | Small custom CNN **and** MobileNetV2 | One model | Shows the accuracy/cost trade-off |
| 9 | Two-stage transfer learning, BN frozen | Frozen backbone only | Closes the ImageNet→sign domain gap (§8) |
| 10 | Select the model on validation accuracy | Pick best on test | Avoids optimistic bias |
| 11 | Grad-CAM + t-SNE | Metrics only | Verify the model uses the sign itself |
| 12 | Confidence abstention | Always answer | Safer behaviour on blurry/dark/occluded input |
| 13 | Reload-parity + TFLite checks | Assume export works | Catches deployment bugs early |

### Limitations
- **Cropped signs only.** Real driving needs a *detector* to find signs in a full frame first (the repo has an optional YOLO extension); end-to-end performance is bounded by that detector.
- **One country, one dataset.** GTSRB is German; other regions use different sign designs. Domain shift is untested.
- **Single seed.** Results are from one training run; a production claim would average several seeds and report confidence intervals.
- **Not safety-certified.** This is a research/educational pipeline; it has not been validated against automotive safety standards (e.g. ISO 26262 / SOTIF).
- **Synthetic robustness tests** approximate, but do not replace, field data (night rain, snow, motion blur at speed).
- **Class weights / focal loss** were not explored for the rarest classes.

### Future work
1. Detection front-end (YOLO) → full-scene recognition and video.
2. Temperature scaling / conformal prediction for reliable confidence.
3. Resolution study (64 vs 96 vs 128) specifically for the speed-limit family.
4. Multi-seed experiments, k-fold over tracks, statistical significance.
5. Out-of-distribution detection (e.g. non-sign objects, unseen countries).
6. Quantisation-aware training and on-device latency benchmarks.

### References
1. J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel, “The German Traffic Sign Recognition Benchmark: A multi-class classification competition,” *IJCNN*, 2011.
2. M. Sandler et al., “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” *CVPR*, 2018.
3. R. R. Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,” *ICCV*, 2017.
4. D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” *ICLR*, 2015.
5. C. Guo et al., “On Calibration of Modern Neural Networks,” *ICML*, 2017.
6. scikit-learn `GroupShuffleSplit` (and the per-class track holdout used here); TensorFlow/Keras documentation (callbacks, preprocessing layers, TFLite).
""")

# ───────────────────────────── write ─────────────────────────────
for i, c in enumerate(cells):
    c["id"] = f"cell-{i:03d}"
    if isinstance(c["source"], str):
        c["source"] = c["source"].splitlines(keepends=True)

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "GTSRB Traffic-Sign Recognition — Panel Demo", "provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"Wrote {OUT.relative_to(ROOT)} with {len(cells)} cells")
