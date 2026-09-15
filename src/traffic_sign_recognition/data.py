"""Dataset discovery, loading, and tf.data construction for GTSRB."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

NUM_CLASSES = 43


def _find_root(root: Path) -> Path:
    candidates = [root, root / "GTSRB"]
    for candidate in candidates:
        if (candidate / "Train").is_dir() and (candidate / "Test").is_dir():
            return candidate
        if (candidate / "Final_Training" / "Images").is_dir():
            return candidate
    for candidate in root.rglob("Final_Training/Images"):
        return candidate.parent.parent
    raise FileNotFoundError("Could not find GTSRB/Final_Training/Images under " + str(root))


def _group_for(path: Path) -> str:
    # GTSRB filenames may be class_track_frame.png or track_frame.ppm.
    parts = path.stem.split("_")
    stem = parts[1] if len(parts) > 2 else parts[0]
    return f"{path.parent.name}_{stem}"


def load_labeled_data(root: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return train/test dataframes with absolute paths and integer labels."""
    root = _find_root(Path(root))
    if (root / "Train").is_dir():
        train_dir = root / "Train"
        train_rows = []
        for class_dir in sorted(train_dir.iterdir()):
            if not class_dir.is_dir() or not class_dir.name.isdigit():
                continue
            label = int(class_dir.name)
            for image_path in sorted(class_dir.glob("*")):
                if image_path.suffix.lower() in {".png", ".ppm", ".jpg", ".jpeg"}:
                    train_rows.append({"path": str(image_path), "label": label, "group": _group_for(image_path)})
        train = pd.DataFrame(train_rows)
        label_file = root / "Test.csv"
        labels = pd.read_csv(label_file)
        test_rows = [{"path": str(root / "Test" / Path(str(row["Path"])).name), "label": int(row["ClassId"]), "group": _group_for(Path(Path(str(row["Path"])).name))} for _, row in labels.iterrows()]
        test = pd.DataFrame(test_rows)
        if train.empty or set(train["label"].unique()) != set(range(NUM_CLASSES)):
            raise ValueError("Training data must contain all 43 GTSRB class directories")
        if test.empty or not test["path"].map(Path).map(Path.exists).all():
            raise ValueError("Some GTSRB test images referenced by Test.csv are missing")
        return train, test

    train_dir = root / "Final_Training" / "Images"
    train_rows = []
    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir() or not class_dir.name.isdigit():
            continue
        label = int(class_dir.name)
        for image_path in sorted(class_dir.glob("*.ppm")):
            train_rows.append({"path": str(image_path), "label": label, "group": _group_for(image_path)})
    train = pd.DataFrame(train_rows)
    if train.empty or set(train["label"].unique()) != set(range(NUM_CLASSES)):
        raise ValueError("Training data must contain all 43 GTSRB class directories")

    test_dir = root / "Final_Test" / "Images"
    label_file = root / "Final_Test" / "GT-final_test.csv"
    if not label_file.exists():
        matches = list(root.rglob("GT-final_test.csv"))
        if matches:
            label_file = matches[0]
    if not test_dir.is_dir() or not label_file.exists():
        raise FileNotFoundError("GTSRB official test images or GT-final_test.csv is missing")
    labels = pd.read_csv(label_file, sep=";")
    path_col = "Filename" if "Filename" in labels.columns else "Path"
    class_col = "ClassId" if "ClassId" in labels.columns else "ClassId"
    test_rows = []
    for _, row in labels.iterrows():
        image_path = test_dir / str(row[path_col])
        test_rows.append({"path": str(image_path), "label": int(row[class_col]), "group": _group_for(image_path)})
    test = pd.DataFrame(test_rows)
    if test.empty or not test["path"].map(Path).map(Path.exists).all():
        raise ValueError("Some official GTSRB test images referenced by the CSV are missing")
    return train, test


def split_train_validation(train: pd.DataFrame, validation_size: float = 0.15, seed: int = 42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(train, train["label"], groups=train["group"]))
    return train.iloc[train_idx].reset_index(drop=True), train.iloc[val_idx].reset_index(drop=True)


def make_dataset(frame: pd.DataFrame, image_size: int = 64, batch_size: int = 64, shuffle: bool = False, seed: int = 42):
    paths = frame["path"].to_numpy(dtype=str)
    labels = frame["label"].to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [image_size, image_size])
        return tf.cast(image, tf.float32) / 255.0, label

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(min(len(frame), 10000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
