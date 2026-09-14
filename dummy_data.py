"""Synthetic dataset generator for smoke-testing the pipeline.

Generates a tiny, schema-valid train/valid/test dataset — random-noise
images plus CSVs with the expected `Image` / `PatientId` / label columns —
so the full pipeline (leakage check, generators, training, prediction,
evaluation, Grad-CAM) can be exercised end to end without the real
ChestX-ray8 data or a GPU.

Patient IDs are drawn from disjoint per-split ranges (train: P0xxx,
valid: P1xxx, test: P2xxx), so `check-leakage` is guaranteed to report a
clean run — the same invariant the real dataset is expected to satisfy.

This is a **smoke test**, not a benchmark: the images are random noise, so
metrics like AUROC on this data are meaningless. It only proves the wiring
(config -> data -> model -> loss -> training -> evaluation -> Grad-CAM)
runs without error using minimal CPU/RAM/disk and no network access.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _make_split(image_dir: str, csv_path: str, labels: List[str], image_col: str,
                 patient_col: str, prefix: str, n_patients: int, images_per_patient: int,
                 w: int, h: int, pos_rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Generate one split's images + rows, and write its CSV."""
    rows = []
    for p in range(n_patients):
        patient_id = f"{prefix}{p:04d}"
        for i in range(images_per_patient):
            fname = f"{prefix}{p:04d}_{i}.png"
            arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(image_dir, fname), arr)
            row = {image_col: fname, patient_col: patient_id}
            for label in labels:
                row[label] = int(rng.random() < pos_rate)
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %d dummy rows / %d images to %s", len(df), len(df), csv_path)
    return df


def generate_dummy_dataset(cfg: Dict[str, Any], n_train_patients: int = 6,
                            n_valid_patients: int = 2, n_test_patients: int = 2,
                            images_per_patient: int = 2, pos_rate: float = 0.3,
                            seed: int = None) -> Dict[str, int]:
    """Create a synthetic train/valid/test dataset matching `cfg`'s schema.

    Uses `cfg["data"]` for paths/column names, `cfg["image"]` for target
    size, and `cfg["labels"]` for the label columns. Safe to call repeatedly
    — it overwrites the same dummy files each time (fixed seed by default).

    Returns a dict with the row count written per split.
    """
    data_cfg = cfg["data"]
    image_cfg = cfg["image"]
    labels = cfg["labels"]

    rng = np.random.default_rng(seed if seed is not None else image_cfg.get("seed", 1))
    w, h = image_cfg["target_w"], image_cfg["target_h"]
    image_dir = data_cfg["image_dir"]
    os.makedirs(image_dir, exist_ok=True)

    logger.info(
        "Generating dummy dataset: %d train / %d valid / %d test patients, "
        "%d images/patient, %dx%d images, in %s",
        n_train_patients, n_valid_patients, n_test_patients, images_per_patient, w, h, image_dir,
    )

    train_df = _make_split(image_dir, data_cfg["train_csv"], labels, data_cfg["image_col"],
                            data_cfg["patient_col"], "P0", n_train_patients, images_per_patient,
                            w, h, pos_rate, rng)
    valid_df = _make_split(image_dir, data_cfg["valid_csv"], labels, data_cfg["image_col"],
                            data_cfg["patient_col"], "P1", n_valid_patients, images_per_patient,
                            w, h, pos_rate, rng)
    test_df = _make_split(image_dir, data_cfg["test_csv"], labels, data_cfg["image_col"],
                           data_cfg["patient_col"], "P2", n_test_patients, images_per_patient,
                           w, h, pos_rate, rng)

    counts = {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)}
    logger.info("Dummy dataset ready: %s", counts)
    return counts
