"""Prediction and ROC/AUROC evaluation.

`get_roc_curve` is a direct port of the assignment's `util.get_roc_curve`,
adapted to save the figure to disk (for headless CLI runs) instead of only
calling `plt.show()`, and to return per-label AUCs alongside the values.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for CLI runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


def predict(model, test_generator, steps: int = None) -> np.ndarray:
    """Run model.predict over the test generator."""
    if steps is None:
        steps = len(test_generator)
    logger.info("Running predictions on test set (%d steps)...", steps)
    predicted_vals = model.predict(test_generator, steps=steps)
    return predicted_vals


def get_roc_curve(labels: List[str], predicted_vals: np.ndarray, generator,
                   output_path: str = None) -> List[float]:
    """Compute per-label AUROC and plot all ROC curves on one figure.

    Args:
        labels: list of pathology names, in the same order as model outputs.
        predicted_vals: array of shape (num_examples, num_labels) of predicted probabilities.
        generator: the Keras test generator (used for its `.labels` ground truth).
        output_path: if given, save the combined ROC plot to this path instead
            of (or in addition to) displaying it.

    Returns:
        List of AUROC values, one per label (in label order). Labels for which
        AUROC could not be computed (e.g. too few positive examples) are
        recorded as `float('nan')`.
    """
    auc_roc_vals = []
    plt.figure(1, figsize=(10, 10))
    plt.plot([0, 1], [0, 1], "k--")

    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.plot(fpr_rf, tpr_rf, label=f"{labels[i]} ({round(auc_roc, 3)})")
        except Exception:
            logger.warning("Could not compute ROC for %s (dataset lacks enough examples).", labels[i])
            auc_roc_vals.append(float("nan"))

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend(loc="best")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        logger.info("Saved ROC plot to %s", output_path)
    plt.close()

    return auc_roc_vals


def save_predictions(predicted_vals: np.ndarray, labels: List[str], test_df: pd.DataFrame,
                      image_col: str, output_csv: str) -> None:
    """Write a CSV of predicted probabilities alongside image filenames."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df = pd.DataFrame(predicted_vals, columns=labels)
    out_df.insert(0, image_col, test_df[image_col].values[: len(out_df)])
    out_df.to_csv(output_csv, index=False)
    logger.info("Saved predictions to %s", output_csv)


def save_auc_scores(labels: List[str], auc_roc_vals: List[float], output_csv: str) -> None:
    """Write a CSV of per-label AUROC scores."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame({"label": labels, "auc_roc": auc_roc_vals}).to_csv(output_csv, index=False)
    logger.info("Saved AUC scores to %s", output_csv)
