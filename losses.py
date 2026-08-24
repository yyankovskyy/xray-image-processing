"""Class-imbalance handling: frequency computation and weighted loss.

Ports the assignment's Exercise 2 (`compute_class_freqs`) and Exercise 3
(`get_weighted_loss`).
"""
from __future__ import annotations

import logging
from typing import Callable, Tuple

import numpy as np
from keras import backend as K

logger = logging.getLogger(__name__)


def compute_class_freqs(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute positive and negative frequencies for each class.

    Args:
        labels: array of shape (num_examples, num_classes).

    Returns:
        (positive_frequencies, negative_frequencies), each of shape (num_classes,).
    """
    n = labels.shape[0]
    positive_frequencies = np.sum(labels, axis=0) / n
    negative_frequencies = 1 - positive_frequencies
    return positive_frequencies, negative_frequencies


def get_weighted_loss(pos_weights: np.ndarray, neg_weights: np.ndarray,
                       epsilon: float = 1e-7) -> Callable:
    """Build a weighted binary cross-entropy loss function for multi-label output.

    Args:
        pos_weights: per-class weight applied to positive-label terms
            (conventionally set to the class's negative frequency, so rare
            positive classes get up-weighted).
        neg_weights: per-class weight applied to negative-label terms.
        epsilon: small constant added inside log() calls for numerical stability.

    Returns:
        A `weighted_loss(y_true, y_pred)` function suitable for `model.compile(loss=...)`.
    """

    def weighted_loss(y_true, y_pred):
        loss = 0.0
        for i in range(len(pos_weights)):
            loss += K.mean(
                -(
                    pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon)
                    + neg_weights[i] * (1.0 - y_true[:, i]) * K.log(1.0 - y_pred[:, i] + epsilon)
                )
            )
        return loss

    return weighted_loss
