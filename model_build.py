"""DenseNet121-based multi-label classification model."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import numpy as np
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

from losses import get_weighted_loss

logger = logging.getLogger(__name__)


def build_model(labels: List[str], pos_weights: np.ndarray, neg_weights: np.ndarray,
                 base_weights: str = "imagenet", optimizer: str = "adam",
                 loss_epsilon: float = 1e-7) -> Model:
    """Build and compile the DenseNet121 transfer-learning model.

    Args:
        labels: list of pathology names (defines the output layer width/order).
        pos_weights: per-class positive-term loss weights.
        neg_weights: per-class negative-term loss weights.
        base_weights: "imagenet", a path to local backbone weights, or None
            for random initialization.
        optimizer: Keras optimizer name passed to `model.compile`.
        loss_epsilon: numerical-stability constant for the weighted loss.

    Returns:
        A compiled Keras `Model`.
    """
    weights_arg = None if base_weights in (None, "none", "None") else base_weights
    if weights_arg not in (None, "imagenet") and not os.path.exists(weights_arg):
        raise FileNotFoundError(f"DenseNet121 base_weights file not found: {weights_arg}")

    logger.info("Building DenseNet121 backbone (weights=%s)", weights_arg)
    base_model = DenseNet121(weights=weights_arg, include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(len(labels), activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=optimizer, loss=get_weighted_loss(pos_weights, neg_weights, loss_epsilon))
    logger.info("Model built and compiled with %d output classes.", len(labels))
    return model


def load_pretrained_weights(model: Model, weights_path: str) -> Model:
    """Load a full model checkpoint (backbone + head) into `model` in place."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Pretrained weights file not found: {weights_path}")
    logger.info("Loading pretrained weights from %s", weights_path)
    model.load_weights(weights_path)
    return model
