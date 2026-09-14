"""Grad-CAM visualization utilities.

Direct port of util.py's `get_mean_std_per_batch`, `load_image`, `grad_cam`,
and `compute_gradcam`, adapted to save figures to disk (for headless CLI
runs) and to be driven by config values instead of hard-coded literals.
"""
from __future__ import annotations

import logging
import os
import random
from typing import List

import cv2
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for CLI runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.compat.v1.logging import INFO, set_verbosity

random.seed(a=None, version=2)
set_verbosity(INFO)

logger = logging.getLogger(__name__)


def get_mean_std_per_batch(image_dir: str, df: pd.DataFrame, image_col: str = "Image",
                            h: int = 320, w: int = 320, sample_size: int = 100):
    """Estimate per-channel mean/std from a random sample of images."""
    n = min(sample_size, len(df))
    sample_data = []
    for img in df.sample(n)[image_col].values:
        image_path = os.path.join(image_dir, img)
        sample_data.append(np.array(image.load_img(image_path, target_size=(h, w))))

    mean = np.mean(sample_data, axis=(0, 1, 2, 3))
    std = np.std(sample_data, axis=(0, 1, 2, 3), ddof=1)
    return mean, std


def load_image(img: str, image_dir: str, df: pd.DataFrame, image_col: str = "Image",
                preprocess: bool = True, h: int = 320, w: int = 320, sample_size: int = 100):
    """Load and (optionally) mean/std-normalize a single image for the model."""
    mean, std = get_mean_std_per_batch(image_dir, df, image_col=image_col, h=h, w=w,
                                        sample_size=sample_size)
    img_path = os.path.join(image_dir, img)
    x = image.load_img(img_path, target_size=(h, w))
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x


def grad_cam(input_model, img_array, cls: int, layer_name: str, h: int = 320, w: int = 320):
    """GradCAM method for visualizing input saliency for a given class index.

    Uses `tf.GradientTape` rather than the TF1-era `keras.backend.K.gradients`
    / `K.function` — those relied on static-graph mode and no longer exist
    under Keras 3 (TF>=2.16). GradientTape works identically under Keras 2
    and Keras 3, so this is safe on any currently-installable TensorFlow.
    """
    grad_model = tf.keras.models.Model(
        inputs=input_model.inputs,
        outputs=[input_model.get_layer(layer_name).output, input_model.output],
    )

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_tensor)
        y_c = predictions[:, cls]

    grads_val = tape.gradient(y_c, conv_output)

    output = conv_output[0].numpy()
    grads_val = grads_val[0].numpy()

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    cam = cv2.resize(cam, (w, h), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam


def compute_gradcam(model, img: str, image_dir: str, df: pd.DataFrame, labels: List[str],
                     selected_labels: List[str], image_col: str = "Image",
                     layer_name: str = "bn", h: int = 320, w: int = 320,
                     sample_size: int = 100, output_path: str = None) -> None:
    """Generate and save a Grad-CAM panel (original + one heatmap per selected label)."""
    preprocessed_input = load_image(img, image_dir, df, image_col=image_col, h=h, w=w,
                                     sample_size=sample_size)
    predictions = model.predict(preprocessed_input)

    logger.info("Loading original image: %s", img)
    plt.figure(figsize=(15, 10))
    plt.subplot(151)
    plt.title("Original")
    plt.axis("off")
    plt.imshow(load_image(img, image_dir, df, image_col=image_col, preprocess=False, h=h, w=w,
                           sample_size=sample_size), cmap="gray")

    j = 1
    for i in range(len(labels)):
        if labels[i] in selected_labels:
            logger.info("Generating gradcam for class %s", labels[i])
            gradcam = grad_cam(model, preprocessed_input, i, layer_name, h=h, w=w)
            plt.subplot(151 + j)
            plt.title(f"{labels[i]}: p={predictions[0][i]:.3f}")
            plt.axis("off")
            plt.imshow(load_image(img, image_dir, df, image_col=image_col, preprocess=False,
                                   h=h, w=w, sample_size=sample_size), cmap="gray")
            plt.imshow(gradcam, cmap="jet", alpha=min(0.5, predictions[0][i]))
            j += 1

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        logger.info("Saved gradcam panel to %s", output_path)
    plt.close()
