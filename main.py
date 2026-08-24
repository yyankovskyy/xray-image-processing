#!/usr/bin/env python
"""
Chest X-Ray Multi-Label Diagnosis — CLI pipeline entrypoint.

Usage examples
--------------
  python main.py check-leakage --config config.yaml
  python main.py train          --config config.yaml
  python main.py predict        --config config.yaml
  python main.py evaluate       --config config.yaml
  python main.py gradcam        --config config.yaml
  python main.py run-all        --config config.yaml

Every parameter (paths, image size, batch size, training schedule, labels,
Grad-CAM settings, ...) is read from the YAML file passed via --config —
see config.yaml at the repo root for the full set of options.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline_config import ensure_dir, load_config, setup_logging
from data_utils import build_all_generators, load_dataframes, run_leakage_checks
from evaluate import get_roc_curve, predict, save_auc_scores, save_predictions
from gradcam import compute_gradcam
from losses import compute_class_freqs
from model_build import build_model, load_pretrained_weights

logger = logging.getLogger(__name__)


def cmd_check_leakage(cfg):
    train_df, valid_df, test_df = load_dataframes(cfg)
    results = run_leakage_checks(train_df, valid_df, test_df, cfg["data"]["patient_col"])
    if any(results.values()):
        logger.warning("Leakage detected between splits: %s", [k for k, v in results.items() if v])
    else:
        logger.info("No patient-level leakage detected between any splits.")
    return results


def _build_generators_and_freqs(cfg):
    train_df, valid_df, test_df = load_dataframes(cfg)
    train_generator, valid_generator, test_generator = build_all_generators(cfg, train_df, valid_df, test_df)
    pos_weights, neg_weights = compute_class_freqs(train_generator.labels)
    return train_df, valid_df, test_df, train_generator, valid_generator, test_generator, pos_weights, neg_weights


def cmd_train(cfg):
    (_, _, _, train_generator, valid_generator, test_generator,
     pos_weights, neg_weights) = _build_generators_and_freqs(cfg)

    model = build_model(
        labels=cfg["labels"], pos_weights=pos_weights, neg_weights=neg_weights,
        base_weights=cfg["model"]["base_weights"], optimizer=cfg["model"]["optimizer"],
        loss_epsilon=float(cfg["model"]["loss_epsilon"]),
    )

    train_cfg = cfg["training"]
    logger.info("Starting training: epochs=%d, steps_per_epoch=%d, validation_steps=%d",
                train_cfg["epochs"], train_cfg["steps_per_epoch"], train_cfg["validation_steps"])

    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        steps_per_epoch=train_cfg["steps_per_epoch"],
        validation_steps=train_cfg["validation_steps"],
        epochs=train_cfg["epochs"],
    )

    ensure_dir(os.path.dirname(train_cfg["output_weights"]) or ".")
    model.save_weights(train_cfg["output_weights"])
    logger.info("Saved trained weights to %s", train_cfg["output_weights"])

    if train_cfg.get("history_plot"):
        ensure_dir(os.path.dirname(train_cfg["history_plot"]) or ".")
        plt.figure()
        plt.plot(history.history["loss"], label="train_loss")
        if "val_loss" in history.history:
            plt.plot(history.history["val_loss"], label="val_loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.savefig(train_cfg["history_plot"], bbox_inches="tight", dpi=150)
        plt.close()
        logger.info("Saved training loss plot to %s", train_cfg["history_plot"])

    return model


def _load_model_for_inference(cfg, pos_weights, neg_weights):
    model = build_model(
        labels=cfg["labels"], pos_weights=pos_weights, neg_weights=neg_weights,
        base_weights=cfg["model"]["base_weights"], optimizer=cfg["model"]["optimizer"],
        loss_epsilon=float(cfg["model"]["loss_epsilon"]),
    )
    weights_path = cfg["model"].get("pretrained_weights")
    if weights_path:
        load_pretrained_weights(model, weights_path)
    else:
        logger.warning("No 'model.pretrained_weights' set in config — using freshly "
                        "initialized/backbone-only weights.")
    return model


def cmd_predict(cfg):
    (_, _, test_df, train_generator, _, test_generator,
     pos_weights, neg_weights) = _build_generators_and_freqs(cfg)

    model = _load_model_for_inference(cfg, pos_weights, neg_weights)
    predicted_vals = predict(model, test_generator)

    eval_cfg = cfg["evaluation"]
    out_dir = ensure_dir(eval_cfg["output_dir"])
    save_predictions(
        predicted_vals, cfg["labels"], test_df, cfg["data"]["image_col"],
        os.path.join(out_dir, eval_cfg["predictions_csv"]),
    )
    return model, predicted_vals, test_generator


def cmd_evaluate(cfg):
    model, predicted_vals, test_generator = cmd_predict(cfg)

    eval_cfg = cfg["evaluation"]
    out_dir = ensure_dir(eval_cfg["output_dir"])
    roc_path = os.path.join(out_dir, eval_cfg["roc_plot_filename"])

    auc_vals = get_roc_curve(cfg["labels"], predicted_vals, test_generator, output_path=roc_path)
    save_auc_scores(cfg["labels"], auc_vals, os.path.join(out_dir, eval_cfg["auc_csv"]))

    for label, auc in zip(cfg["labels"], auc_vals):
        logger.info("AUROC %-20s %.3f" % (label, auc) if auc == auc else "AUROC %-20s n/a" % label)

    return auc_vals


def cmd_gradcam(cfg):
    train_df, _, test_df, train_generator, _, test_generator, pos_weights, neg_weights = _build_generators_and_freqs(cfg)
    model = _load_model_for_inference(cfg, pos_weights, neg_weights)

    # Need per-label AUROC to pick the top-performing labels, same as the notebook.
    predicted_vals = predict(model, test_generator)
    auc_vals = get_roc_curve(cfg["labels"], predicted_vals, test_generator, output_path=None)
    auc_vals = [a if a == a else -1 for a in auc_vals]  # NaN -> sorts last

    gc_cfg = cfg["gradcam"]
    top_n = gc_cfg["num_top_labels"]
    labels_to_show = list(np.take(cfg["labels"], np.argsort(auc_vals)[::-1])[:top_n])
    logger.info("Grad-CAM will visualize top-%d labels by AUROC: %s", top_n, labels_to_show)

    images = gc_cfg.get("images") or []
    if not images:
        n_random = gc_cfg.get("num_random_images", 4)
        images = list(train_df[cfg["data"]["image_col"]].sample(n_random, random_state=cfg["image"]["seed"]))
        logger.info("No explicit gradcam.images set — sampled %d random training images.", n_random)

    out_dir = ensure_dir(gc_cfg["output_dir"])
    for img in images:
        out_path = os.path.join(out_dir, f"gradcam_{os.path.splitext(img)[0]}.png")
        compute_gradcam(
            model, img, cfg["data"]["image_dir"], train_df, cfg["labels"], labels_to_show,
            image_col=cfg["data"]["image_col"], layer_name=gc_cfg["layer_name"],
            h=cfg["image"]["target_h"], w=cfg["image"]["target_w"],
            sample_size=cfg["image"]["sample_size"], output_path=out_path,
        )

    return images, labels_to_show


def cmd_run_all(cfg):
    cmd_check_leakage(cfg)
    if cfg["training"].get("enabled", False):
        cmd_train(cfg)
    cmd_evaluate(cfg)
    cmd_gradcam(cfg)


COMMANDS = {
    "check-leakage": cmd_check_leakage,
    "train": cmd_train,
    "predict": cmd_predict,
    "evaluate": cmd_evaluate,
    "gradcam": cmd_gradcam,
    "run-all": cmd_run_all,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Chest X-Ray multi-label diagnosis pipeline (DenseNet121 + Grad-CAM)."
    )
    parser.add_argument("command", choices=list(COMMANDS.keys()),
                         help="Pipeline stage to run.")
    parser.add_argument("--config", default="config.yaml",
                         help="Path to the pipeline YAML config (default: config.yaml).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg)

    logger.info("Running command '%s' with config '%s'", args.command, args.config)
    COMMANDS[args.command](cfg)
    logger.info("Done.")


if __name__ == "__main__":
    sys.exit(main())
