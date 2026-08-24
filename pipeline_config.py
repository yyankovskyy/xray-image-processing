"""Configuration loading utilities.

All pipeline parameters live in a single YAML file (see config.yaml at the
repo root). This module loads that file into a plain nested dict and does a
light validation pass so failures surface early with a clear message instead
of a cryptic KeyError three modules deep.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

REQUIRED_TOP_LEVEL_KEYS = ["data", "labels", "image", "model", "training", "evaluation", "gradcam"]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and lightly validate the pipeline YAML config.

    Args:
        config_path: path to a YAML file.

    Returns:
        Nested dict of configuration values.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Config is missing required top-level section(s): {missing}")

    if not isinstance(cfg["labels"], list) or len(cfg["labels"]) == 0:
        raise ValueError("Config 'labels' must be a non-empty list of pathology names.")

    return cfg


def setup_logging(cfg: Dict[str, Any]) -> None:
    """Configure root logging based on config['logging']['level']."""
    level_name = cfg.get("logging", {}).get("level", "INFO")
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_dir(path: str) -> str:
    """Create a directory (and parents) if it doesn't exist; return the path."""
    os.makedirs(path, exist_ok=True)
    return path
