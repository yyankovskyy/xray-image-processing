"""Data loading, patient-level leakage checks, and image generators.

Ports the assignment's Exercise 1 (`check_for_leakage`) plus the generator
factory functions, parameterized entirely from the config dict instead of
hard-coded literals.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

logger = logging.getLogger(__name__)


def load_dataframes(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the train/valid/test CSVs referenced in the config."""
    data_cfg = cfg["data"]
    train_df = pd.read_csv(data_cfg["train_csv"])
    valid_df = pd.read_csv(data_cfg["valid_csv"])
    test_df = pd.read_csv(data_cfg["test_csv"])
    logger.info(
        "Loaded dataframes: train=%d, valid=%d, test=%d rows",
        len(train_df), len(valid_df), len(test_df),
    )
    return train_df, valid_df, test_df


def check_for_leakage(df1: pd.DataFrame, df2: pd.DataFrame, patient_col: str) -> bool:
    """Return True if any patient ID appears in both df1 and df2.

    Args:
        df1: first dataframe.
        df2: second dataframe.
        patient_col: name of the column holding patient IDs.

    Returns:
        True if there is patient overlap between the two dataframes.
    """
    df1_patients_unique = set(df1[patient_col].values)
    df2_patients_unique = set(df2[patient_col].values)

    patients_in_both_groups = df1_patients_unique.intersection(df2_patients_unique)
    leakage = len(patients_in_both_groups) > 0
    return leakage


def run_leakage_checks(train_df: pd.DataFrame, valid_df: pd.DataFrame,
                        test_df: pd.DataFrame, patient_col: str) -> Dict[str, bool]:
    """Run and log the three standard pairwise leakage checks."""
    results = {
        "train_valid": check_for_leakage(train_df, valid_df, patient_col),
        "train_test": check_for_leakage(train_df, test_df, patient_col),
        "valid_test": check_for_leakage(valid_df, test_df, patient_col),
    }
    for pair, leaked in results.items():
        level = logging.WARNING if leaked else logging.INFO
        logger.log(level, "Leakage check [%s]: %s", pair, leaked)
    return results


def get_train_generator(df: pd.DataFrame, image_dir: str, x_col: str, y_cols: List[str],
                         shuffle: bool = True, batch_size: int = 8, seed: int = 1,
                         target_w: int = 320, target_h: int = 320):
    """Generator for the training set, normalized using per-batch statistics."""
    logger.info("Building train generator (batch_size=%d, target=%dx%d)",
                batch_size, target_w, target_h)

    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
    )

    generator = image_generator.flow_from_dataframe(
        dataframe=df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        target_size=(target_w, target_h),
    )
    return generator


def get_test_and_valid_generator(valid_df: pd.DataFrame, test_df: pd.DataFrame,
                                  train_df: pd.DataFrame, image_dir: str, x_col: str,
                                  y_cols: List[str], sample_size: int = 100,
                                  batch_size: int = 8, seed: int = 1,
                                  target_w: int = 320, target_h: int = 320):
    """Generators for validation and test sets, normalized using train-set statistics.

    We deliberately do NOT normalize valid/test images using their own batch
    statistics — that would leak information about the evaluation data into
    the model's inputs. Instead we sample from the training set to estimate
    a fixed per-channel mean/std and reuse it for both valid and test.
    """
    logger.info("Building validation/test generators (sample_size=%d)", sample_size)

    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        batch_size=sample_size,
        shuffle=True,
        target_size=(target_w, target_h),
    )

    batch = raw_train_generator.next()
    data_sample = batch[0]

    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
    )
    image_generator.fit(data_sample)

    valid_generator = image_generator.flow_from_dataframe(
        dataframe=valid_df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        target_size=(target_w, target_h),
    )

    test_generator = image_generator.flow_from_dataframe(
        dataframe=test_df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        target_size=(target_w, target_h),
    )

    return valid_generator, test_generator


def build_all_generators(cfg: Dict[str, Any], train_df: pd.DataFrame, valid_df: pd.DataFrame,
                          test_df: pd.DataFrame):
    """Convenience wrapper that builds train/valid/test generators from config."""
    data_cfg = cfg["data"]
    image_cfg = cfg["image"]
    labels = cfg["labels"]

    train_generator = get_train_generator(
        train_df, data_cfg["image_dir"], data_cfg["image_col"], labels,
        batch_size=image_cfg["batch_size"], seed=image_cfg["seed"],
        target_w=image_cfg["target_w"], target_h=image_cfg["target_h"],
    )

    valid_generator, test_generator = get_test_and_valid_generator(
        valid_df, test_df, train_df, data_cfg["image_dir"], data_cfg["image_col"], labels,
        sample_size=image_cfg["sample_size"], batch_size=image_cfg["batch_size"],
        seed=image_cfg["seed"], target_w=image_cfg["target_w"], target_h=image_cfg["target_h"],
    )

    return train_generator, valid_generator, test_generator
