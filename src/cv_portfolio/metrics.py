"""Evaluation helpers for classification and segmentation experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np


def pixel_accuracy(y_true: np.ndarray, y_pred: np.ndarray, ignore_class: int | None = None) -> float:
    """Compute pixel accuracy for dense segmentation labels."""
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    mask = np.ones(truth.shape, dtype=bool)
    if ignore_class is not None:
        mask &= truth != ignore_class
    if not np.any(mask):
        return 0.0
    return float(np.mean(truth[mask] == prediction[mask]))


def mean_iou(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    ignore_class: int | None = None,
) -> float:
    """Compute mean intersection-over-union over present classes."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    scores: list[float] = []
    for class_id in range(num_classes):
        if class_id == ignore_class:
            continue
        true_mask = truth == class_id
        pred_mask = prediction == class_id
        union = np.logical_or(true_mask, pred_mask).sum()
        if union == 0:
            continue
        intersection = np.logical_and(true_mask, pred_mask).sum()
        scores.append(float(intersection / union))

    return float(np.mean(scores)) if scores else 0.0


def best_classification_variant(rows: Iterable[dict[str, str]]) -> dict[str, str]:
    """Return the row with the highest final test accuracy."""
    return max(rows, key=lambda row: float(row["FinalTestAcc"]))


def load_task2_summary(path: str | Path) -> list[dict[str, str]]:
    """Load the Task 2 classification comparison table."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
