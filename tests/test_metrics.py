import numpy as np

from cv_portfolio.metrics import best_classification_variant, mean_iou, pixel_accuracy


def test_pixel_accuracy_with_ignore_class():
    y_true = np.array([[0, 1, 255], [1, 1, 0]])
    y_pred = np.array([[0, 0, 255], [1, 1, 1]])

    assert pixel_accuracy(y_true, y_pred, ignore_class=255) == 0.6


def test_mean_iou_for_present_classes():
    y_true = np.array([[0, 1], [1, 2]])
    y_pred = np.array([[0, 1], [2, 2]])

    assert round(mean_iou(y_true, y_pred, num_classes=3), 4) == 0.6667


def test_best_classification_variant():
    rows = [
        {"Variant": "A", "FinalTestAcc": "35.0"},
        {"Variant": "B", "FinalTestAcc": "45.23"},
    ]

    assert best_classification_variant(rows)["Variant"] == "B"
