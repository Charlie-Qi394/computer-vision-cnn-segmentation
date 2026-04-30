"""Generate lightweight SVG plots from included metric artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = REPO_ROOT / "artifacts" / "metrics"
PLOTS_DIR = REPO_ROOT / "artifacts" / "plots"


def plot_task2_summary() -> Path:
    data = pd.read_csv(METRICS_DIR / "task2_summary.csv")
    data = data.sort_values("FinalTestAcc", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(data["Variant"], data["FinalTestAcc"], color="#2f6f73")
    ax.set_xlabel("Final test accuracy (%)")
    ax.set_title("CIFAR-100 Architecture Comparison")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output = PLOTS_DIR / "cifar_architecture_comparison.svg"
    fig.savefig(output, format="svg")
    plt.close(fig)
    return output


def plot_task4_training() -> Path:
    data = pd.read_csv(METRICS_DIR / "task4_training_log.csv")
    with (METRICS_DIR / "task4_final_results.json").open(encoding="utf-8") as handle:
        final = json.load(handle)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(data["epoch"], data["accuracy"], label="train accuracy", color="#2f6f73")
    ax.plot(data["epoch"], data["val_accuracy"], label="validation accuracy", color="#9b5de5")
    ax.axhline(final["final_test_accuracy"], color="#c2410c", linestyle="--", label="final test accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Final Tuned Model Training")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output = PLOTS_DIR / "final_tuned_training_accuracy.svg"
    fig.savefig(output, format="svg")
    plt.close(fig)
    return output


def main() -> None:
    for path in (plot_task2_summary(), plot_task4_training()):
        print(path)


if __name__ == "__main__":
    main()
