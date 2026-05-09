"""Метрики и графики для двух ML-задач."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "support_positive": int(y_true.sum()),
        "support_total": int(len(y_true)),
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    t_min: float,
    t_max: float,
    t_step: float,
) -> tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    thresholds = np.arange(t_min, t_max + t_step / 2, t_step)
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_t = float(t)
    return best_t, best_f1


def multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str],
) -> dict:
    report = classification_report(
        y_true, y_pred, labels=list(classes), zero_division=0, output_dict=True
    )
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "support_total": int(len(y_true)),
        "per_class": {
            cls: {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"]),
            }
            for cls in classes
            if cls in report
        },
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence,
    out_path: Path,
    title: str,
    display_labels: Sequence[str] | None = None,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    tick_labels = list(display_labels) if display_labels is not None else list(labels)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), max(5, len(labels) * 1.0)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ax=ax,
    )
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: Sequence[str],
    out_path: Path,
    title: str,
    top_k: int = 20,
) -> None:
    order = np.argsort(importances)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(8, max(4, len(order) * 0.3)))
    sns.barplot(
        x=importances[order],
        y=[feature_names[i] for i in order],
        ax=ax,
        color="steelblue",
    )
    ax.set_xlabel("Важность признака")
    ax.set_ylabel("")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
