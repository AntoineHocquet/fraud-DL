from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Tuple, Dict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, confusion_matrix, f1_score
)
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class Metrics:
    roc_auc: float
    pr_auc: float
    best_threshold: float
    f1_at_best: float
    conf_matrix: np.ndarray


def pick_best_threshold_by_f1(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """Return (best_threshold, f1_at_best) by sweeping thresholds on y_scores."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    # Map PR curve points to F1; skip first point (threshold undefined)
    f1s = (2 * precisions[1:] * recalls[1:]) / (precisions[1:] + recalls[1:] + 1e-12)
    idx = int(np.argmax(f1s))
    best_thr = float(thresholds[idx])
    return best_thr, float(f1s[idx])


def evaluate_binary(y_true: np.ndarray, y_scores: np.ndarray, out_dir: Path) -> Metrics:
    out_dir.mkdir(parents=True, exist_ok=True)
    roc_auc = float(roc_auc_score(y_true, y_scores))
    pr_auc = float(average_precision_score(y_true, y_scores))
    best_thr, f1b = pick_best_threshold_by_f1(y_true, y_scores)

    # Confusion matrix at best threshold
    y_pred = (y_scores >= best_thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png")
    plt.close()

    # PR curve
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(rec, prec, label=f"PR AUC={pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png")
    plt.close()

    return Metrics(roc_auc=roc_auc, pr_auc=pr_auc, best_threshold=best_thr, f1_at_best=f1b, conf_matrix=cm)
