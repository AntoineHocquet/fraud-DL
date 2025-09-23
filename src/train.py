from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from utils import evaluate_binary

# Keep it dead simple: we split once and compare LR vs MLP by PR-AUC on val.
# Best model is saved to models/best_model.joblib along with scaler in the pipeline.

def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    y = df["Class"].values.astype(int)
    # Use all V1..V28 + Amount; Time is often omitted, we'll drop it for simplicity
    feature_cols = [c for c in df.columns if c not in ("Class", "Time")]
    X = df[feature_cols].values.astype(float)
    return X, y, feature_cols

def main():
    parser = argparse.ArgumentParser(description="Train simple fraud models (LR & MLP) and pick best by PR-AUC.")
    parser.add_argument("--data", type=str, required=True, help="Path to data/raw/creditcard.csv")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation fraction (of train)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_path = Path(args.data)
    X, y, feature_cols = load_data(data_path)

    # Train/Val/Test split (two-stage to keep stratification simple)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=args.val_size, stratify=y_trainval, random_state=args.random_state
    )

    # Two pipelines: Logistic Regression and MLP (both with scaling)
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None, penalty="l2", C=1.0)),
    ])

    mlp_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                              max_iter=30, early_stopping=True, n_iter_no_change=5, random_state=args.random_state)),
    ])

    # Fit both
    lr_pipe.fit(X_train, y_train)
    mlp_pipe.fit(X_train, y_train)

    # Validate: choose by PR-AUC
    from sklearn.metrics import average_precision_score
    lr_val_scores = lr_pipe.predict_proba(X_val)[:, 1]
    mlp_val_scores = mlp_pipe.predict_proba(X_val)[:, 1]
    lr_pr = float(average_precision_score(y_val, lr_val_scores))
    mlp_pr = float(average_precision_score(y_val, mlp_val_scores))

    chosen_name, chosen_pipe = ("lr", lr_pipe) if lr_pr >= mlp_pr else ("mlp", mlp_pipe)
    print(f"[Selection] PR-AUC val — LR={lr_pr:.5f}, MLP={mlp_pr:.5f} -> chosen={chosen_name.upper()}")

    # Final evaluation on test
    y_scores = chosen_pipe.predict_proba(X_test)[:, 1]
    metrics = evaluate_binary(y_true=y_test, y_scores=y_scores, out_dir=Path("figs"))

    # Persist best
    out_model = Path("models") / "best_model.joblib"
    joblib.dump({"model": chosen_pipe, "features": feature_cols, "threshold": metrics.best_threshold}, out_model)
    print(f"[Saved] {out_model} (threshold={metrics.best_threshold:.6f})")

    # Show confusion matrix & summary
    from sklearn.metrics import confusion_matrix
    y_pred = (y_scores >= metrics.best_threshold).astype(int)
    print("[Test] Confusion matrix at F1-optimal threshold:")
    print(confusion_matrix(y_test, y_pred))
    print(f"[Test] ROC-AUC={metrics.roc_auc:.6f} | PR-AUC={metrics.pr_auc:.6f} | F1@best={metrics.f1_at_best:.6f}")

if __name__ == "__main__":
    main()
