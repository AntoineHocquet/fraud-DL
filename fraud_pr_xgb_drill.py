#!/usr/bin/env python3
"""
fraud_pr_xgb_drill.py

End-to-end practice script for an imbalanced "fraud/risk" classification workflow:
- Synthetic data generation (heavy class imbalance, ~0.5% positives)
- Basic feature engineering (time, user aggregates, simple categoricals)
- Chronological split (60/20/20)
- Model training (XGBoost if available; otherwise HistGradientBoosting as fallback)
- Optimize threshold using PR curve (F1-opt and/or precision constraint)
- Report metrics and save PR curve plot

Run:
    python fraud_pr_xgb_drill.py --n 120000 --pos_rate 0.005 --min_precision 0.90

Requirements:
    - numpy, pandas, scikit-learn, matplotlib
    - xgboost (optional; script falls back gracefully if not installed)
"""

import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from datetime import timedelta

# Optional XGBoost import with fallback
USE_XGB = True
try:
    from xgboost import XGBClassifier
except Exception:
    USE_XGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

import matplotlib.pyplot as plt


def synthesize_transactions(n=100_000, pos_rate=0.005, seed=42):
    rng = np.random.RandomState(seed)
    X, y = make_classification(
        n_samples=n,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        weights=[1.0 - pos_rate, pos_rate],
        class_sep=1.0,
        flip_y=0.001,
        random_state=seed
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["is_fraud"] = y

    # Time column: one event per 10 seconds
    start = pd.Timestamp("2024-01-01 00:00:00")
    df["ts"] = pd.to_datetime(start) + pd.to_timedelta(np.arange(n)*10, unit="s")

    # User/device/country to mimic tabular heterogeneity
    df["user_id"] = rng.randint(1, 8000, size=n)
    df["amount"] = rng.exponential(scale=80, size=n).round(2)
    devices = np.array(["ios","android","web"])
    countries = np.array(["DE","FR","IT","ES","NL","PL","CZ","SE","FI"])
    df["device"] = devices[rng.randint(0, len(devices), size=n)]
    df["country"] = countries[rng.randint(0, len(countries), size=n)]

    # Light signal leakage-free user aggregate: prior avg amount
    df = df.sort_values(["user_id","ts"]).reset_index(drop=True)
    df["user_prior_avg_amount"] = (
        df.groupby("user_id")["amount"]
          .apply(lambda s: s.shift().expanding().mean())
          .fillna(df["amount"].median())
          .values
    )

    # Extract simple time features
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.dayofweek

    # Shuffle back by time only (we'll chrono-split later across the full dataset)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def chrono_split(df, train_frac=0.6, val_frac=0.2):
    n = len(df)
    cut1 = int(n * train_frac)
    cut2 = int(n * (train_frac + val_frac))
    return df.iloc[:cut1], df.iloc[cut1:cut2], df.iloc[cut2:]


def pick_threshold(y_true, scores, min_precision=None):
    p, r, t = precision_recall_curve(y_true, scores)
    if min_precision is None:
        f1 = 2*p*r/(p+r+1e-12)
        best = np.argmax(f1)
        thr = t[best-1] if best > 0 else 0.5
        return thr, {"prec": float(p[best]), "rec": float(r[best]), "criterion": "max F1"}
    ok = np.where(p >= min_precision)[0]
    if len(ok) == 0:
        # No threshold satisfies the precision; return degenerate "predict none"
        return 1.0, {"prec": 1.0, "rec": 0.0, "criterion": f"precisionâ‰¥{min_precision} (no feasible threshold)"}
    best = ok[np.argmax(r[ok])]
    thr = t[best-1] if best > 0 else 1.0
    return thr, {"prec": float(p[best]), "rec": float(r[best]), "criterion": f"precisionâ‰¥{min_precision} max recall"}


def train_and_eval(df, min_precision=None, seed=0):
    features_num = ["amount","user_prior_avg_amount","hour","dow"] + [f"f{i}" for i in range(10)]
    features_cat = ["device","country"]
    target = "is_fraud"

    train, val, test = chrono_split(df, 0.6, 0.2)
    X_train, y_train = train[features_num+features_cat], train[target].astype(int).values
    X_val,   y_val   = val[features_num+features_cat],   val[target].astype(int).values
    X_test,  y_test  = test[features_num+features_cat],  test[target].astype(int).values

    pre = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore"), features_cat)
    ], remainder="passthrough")

    if USE_XGB:
        neg, pos = np.bincount(y_train)
        scale_pos_weight = neg / max(pos, 1)
        model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            max_delta_step=1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=seed
        )
    else:
        # Fallback: HistGradientBoosting (use class_weight via sample_weight)
        model = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=500,
            l2_regularization=1.0,
            random_state=seed
        )

    pipe = Pipeline([("pre", pre), ("clf", model)])

    if USE_XGB:
        pipe.fit(X_train, y_train)
        val_scores = pipe.predict_proba(X_val)[:,1]
        test_scores = pipe.predict_proba(X_test)[:,1]
    else:
        # For HGB, approximate class weighting via sample_weight
        neg, pos = np.bincount(y_train)
        scale_pos_weight = neg / max(pos, 1)
        sample_weight = np.where(y_train==1, scale_pos_weight, 1.0).astype(float)
        pipe.fit(X_train, y_train, clf__sample_weight=sample_weight)
        # decision_function_ not available; use predicted probabilities via predict_proba if supported
        # HGB supports predict_proba from sklearn >= 1.0
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            val_scores = pipe.predict_proba(X_val)[:,1]
            test_scores = pipe.predict_proba(X_test)[:,1]
        else:
            # Fallback: use decision_function if available; else use predictions
            if hasattr(pipe.named_steps["clf"], "decision_function"):
                val_scores = pipe.decision_function(X_val)
                test_scores = pipe.decision_function(X_test)
            else:
                val_scores = pipe.predict(X_val).astype(float)
                test_scores = pipe.predict(X_test).astype(float)

    ap_val = average_precision_score(y_val, val_scores)

    thr, crit = pick_threshold(y_val, val_scores, min_precision=min_precision)
    y_pred = (test_scores >= thr).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    # Plot PR curve (validation) and save
    p, r, t = precision_recall_curve(y_val, val_scores)
    plt.figure()
    plt.plot(r, p, label=f'PR (AP={ap_val:.4f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Validation Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_png = "pr_curve_validation.png"
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close()

    return {
        "ap_val": float(ap_val),
        "threshold": float(thr),
        "criterion": crit,
        "confusion_matrix_test": cm.tolist(),
        "classification_report_test": report,
        "pr_curve_png": out_png,
        "used_model": "XGBoost" if USE_XGB else "HistGradientBoosting (fallback)"
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=120_000, help="Number of samples")
    parser.add_argument("--pos_rate", type=float, default=0.005, help="Positive class rate (e.g., 0.005 = 0.5%)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_precision", type=float, default=None, help="If set, choose threshold achieving at least this precision (maximize recall).")
    args = parser.parse_args()

    print("[1/4] Synthesizing data...")
    df = synthesize_transactions(n=args.n, pos_rate=args.pos_rate, seed=args.seed)
    pos_rate_emp = df["is_fraud"].mean()
    print(f"    -> shape={df.shape}, empirical_pos_rate={pos_rate_emp:.4%}")

    print("[2/4] Training and evaluating...")
    results = train_and_eval(df, min_precision=args.min_precision, seed=args.seed)

    print("[3/4] Results")
    print(f"    Model: {results['used_model']}")
    print(f"    Validation AP (PR-AUC): {results['ap_val']:.6f}")
    print(f"    Threshold: {results['threshold']:.6f} | Criterion: {results['criterion']}")
    print("    Confusion matrix (TEST) [ [TN, FP], [FN, TP] ]:")
    print(f"    {results['confusion_matrix_test']}")
    print("    Classification report (TEST):")
    print(results["classification_report_test"].strip())

    print("[4/4] Saved PR curve plot to:", results["pr_curve_png"])
    print("\nTip: Try `--min_precision 0.90` to simulate business precision constraints, or remove it to use F1-opt threshold.")


if __name__ == "__main__":
    main()