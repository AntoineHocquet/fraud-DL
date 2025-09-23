from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

def main():
    parser = argparse.ArgumentParser(description="Load trained model and score samples.")
    parser.add_argument("--model", type=str, required=True, help="Path to models/best_model.joblib")
    parser.add_argument("--data", type=str, required=True, help="Path to creditcard.csv (for demo) or your own CSV with same columns")
    parser.add_argument("--head", type=int, default=10, help="Score only first N rows (for a quick smoke test)")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    pipe = bundle["model"]
    features = bundle["features"]
    thr = float(bundle.get("threshold", 0.5))

    df = pd.read_csv(args.data)
    if not all(col in df.columns for col in features):
        missing = [c for c in features if c not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")

    X = df[features].values.astype(float)
    if args.head:
        X = X[:args.head]

    scores = pipe.predict_proba(X)[:, 1]
    preds = (scores >= thr).astype(int)

    out = pd.DataFrame({"score": scores, "pred": preds})
    print(out.head(len(scores)))

if __name__ == "__main__":
    main()
