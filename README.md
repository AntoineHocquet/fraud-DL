# fraud-DL (ULB Credit Card Fraud)

A **tiny**, **simple** and **modular-ish** baseline for fraud detection on the
[Kaggle ULB Credit Card Fraud dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).
Focus: clarity > features. Imbalanced learning done the simple way.

## Quickstart

```bash
# 1) Run the scaffold (from repo root named 'fraud-DL')
bash init_scaffold.sh

# 2) Put the dataset at:
#    data/raw/creditcard.csv
#    (Download from Kaggle and accept terms.)

# 3) Setup Python env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4) Train
python src/train.py --data data/raw/creditcard.csv

# 5) Predict (quick smoke-test; uses top rows by default)
python src/predict.py --model models/best_model.joblib --data data/raw/creditcard.csv --head 5
```

## What you get

- **Baselines**: LogisticRegression (+ class_weight) and MLPClassifier
- **Metrics**: ROC-AUC, PR-AUC; confusion matrix at F1-optimal threshold
- **Plots**: ROC, PR curves in `figs/`
- **Artifacts**: Best model (by PR-AUC) saved in `models/`

## Why this structure?

- **One-file training** (`src/train.py`) with tiny helpers in `src/utils.py`
- **One-file inference** (`src/predict.py`) you can re-use in interviews
- **No Kaggle API dependency** — just drop the CSV

## Notes

- This is intentionally minimal. Add CV, resampling, feature engineering, or
  more models as needed.
- The dataset is highly imbalanced (fraud ≈ 0.17%). We use:
  - `class_weight="balanced"` for LR and MLP
  - Threshold chosen by **max F1** on the validation set
- For a quick *DL-ish* flavor without extra deps we use `sklearn.neural_network.MLPClassifier`.
