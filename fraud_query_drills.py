#!/usr/bin/env python3
"""
fraud_query_drills.py

Pandas + (optional) SQL-style data-query drills on a synthetic transactions dataset.
Covers:
- Time filtering (last 30 days)
- GroupBy aggregations (fraud rates, counts)
- Rolling/expanding user features (prior averages)
- Sessionization by (user, device)
- Window-like ops in pandas
- Optional: run equivalent SQL via DuckDB if installed

Run:
    python fraud_query_drills.py --n 200000 --pos_rate 0.005
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Optional DuckDB for SQL mirrors
USE_DUCKDB = True
try:
    import duckdb
except Exception:
    USE_DUCKDB = False

def synthesize_transactions(n=200_000, pos_rate=0.005, seed=123):
    rng = np.random.RandomState(seed)
    X, y = make_classification(
        n_samples=n,
        n_features=8,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        weights=[1.0 - pos_rate, pos_rate],
        class_sep=1.2,
        random_state=seed
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["is_fraud"] = y.astype(int)

    # Time/user/device/country columns
    start = pd.Timestamp("2024-01-01 00:00:00")
    df["ts"] = pd.to_datetime(start) + pd.to_timedelta(np.arange(n), unit="min")
    df["user_id"] = rng.randint(1, 12000, size=n)
    df["amount"] = rng.exponential(scale=70, size=n).round(2)
    devices = np.array(["ios","android","web"])
    countries = np.array(["DE","FR","IT","ES","NL","PL","CZ","SE","FI"])
    df["device"] = devices[rng.randint(0, len(devices), size=n)]
    df["country"] = countries[rng.randint(0, len(countries), size=n)]

    return df

def drills_pandas(df):
    out = {}

    # 1) Fraud rate by country in the last 30 days with min 200 tx
    last_ts = df["ts"].max()
    cutoff = last_ts - pd.Timedelta(days=30)
    recent = df[df["ts"] >= cutoff]
    country_stats = (recent.groupby("country")["is_fraud"]
                        .agg(fraud_rate="mean", n="size")
                        .query("n >= 200")
                        .sort_values("fraud_rate", ascending=False)
                        .reset_index())
    out["fraud_rate_by_country_last30"] = country_stats

    # 2) User prior average amount (expanding mean, shifted) â€” leakage-safe
    df_sorted = df.sort_values(["user_id","ts"]).copy()
    df_sorted["user_prior_avg_amount"] = (
        df_sorted.groupby("user_id")["amount"]
                 .apply(lambda s: s.shift().expanding().mean())
                 .fillna(df["amount"].median())
                 .values
    )
    out["with_user_prior_avg_amount_head"] = df_sorted.head(10)

    # 3) Sessionization by (user, device): 30-min gaps
    g = df_sorted.groupby(["user_id","device"])["ts"]
    new_sess = g.diff().gt(pd.Timedelta(minutes=30)).astype(int)
    df_sorted["session_id"] = new_sess.groupby([df_sorted["user_id"], df_sorted["device"]]).cumsum()
    sessions = (df_sorted.groupby(["user_id","device","session_id"])
                          .agg(n_events=("amount","size"),
                               total_amt=("amount","sum"),
                               start=("ts","min"),
                               end=("ts","max"))
                          .reset_index())
    out["sessions_head"] = sessions.head(10)

    # 4) Top devices by fraud count (with distinct users)
    dev_stats = (df.groupby("device")
                   .agg(fraud_cnt=("is_fraud", "sum"),
                        users=("user_id", "nunique"),
                        n=("is_fraud","size"))
                   .sort_values("fraud_cnt", ascending=False)
                   .reset_index())
    out["top_devices_by_fraud"] = dev_stats

    # 5) Rolling fraud rate over time (overall, daily)
    daily = (df.set_index("ts")
               .resample("D")["is_fraud"]
               .mean()
               .rename("daily_fraud_rate")
               .to_frame())
    out["daily_fraud_rate_head"] = daily.head(15).reset_index()

    return out, df_sorted

def drills_sql(df):
    if not USE_DUCKDB:
        return {"note": "DuckDB not installed; skipping SQL mirrors."}

    con = duckdb.connect(database=":memory:")
    con.register("transactions", df)  # registers pandas df as a table

    results = {}

    # 1) Fraud rate by country last 30 days (min 200 tx)
    q1 = """
    WITH mx AS (SELECT MAX(ts) AS max_ts FROM transactions),
    recent AS (
      SELECT * FROM transactions, mx
      WHERE ts >= max_ts - INTERVAL 30 DAY
    )
    SELECT country,
           AVG(CASE WHEN is_fraud=1 THEN 1.0 ELSE 0.0 END) AS fraud_rate,
           COUNT(*) AS n
    FROM recent
    GROUP BY country
    HAVING COUNT(*) >= 200
    ORDER BY fraud_rate DESC;
    """
    results["sql_fraud_rate_by_country_last30"] = con.execute(q1).df()

    # 2) User prior average amount (window)
    q2 = """
    SELECT
      user_id, ts, amount,
      AVG(amount) OVER (
        PARTITION BY user_id
        ORDER BY ts
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
      ) AS user_prior_avg_amount
    FROM transactions
    ORDER BY user_id, ts
    LIMIT 10;
    """
    results["sql_user_prior_avg_amount_head"] = con.execute(q2).df()

    # 3) Top devices by fraud count with distinct users
    q3 = """
    SELECT device,
           SUM(is_fraud) AS fraud_cnt,
           COUNT(DISTINCT user_id) AS users,
           COUNT(*) AS n
    FROM transactions
    GROUP BY device
    ORDER BY fraud_cnt DESC;
    """
    results["sql_top_devices_by_fraud"] = con.execute(q3).df()

    # 4) Daily fraud rate
    q4 = """
    SELECT CAST(ts AS DATE) AS day,
           AVG(CASE WHEN is_fraud=1 THEN 1.0 ELSE 0.0 END) AS daily_fraud_rate
    FROM transactions
    GROUP BY day
    ORDER BY day
    LIMIT 15;
    """
    results["sql_daily_fraud_rate_head"] = con.execute(q4).df()

    con.close()
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200_000)
    parser.add_argument("--pos_rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    print("[1/3] Synthesizing transactions...")
    df = synthesize_transactions(args.n, args.pos_rate, args.seed)
    print(f"    -> shape={df.shape}, pos_rate={df['is_fraud'].mean():.4%}, time range: {df['ts'].min()} â€” {df['ts'].max()}")

    print("[2/3] Running pandas drills...")
    out, df_sorted = drills_pandas(df)
    # Save key outputs as CSV for quick inspection
    out["fraud_rate_by_country_last30"].to_csv("fraud_rate_by_country_last30.csv", index=False)
    out["top_devices_by_fraud"].to_csv("top_devices_by_fraud.csv", index=False)
    out["daily_fraud_rate_head"].to_csv("daily_fraud_rate_head.csv", index=False)

    print("    -> Samples:")
    print(out["fraud_rate_by_country_last30"].head(10).to_string(index=False))
    print(out["top_devices_by_fraud"].head(10).to_string(index=False))
    print(out["daily_fraud_rate_head"].head(10).to_string(index=False))

    print("[3/3] SQL mirrors (if DuckDB available)...")
    sql_res = drills_sql(df)
    if "note" in sql_res:
        print("    ->", sql_res["note"])
    else:
        # Save SQL outputs too
        for k, v in sql_res.items():
            v.to_csv(f"{k}.csv", index=False)
        print("    -> SQL examples executed via DuckDB. Saved CSVs.")

    print("Done. CSVs saved: fraud_rate_by_country_last30.csv, top_devices_by_fraud.csv, daily_fraud_rate_head.csv")
    print("If DuckDB was available, SQL CSVs are saved too.")

if __name__ == "__main__":
    main()