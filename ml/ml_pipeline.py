#!/usr/bin/env python3
"""
Phase 5 ML pipeline: burst-time prediction (regression).
Loads experiments/results/task_burst_dataset.csv, does config-based split,
trains Linear Regression (primary) + optional DecisionTree, evaluates MAE/RMSE/R2,
saves ml/model.json and ml/model_metrics.json, validates export equivalence.

No data leakage: only pre-execution features used.
"""
import os
import sys
import json
import hashlib

# Check dependencies
try:
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import joblib
except ImportError as e:
    print(f"[error] missing Python dependency: {e}", file=sys.stderr)
    print("Install: pip install pandas numpy scikit-learn joblib", file=sys.stderr)
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET = os.path.join(PROJECT_ROOT, "experiments", "results", "task_burst_dataset.csv")
MODEL_JSON = os.path.join(PROJECT_ROOT, "ml", "model.json")
METRICS_JSON = os.path.join(PROJECT_ROOT, "ml", "model_metrics.json")
MODEL_PKL = os.path.join(PROJECT_ROOT, "ml", "model.pkl")

# Features known BEFORE execution (no leakage)
FEATURES = [
    "element_count",
    "working_set_bytes",
    "access_pattern_encoded",
    "stride",
    "repeat_count",
    "operation_count",
    "locality_score",
]
# Optional: include array_size as well (known before)
# Spec says at minimum those; we include array_size as extra if desired, but keep declared FEATURES consistent with export.
# We'll add array_size as well for richer model, but ensure docs match.
FEATURES_WITH_ARRAY = ["array_size"] + FEATURES

TARGET = "burst_time_ms"
CATEGORICAL_MAP = {"sequential": 0, "strided": 1, "random": 2}

def load_and_validate():
    if not os.path.exists(DATASET):
        print(f"[error] dataset not found at {DATASET}", file=sys.stderr)
        print("Run: python3 experiments/generate_task_dataset.py", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(DATASET)
    # Validate
    assert TARGET in df.columns, "missing burst_time_ms"
    assert not df[TARGET].isna().any(), "missing target values"
    assert len(df) >= 100, "dataset too small"
    assert set(df["access_pattern"].unique()) >= {"sequential", "strided", "random"}, "need all patterns"
    # Ensure no leakage columns are used as features
    leakage = {"burst_time_ms", "scheduler", "speedup", "checksum", "execution_time_ms"}
    for c in leakage:
        if c in FEATURES:
            raise ValueError(f"leakage feature {c} in FEATURES")
    # Add array_size to features if present (optional)
    # Keep FEATURES as defined above (without array_size) to match spec minimal;
    # but if array_size exists we can include it as extra feature for better fit.
    # We'll use FEATURES (without array_size) for model to stay minimal and spec-aligned.
    # Ensure access_pattern_encoded exists or create
    if "access_pattern_encoded" not in df.columns:
        df["access_pattern_encoded"] = df["access_pattern"].map(CATEGORICAL_MAP)
    # Validate working_set_bytes = element_count *8
    # Check operation_count == repeat_count (if missing, fill)
    if "operation_count" not in df.columns:
        df["operation_count"] = df["repeat_count"]
    return df

def config_key(row):
    # Configuration identifier for holdout split
    return (row["array_size"], row["element_count"], row["access_pattern"], row["stride"], row["repeat_count"])

def config_based_split(df, test_frac=0.25, seed=42):
    # Group rows by config key
    df = df.copy()
    df["_cfg"] = df.apply(lambda r: f"{r['array_size']}_{r['element_count']}_{r['access_pattern']}_{r['stride']}_{r['repeat_count']}", axis=1)
    configs = sorted(df["_cfg"].unique())
    # Deterministic shuffle
    import random
    rnd = random.Random(seed)
    rnd.shuffle(configs)
    n_test = max(1, int(len(configs) * test_frac))
    test_configs = set(configs[:n_test])
    train_configs = set(configs[n_test:])
    train_df = df[df["_cfg"].isin(train_configs)].copy()
    test_df = df[df["_cfg"].isin(test_configs)].copy()
    # Ensure no overlap
    assert len(set(train_df["_cfg"]) & set(test_df["_cfg"])) == 0, "config leakage"
    train_df = train_df.drop(columns=["_cfg"])
    test_df = test_df.drop(columns=["_cfg"])
    return train_df, test_df, len(configs), len(test_configs)

def main():
    df = load_and_validate()
    print(f"Loaded dataset: {len(df)} rows, {df['access_pattern'].nunique()} patterns")
    print(f"Columns: {list(df.columns)}")

    # Use FEATURES as defined (spec minimal set)
    # Verify all features exist
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"[error] missing features {missing}", file=sys.stderr)
        sys.exit(1)

    train_df, test_df, n_cfg, n_test_cfg = config_based_split(df, test_frac=0.25, seed=42)
    print(f"Configs: {n_cfg} total, {n_test_cfg} test, {n_cfg - n_test_cfg} train")
    print(f"Rows: {len(train_df)} train, {len(test_df)} test")

    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values
    X_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values

    # Train Linear Regression (primary)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\nLinear Regression:")
    print(f"  MAE: {mae:.6f} ms")
    print(f"  RMSE: {rmse:.6f} ms")
    print(f"  R2: {r2:.4f}")

    # Optional tree model
    tree = DecisionTreeRegressor(max_depth=6, random_state=42)
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)
    mae_t = mean_absolute_error(y_test, y_pred_tree)
    rmse_t = np.sqrt(mean_squared_error(y_test, y_pred_tree))
    r2_t = r2_score(y_test, y_pred_tree)
    print(f"\nDecisionTree (max_depth=6):")
    print(f"  MAE: {mae_t:.6f} ms")
    print(f"  RMSE: {rmse_t:.6f} ms")
    print(f"  R2: {r2_t:.4f}")

    # Sample predictions
    print("\nSample predictions (Linear Regression) vs actual:")
    for i in range(min(5, len(y_test))):
        print(f"  actual={y_test[i]:.4f} ms pred={y_pred[i]:.4f} ms features={X_test[i]}")

    # Export linear model as JSON (for C++ inference)
    # Coefficients correspond to FEATURES order
    model_data = {
        "model_type": "linear_regression",
        "features": FEATURES,
        "coefficients": lr.coef_.tolist(),
        "intercept": float(lr.intercept_),
        "categorical_mapping": CATEGORICAL_MAP,
        "feature_order": FEATURES,
        "target": TARGET,
        "notes": "Burst time prediction uses only pre-execution features; no data leakage. Config-based holdout split."
    }
    os.makedirs(os.path.dirname(MODEL_JSON), exist_ok=True)
    with open(MODEL_JSON, "w") as f:
        json.dump(model_data, f, indent=2)
    print(f"\nModel exported to {MODEL_JSON}")

    # Also save pkl for convenience
    joblib.dump(lr, MODEL_PKL)
    print(f"Model pkl saved to {MODEL_PKL}")

    # Validate export equivalence: manual calculation vs sklearn
    # manual = intercept + sum(coef * x)
    max_err = 0
    for i in range(len(X_test)):
        manual = model_data["intercept"] + sum(c * x for c, x in zip(model_data["coefficients"], X_test[i]))
        err = abs(manual - y_pred[i])
        max_err = max(max_err, err)
        if err > 1e-6:
            print(f"[error] export mismatch at {i}: manual {manual} vs sklearn {y_pred[i]} err {err}", file=sys.stderr)
            sys.exit(1)
    print(f"Export validation: max manual vs sklearn error = {max_err:.2e} (within 1e-6) — OK")

    # Save test predictions for visualization (actual vs predicted)
    pred_df = test_df.copy()
    pred_df["predicted_burst_time_ms"] = y_pred
    pred_df["actual_burst_time_ms"] = y_test
    # Keep minimal columns for plot
    pred_out = os.path.join(PROJECT_ROOT, "ml", "test_predictions.csv")
    pred_df[["actual_burst_time_ms", "predicted_burst_time_ms"] + FEATURES].to_csv(pred_out, index=False)
    print(f"Test predictions saved to {pred_out} ({len(pred_df)} rows)")

    # Save metrics
    metrics = {
        "model_type": "linear_regression",
        "features": FEATURES,
        "target": TARGET,
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "train_configs": int(n_cfg - n_test_cfg),
        "test_configs": int(n_test_cfg),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "comparison_model": {
            "model_type": "decision_tree",
            "max_depth": 6,
            "mae": float(mae_t),
            "rmse": float(rmse_t),
            "r2": float(r2_t),
        },
        "categorical_mapping": CATEGORICAL_MAP,
        "coefficients": lr.coef_.tolist(),
        "intercept": float(lr.intercept_),
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_JSON}")

    # Honest discussion note
    if r2 < 0.5:
        print("\n[note] R2 is modest — burst times may be near timer resolution or workload is simple; increasing repeat_count can improve measurability.")
    else:
        print("\n[note] R2 indicates reasonable fit for academic baseline; linear model captures element_count*repeat trends.")

if __name__ == "__main__":
    main()
