"""
Export Pipeline for Power BI Dashboard
Predictive Maintenance AI - Turbofan Engine RUL
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor

def prepare_powerbi_data(data_path="train_FD004.txt", output_csv="powerbi_engines_data.csv"):
    if not os.path.exists(data_path):
        print(f"[-] Data file '{data_path}' not found in current directory.")
        print("[!] Please place 'train_FD004.txt' in the project folder and re-run.")
        return

    print("[+] Loading NASA C-MAPSS dataset...")
    columns = (
        ["unit_number", "time_in_cycles"]
        + ["setting_1", "setting_2", "setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    df = pd.read_csv(data_path, sep=r"\s+", header=None, names=columns)

    # 1. Calculate Actual RUL
    max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df["RUL_unclipped"] = max_cycles - df["time_in_cycles"]
    df["Actual_RUL"] = df["RUL_unclipped"].clip(upper=125)

    # 2. Features and Target
    feature_cols = [c for c in df.columns if c.startswith("sensor_") or c.startswith("setting_")]
    X = df[feature_cols]
    y = df["Actual_RUL"]
    groups = df["unit_number"]

    # 3. Train-Test Split (by unit_number to prevent leakage)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print("[+] Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 4. Generate Predictions for the test fleet
    print("[+] Generating predictions...")
    test_df = df.iloc[test_idx].copy()
    test_df["Predicted_RUL"] = np.round(model.predict(X_test), 1)
    test_df["Absolute_Error"] = np.round(np.abs(test_df["Actual_RUL"] - test_df["Predicted_RUL"]), 1)

    # 5. Business logic for Power BI dashboards
    test_df["Risk_Category"] = np.where(
        test_df["Predicted_RUL"] <= 30,
        "Critical (0-30)",
        np.where(test_df["Predicted_RUL"] <= 75, "Warning (31-75)", "Healthy (>75)")
    )

    test_df["Maintenance_Action"] = np.where(
        test_df["Predicted_RUL"] <= 30,
        "🚨 Urgent Maintenance",
        np.where(test_df["Predicted_RUL"] <= 75, "⚠️ Schedule Inspection", "✅ Normal Operation")
    )

    # Export main fleet table
    export_cols = [
        "unit_number",
        "time_in_cycles",
        "Actual_RUL",
        "Predicted_RUL",
        "Absolute_Error",
        "Risk_Category",
        "Maintenance_Action",
        "sensor_9",
        "sensor_11",
        "sensor_13",
        "sensor_14",
        "sensor_15",
        "setting_1",
        "setting_2",
        "setting_3"
    ]
    test_df[export_cols].to_csv(output_csv, index=False)
    print(f"[✔] Successfully exported Power BI dataset: '{output_csv}' ({len(test_df)} rows)")

    # 6. Export Feature Importance table
    fi_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": np.round(model.feature_importances_, 4)
    }).sort_values("Importance", ascending=False)
    
    fi_csv = "powerbi_feature_importance.csv"
    fi_df.to_csv(fi_csv, index=False)
    print(f"[✔] Successfully exported Feature Importance: '{fi_csv}'")

if __name__ == "__main__":
    prepare_powerbi_data()
