
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from feature_engineering import (
    engineer_retail_features,
    get_retail_model_features
)


def prepare_anomaly_data(df):

    if df.empty:
        raise ValueError("Dataset is empty.")

    original_data = df.copy()

    engineered = engineer_retail_features(original_data)

    model_data = get_retail_model_features(engineered)

    imputer = SimpleImputer(strategy="median")

    imputed = imputer.fit_transform(model_data)

    scaler = RobustScaler()

    scaled = scaler.fit_transform(imputed)

    if not np.isfinite(scaled).all():
        raise ValueError("Non-finite values found.")

    return original_data, scaled, list(model_data.columns)


def get_sample_indices(n_rows, max_rows):

    if n_rows <= max_rows:
        return np.arange(n_rows)

    rng = np.random.default_rng(42)

    return rng.choice(
        n_rows,
        size=max_rows,
        replace=False
    )


def run_isolation_forest(
    X,
    contamination=0.02,
    training_sample_size=20000
):

    start = time.perf_counter()

    indices = get_sample_indices(
        len(X), training_sample_size
    )

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X[indices])

    predictions = (
        model.predict(X) == -1
    ).astype(int)

    scores = -model.decision_function(X)

    return {
        "model": model,
        "predictions": predictions,
        "scores": scores,
        "execution_time": time.perf_counter() - start
    }


def run_lof(
    X,
    contamination=0.02,
    training_sample_size=20000
):

    start = time.perf_counter()

    indices = get_sample_indices(
        len(X), training_sample_size
    )

    # Remove duplicate feature vectors for LOF training
    train = np.unique(X[indices], axis=0)

    if len(train) < 3:
        raise ValueError(
            "LOF requires at least three unique records."
        )

    model = LocalOutlierFactor(
        n_neighbors=min(20, len(train) - 1),
        contamination=contamination,
        novelty=True,
        n_jobs=-1
    )

    model.fit(train)

    predictions = (
        model.predict(X) == -1
    ).astype(int)

    scores = -model.decision_function(X)

    return {
        "model": model,
        "predictions": predictions,
        "scores": scores,
        "execution_time": time.perf_counter() - start
    }


def run_one_class_svm(
    X,
    contamination=0.02,
    training_sample_size=10000
):

    start = time.perf_counter()

    indices = get_sample_indices(
        len(X), training_sample_size
    )

    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=contamination
    )

    model.fit(X[indices])

    predictions = (
        model.predict(X) == -1
    ).astype(int)

    scores = -model.decision_function(X).ravel()

    return {
        "model": model,
        "predictions": predictions,
        "scores": scores,
        "execution_time": time.perf_counter() - start
    }


def run_all_anomaly_models(
    original_data,
    X,
    contamination=0.02
):

    if len(original_data) != len(X):
        raise ValueError(
            "Original data and model data must have equal rows."
        )

    results = engineer_retail_features(original_data)

    models = [
        ("IF", "Isolation Forest", run_isolation_forest),
        ("LOF", "Local Outlier Factor", run_lof),
        ("SVM", "One-Class SVM", run_one_class_svm)
    ]

    summary_rows = []

    for prefix, name, runner in models:

        print(f"Running {name}...")

        outcome = runner(
            X,
            contamination=contamination
        )

        results[f"{prefix}_Anomaly"] = outcome["predictions"]

        results[f"{prefix}_Score"] = outcome["scores"]

        summary_rows.append({
            "Model": name,
            "Execution_Time_Seconds": round(
                outcome["execution_time"], 3
            ),
            "Anomalies_Detected": int(
                outcome["predictions"].sum()
            )
        })

    flag_columns = [
        "IF_Anomaly",
        "LOF_Anomaly",
        "SVM_Anomaly"
    ]

    results["Models_Flagged"] = (
        results[flag_columns].sum(axis=1)
    )

    results["Combined_Anomaly"] = (
        results["Models_Flagged"] >= 2
    ).astype(int)

    results["Anomaly_Status"] = np.where(
        results["Combined_Anomaly"] == 1,
        "Review",
        "Not flagged"
    )

    summary = pd.DataFrame(summary_rows)

    return results, summary


def export_anomaly_results(
    results_df,
    execution_summary,
    export_folder="exports"
):

    os.makedirs(export_folder, exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    complete_path = os.path.join(
        export_folder,
        f"complete_anomaly_analysis_{timestamp}.csv"
    )

    anomaly_path = os.path.join(
        export_folder,
        f"detected_anomalies_{timestamp}.csv"
    )

    summary_path = os.path.join(
        export_folder,
        f"model_execution_summary_{timestamp}.csv"
    )

    results_df.to_csv(
        complete_path,
        index=False
    )

    results_df[
        results_df["Combined_Anomaly"] == 1
    ].to_csv(
        anomaly_path,
        index=False
    )

    execution_summary.to_csv(
        summary_path,
        index=False
    )

    return {
        "complete_results": complete_path,
        "detected_anomalies": anomaly_path,
        "execution_summary": summary_path
    }
