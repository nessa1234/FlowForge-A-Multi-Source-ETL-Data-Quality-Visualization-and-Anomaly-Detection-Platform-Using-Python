
import os
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from feature_engineering import (
    engineer_retail_features,
    get_retail_model_features
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
ANOMALY_FRACTION = 0.05
MAX_TRAINING_SAMPLES = 10000


# ============================================================
# 2. INJECT CONTROLLED ANOMALIES
# ============================================================

def inject_anomalies(
    test_df,
    anomaly_fraction=ANOMALY_FRACTION,
    random_state=RANDOM_STATE
):

    if test_df is None or test_df.empty:
        raise ValueError("Test dataset is empty.")

    if not 0 < anomaly_fraction < 1:
        raise ValueError(
            "Anomaly fraction must be between 0 and 1."
        )

    # Reset index to avoid label-position mismatch.
    data = test_df.copy(deep=True).reset_index(drop=True)

    # Convert numeric columns to float.
    # This allows decimal values during anomaly injection.
    data["Quantity"] = pd.to_numeric(
        data["Quantity"],
        errors="coerce"
    ).astype("float64")

    data["UnitPrice"] = pd.to_numeric(
        data["UnitPrice"],
        errors="coerce"
    ).astype("float64")

    rng = np.random.default_rng(random_state)

    total_rows = len(data)

    anomaly_count = max(
        1,
        int(total_rows * anomaly_fraction)
    )

    anomaly_count = min(
        anomaly_count,
        total_rows
    )

    selected_indices = rng.choice(
        total_rows,
        size=anomaly_count,
        replace=False
    )

    # Ground-truth labels.
    data["True_Anomaly"] = 0

    data.loc[
        selected_indices,
        "True_Anomaly"
    ] = 1

    # Inject three types of artificial anomalies.
    for position, index in enumerate(selected_indices):

        anomaly_type = position % 3

        quantity = data.at[index, "Quantity"]
        price = data.at[index, "UnitPrice"]

        quantity = (
            0 if pd.isna(quantity) else quantity
        )

        price = (
            0 if pd.isna(price) else price
        )

        quantity_factor = rng.uniform(10, 30)
        price_factor = rng.uniform(10, 30)

        # Type 1: Extreme quantity.
        if anomaly_type == 0:

            data.at[index, "Quantity"] = (
                1000
                if quantity == 0
                else quantity * quantity_factor
            )

        # Type 2: Extreme unit price.
        elif anomaly_type == 1:

            data.at[index, "UnitPrice"] = (
                500
                if price == 0
                else price * price_factor
            )

        # Type 3: Extreme quantity and unit price.
        else:

            data.at[index, "Quantity"] = (
                1000
                if quantity == 0
                else quantity * quantity_factor
            )

            data.at[index, "UnitPrice"] = (
                500
                if price == 0
                else price * price_factor
            )

    return data


# ============================================================
# 3. PREPARE TRAINING AND TEST FEATURES
# ============================================================

def prepare_evaluation_features(
    train_df,
    test_df
):

    train_engineered = engineer_retail_features(
        train_df
    )

    test_engineered = engineer_retail_features(
        test_df
    )

    X_train = get_retail_model_features(
        train_engineered
    )

    X_test = get_retail_model_features(
        test_engineered
    )

    # Replace infinite values with missing values.
    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fit imputer only on training data.
    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imputed = imputer.fit_transform(
        X_train
    )

    X_test_imputed = imputer.transform(
        X_test
    )

    # Fit scaler only on training data.
    scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_imputed
    )

    X_test_scaled = scaler.transform(
        X_test_imputed
    )

    return X_train_scaled, X_test_scaled


# ============================================================
# 4. SELECT TRAINING SAMPLE
# ============================================================

def get_training_sample(
    X,
    max_samples=MAX_TRAINING_SAMPLES
):

    if len(X) <= max_samples:
        return X

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    indices = rng.choice(
        len(X),
        size=max_samples,
        replace=False
    )

    return X[indices]


# ============================================================
# 5. CALCULATE EVALUATION METRICS
# ============================================================

def calculate_metrics(
    model_name,
    y_true,
    y_pred,
    execution_time
):

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "Model": model_name,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_Score": round(f1, 4),
        "True_Positive": int(tp),
        "True_Negative": int(tn),
        "False_Positive": int(fp),
        "False_Negative": int(fn),
        "Execution_Time_Seconds": round(
            execution_time,
            3
        ),
        "Anomalies_Detected": int(
            np.sum(y_pred)
        )
    }


# ============================================================
# 6. EVALUATE ALL THREE ANOMALY DETECTION MODELS
# ============================================================

def evaluate_anomaly_models(
    df,
    test_size=TEST_SIZE,
    anomaly_fraction=ANOMALY_FRACTION,
    random_state=RANDOM_STATE
):

    if df is None or df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    if not 0 < test_size < 1:
        raise ValueError(
            "Test size must be between 0 and 1."
        )

    if not 0 < anomaly_fraction < 1:
        raise ValueError(
            "Anomaly fraction must be between 0 and 1."
        )

    required_columns = [
        "Quantity",
        "UnitPrice"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if len(df) < 10:
        raise ValueError(
            "Dataset must contain at least 10 records."
        )

    print("\n========================================")
    print("FLOWFORGE MODEL EVALUATION")
    print("========================================")

    print(
        "Original dataset:",
        len(df)
    )

    # Split original data before injecting anomalies.
    train_df, test_df = train_test_split(
        df.reset_index(drop=True),
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    print(
        "Training records:",
        len(train_df)
    )

    print(
        "Testing records:",
        len(test_df)
    )

    # Inject controlled anomalies only into test data.
    test_with_labels = inject_anomalies(
        test_df,
        anomaly_fraction=anomaly_fraction,
        random_state=random_state
    )

    y_true = test_with_labels[
        "True_Anomaly"
    ].to_numpy()

    # Keep labels separate from model features.
    test_features = test_with_labels.drop(
        columns=["True_Anomaly"]
    )

    # Preprocess without fitting on test data.
    X_train, X_test = prepare_evaluation_features(
        train_df,
        test_features
    )

    # Use a manageable training sample.
    X_train_sample = get_training_sample(
        X_train
    )

    print(
        "Training sample used:",
        len(X_train_sample)
    )

    print(
        "Injected anomalies:",
        int(y_true.sum())
    )

    if len(X_train_sample) < 3:
        raise ValueError(
            "Insufficient training records."
        )

    contamination = anomaly_fraction

    # --------------------------------------------------------
    # MODEL DEFINITIONS
    # --------------------------------------------------------

    models = [

        (
            "Isolation Forest",
            IsolationForest(
                n_estimators=100,
                contamination=contamination,
                random_state=random_state,
                n_jobs=-1
            )
        ),

        (
            "Local Outlier Factor",
            LocalOutlierFactor(
                n_neighbors=min(
                    20,
                    len(X_train_sample) - 1
                ),
                contamination=contamination,
                novelty=True,
                n_jobs=-1
            )
        ),

        (
            "One-Class SVM",
            OneClassSVM(
                kernel="rbf",
                gamma="scale",
                nu=contamination
            )
        )

    ]

    metrics_results = []

    prediction_results = pd.DataFrame({
        "True_Anomaly": y_true
    })

    # --------------------------------------------------------
    # TRAIN AND EVALUATE EACH MODEL
    # --------------------------------------------------------

    for model_name, model in models:

        print(
            f"\nTraining {model_name}..."
        )

        start_time = time.perf_counter()

        model.fit(
            X_train_sample
        )

        predictions = (
            model.predict(X_test) == -1
        ).astype(int)

        execution_time = (
            time.perf_counter() - start_time
        )

        prediction_column = (
            model_name.replace(" ", "_")
            + "_Prediction"
        )

        prediction_results[
            prediction_column
        ] = predictions

        metrics = calculate_metrics(
            model_name,
            y_true,
            predictions,
            execution_time
        )

        metrics_results.append(
            metrics
        )

        print(
            f"{model_name} completed in "
            f"{execution_time:.2f} seconds"
        )

    # --------------------------------------------------------
    # INDIVIDUAL MODEL RESULTS
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        metrics_results
    )

    # --------------------------------------------------------
    # MAJORITY VOTING ENSEMBLE
    # --------------------------------------------------------

    prediction_columns = [
        column
        for column in prediction_results.columns
        if column.endswith("_Prediction")
    ]

    prediction_results["Models_Flagged"] = (
        prediction_results[prediction_columns]
        .sum(axis=1)
    )

    prediction_results["Ensemble_Prediction"] = (
        prediction_results["Models_Flagged"] >= 2
    ).astype(int)

    ensemble_metrics = calculate_metrics(
        "Majority Voting Ensemble",
        y_true,
        prediction_results["Ensemble_Prediction"],
        0
    )

    metrics_df = pd.concat(
        [
            metrics_df,
            pd.DataFrame([ensemble_metrics])
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # DISPLAY FINAL RESULTS
    # --------------------------------------------------------

    print("\n========================================")
    print("MODEL EVALUATION RESULTS")
    print("========================================")

    print(
        metrics_df.to_string(index=False)
    )

    return metrics_df, prediction_results


# ============================================================
# 7. EXPORT EVALUATION RESULTS
# ============================================================

def export_evaluation_results(
    metrics_df,
    prediction_results,
    export_folder="exports"
):

    os.makedirs(
        export_folder,
        exist_ok=True
    )

    metrics_path = os.path.join(
        export_folder,
        "model_evaluation_metrics.csv"
    )

    predictions_path = os.path.join(
        export_folder,
        "model_evaluation_predictions.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    prediction_results.to_csv(
        predictions_path,
        index=False
    )

    print("\n========================================")
    print("EVALUATION EXPORTS COMPLETED")
    print("========================================")

    print(
        "Metrics:",
        metrics_path
    )

    print(
        "Predictions:",
        predictions_path
    )

    return metrics_path, predictions_path


# ============================================================
# 8. MAIN EXECUTION FUNCTION
# ============================================================

def run_model_evaluation(df):

    metrics_df, prediction_results = (
        evaluate_anomaly_models(df)
    )

    export_evaluation_results(
        metrics_df,
        prediction_results
    )

    return metrics_df, prediction_results
