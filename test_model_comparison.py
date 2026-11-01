
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# -----------------------------------------
# 1. LOAD RETAIL DATASET
# -----------------------------------------

file_path = r"C:\Users\sarat\Downloads\online+retail\Online Retail.xlsx"

df = pd.read_excel(file_path)

print("Original dataset:", df.shape)

# -----------------------------------------
# 2. PREPARE TEST DATA
# -----------------------------------------

features = ["Quantity", "UnitPrice"]

test_df = df[features].copy()

for column in features:
    test_df[column] = pd.to_numeric(
        test_df[column], errors="coerce"
    )

test_df = test_df.replace([np.inf, -np.inf], np.nan)
test_df = test_df.dropna().copy()

# Use a manageable, reproducible sample
test_df = test_df.sample(
    n=min(20000, len(test_df)),
    random_state=42
).copy()

# -----------------------------------------
# 3. INJECT KNOWN ANOMALIES
# -----------------------------------------

test_df["Actual_Anomaly"] = 0

rng = np.random.default_rng(42)

anomaly_count = int(len(test_df) * 0.02)

anomaly_indices = rng.choice(
    test_df.index.to_numpy(),
    size=anomaly_count,
    replace=False
)

# Create unusual transaction values
test_df.loc[anomaly_indices, "Quantity"] = (
    rng.choice([-1, 1], size=anomaly_count)
    * rng.integers(1000, 5000, size=anomaly_count)
)

test_df.loc[anomaly_indices, "UnitPrice"] = (
    rng.uniform(500, 2000, size=anomaly_count)
)

test_df.loc[anomaly_indices, "Actual_Anomaly"] = 1

print("\nTEST DATA")
print("----------------------------")
print("Total records:", len(test_df))
print("Injected anomalies:", anomaly_count)
print("Normal records:", len(test_df) - anomaly_count)

# -----------------------------------------
# 4. PREPARE FEATURES
# -----------------------------------------

X = test_df[features].to_numpy()
y_true = test_df["Actual_Anomaly"].to_numpy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------
# 5. RUN THREE MODELS
# -----------------------------------------

model_predictions = {}

# Isolation Forest
print("\nRunning Isolation Forest...")

if_model = IsolationForest(
    contamination=0.02,
    random_state=42,
    n_estimators=100
)

model_predictions["Isolation Forest"] = (
    if_model.fit_predict(X_scaled) == -1
).astype(int)

# Local Outlier Factor
print("Running Local Outlier Factor...")

lof_model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.02,
    novelty=False
)

model_predictions["Local Outlier Factor"] = (
    lof_model.fit_predict(X_scaled) == -1
).astype(int)

# One-Class SVM
print("Running One-Class SVM...")

svm_model = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.02
)

model_predictions["One-Class SVM"] = (
    svm_model.fit_predict(X_scaled) == -1
).astype(int)

# -----------------------------------------
# 6. EVALUATE MODELS
# -----------------------------------------

results = []

for model_name, y_pred in model_predictions.items():

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_Score": round(f1, 4),
        "True_Positives": tp,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Negatives": tn
    })

# -----------------------------------------
# 7. DISPLAY COMPARISON
# -----------------------------------------

comparison_df = pd.DataFrame(results)

print("\nMODEL PERFORMANCE COMPARISON")
print("================================")

print(
    comparison_df.to_string(index=False)
)

# -----------------------------------------
# 8. SAVE RESULTS
# -----------------------------------------

comparison_df.to_csv(
    "exports/model_comparison.csv",
    index=False
)

print("\nComparison saved to exports/model_comparison.csv")
print("\nModel evaluation completed!")
