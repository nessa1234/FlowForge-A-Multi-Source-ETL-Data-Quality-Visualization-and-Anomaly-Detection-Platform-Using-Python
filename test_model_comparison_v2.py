
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================
# 1. LOAD DATASET
# =========================================

file_path = r"C:\Users\sarat\Downloads\online+retail\Online Retail.xlsx"

df = pd.read_excel(file_path)

print("Original dataset:", df.shape)

# =========================================
# 2. PREPARE FEATURES
# =========================================

features = ["Quantity", "UnitPrice"]

data = df[features].copy()

for column in features:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

print("Valid records:", len(data))

# =========================================
# 3. SPLIT TRAINING AND TEST DATA
# =========================================

X_train, X_test = train_test_split(
    data,
    test_size=0.20,
    random_state=42
)

# Use a manageable training sample for all models
X_train = X_train.sample(
    n=min(20000, len(X_train)),
    random_state=42
).copy()

# Use a manageable testing sample
X_test = X_test.sample(
    n=min(19600, len(X_test)),
    random_state=42
).copy()

print("\nDATA SPLIT")
print("----------------------------")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# =========================================
# 4. INJECT ANOMALIES INTO TEST DATA ONLY
# =========================================

test_data = X_test.copy()

test_data["Actual_Anomaly"] = 0

rng = np.random.default_rng(42)

anomaly_count = 400

anomaly_indices = rng.choice(
    test_data.index.to_numpy(),
    size=min(anomaly_count, len(test_data)),
    replace=False
)

test_data.loc[anomaly_indices, "Quantity"] = (
    rng.choice([-1, 1], size=len(anomaly_indices))
    * rng.integers(
        1000,
        5000,
        size=len(anomaly_indices)
    )
)

test_data.loc[anomaly_indices, "UnitPrice"] = (
    rng.uniform(
        500,
        2000,
        size=len(anomaly_indices)
    )
)

test_data.loc[anomaly_indices, "Actual_Anomaly"] = 1

y_true = test_data["Actual_Anomaly"].to_numpy()

print("\nTEST DATA")
print("----------------------------")
print("Total records:", len(test_data))
print("Injected anomalies:", int(y_true.sum()))
print("Normal records:", int((y_true == 0).sum()))

# =========================================
# 5. SCALE USING TRAINING DATA
# =========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train[features]
)

X_test_scaled = scaler.transform(
    test_data[features]
)

# =========================================
# 6. ISOLATION FOREST
# =========================================

print("\nTraining Isolation Forest...")

if_model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)

if_model.fit(X_train_scaled)

if_predictions = if_model.predict(X_test_scaled)

# =========================================
# 7. LOCAL OUTLIER FACTOR
# =========================================

print("Training Local Outlier Factor...")

lof_model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.02,
    novelty=True,
    n_jobs=-1
)

lof_model.fit(X_train_scaled)

lof_predictions = lof_model.predict(X_test_scaled)

# =========================================
# 8. ONE-CLASS SVM
# =========================================

print("Training One-Class SVM...")

svm_model = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.02
)

svm_model.fit(X_train_scaled)

svm_predictions = svm_model.predict(X_test_scaled)

# =========================================
# 9. MODEL EVALUATION
# =========================================

predictions = {
    "Isolation Forest": (if_predictions == -1).astype(int),
    "Local Outlier Factor": (lof_predictions == -1).astype(int),
    "One-Class SVM": (svm_predictions == -1).astype(int)
}

results = []

for model_name, y_pred in predictions.items():

    accuracy = accuracy_score(y_true, y_pred)

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

# =========================================
# 10. DISPLAY RESULTS
# =========================================

comparison_df = pd.DataFrame(results)

print("\nIMPROVED MODEL PERFORMANCE COMPARISON")
print("==========================================")

print(comparison_df.to_string(index=False))

# =========================================
# 11. SAVE RESULTS
# =========================================

comparison_df.to_csv(
    "exports/model_comparison_v2.csv",
    index=False
)

print("\nResults saved to exports/model_comparison_v2.csv")
print("\nExperiment 2 completed successfully!")
