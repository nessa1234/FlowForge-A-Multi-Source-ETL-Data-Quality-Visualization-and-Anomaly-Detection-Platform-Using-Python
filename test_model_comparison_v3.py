
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

from feature_engineering import (
    engineer_retail_features,
    get_retail_model_features
)

# =========================================
# 1. LOAD RETAIL DATA
# =========================================

file_path = r"C:\Users\sarat\Downloads\online+retail\Online Retail.xlsx"

df = pd.read_excel(file_path)

print("Original dataset:", df.shape)

# =========================================
# 2. SPLIT ORIGINAL DATA
# =========================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)

train_df = train_df.sample(
    n=min(20000, len(train_df)),
    random_state=42
).copy()

test_df = test_df.sample(
    n=min(19600, len(test_df)),
    random_state=42
).copy()

print("\nDATA SPLIT")
print("----------------------------")
print("Training records:", len(train_df))
print("Testing records:", len(test_df))

# =========================================
# 3. INJECT ANOMALIES INTO TEST DATA
# =========================================

test_df["Actual_Anomaly"] = 0

rng = np.random.default_rng(42)

anomaly_count = min(400, len(test_df))

anomaly_indices = rng.choice(
    test_df.index.to_numpy(),
    size=anomaly_count,
    replace=False
)

test_df.loc[anomaly_indices, "Quantity"] = (
    rng.choice([-1, 1], size=anomaly_count)
    * rng.integers(1000, 5000, size=anomaly_count)
)

test_df.loc[anomaly_indices, "UnitPrice"] = (
    rng.uniform(500, 2000, size=anomaly_count)
)

test_df.loc[anomaly_indices, "Actual_Anomaly"] = 1

y_true = test_df["Actual_Anomaly"].to_numpy()

print("\nINJECTED ANOMALIES")
print("----------------------------")
print("Known anomalies:", int(y_true.sum()))
print("Normal records:", int((y_true == 0).sum()))

# =========================================
# 4. ENGINEER FEATURES
# =========================================

train_engineered = engineer_retail_features(train_df)

test_engineered = engineer_retail_features(test_df)

X_train = get_retail_model_features(train_engineered)

X_test = get_retail_model_features(test_engineered)

# Handle any invalid feature values
X_train = X_train.replace(
    [np.inf, -np.inf], np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf], np.nan
)

# Use training medians for missing values
training_medians = X_train.median()

X_train = X_train.fillna(training_medians)
X_test = X_test.fillna(training_medians)

print("\nFEATURE ENGINEERING")
print("----------------------------")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

# =========================================
# 5. SCALE FEATURES
# =========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# =========================================
# 6. ISOLATION FOREST
# =========================================

print("\nRunning Isolation Forest...")

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

print("Running Local Outlier Factor...")

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

print("Running One-Class SVM...")

svm_model = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.02
)

svm_model.fit(X_train_scaled)

svm_predictions = svm_model.predict(X_test_scaled)

# =========================================
# 9. EVALUATE MODELS
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
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
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

comparison_df = pd.DataFrame(results)

# =========================================
# 10. DISPLAY AND SAVE RESULTS
# =========================================

print("\nFEATURE-ENGINEERED MODEL COMPARISON")
print("==========================================")

print(comparison_df.to_string(index=False))

comparison_df.to_csv(
    "exports/model_comparison_v3.csv",
    index=False
)

print("\nSaved: exports/model_comparison_v3.csv")
print("\nExperiment 3 completed successfully!")
