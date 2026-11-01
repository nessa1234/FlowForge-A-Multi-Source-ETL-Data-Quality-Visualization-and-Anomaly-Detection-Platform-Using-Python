
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
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


# =========================================
# 1. LOAD DATA
# =========================================

file_path = (
    r"C:\Users\sarat\Downloads\online+retail"
    r"\Online Retail.xlsx"
)

print("Loading Online Retail dataset...")

df = pd.read_excel(file_path)

print(f"Original dataset shape: {df.shape}")


# =========================================
# 2. TRAIN / TEST SPLIT
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

print(f"Training records: {len(train_df)}")
print(f"Testing records: {len(test_df)}")


# =========================================
# 3. INJECT ARTIFICIAL ANOMALIES
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

print(f"Injected anomalies: {int(y_true.sum())}")


# =========================================
# 4. FEATURE ENGINEERING
# =========================================

print("\nEngineering retail features...")

train_engineered = engineer_retail_features(train_df)
test_engineered = engineer_retail_features(test_df)

train_features = get_retail_model_features(
    train_engineered
).copy()

test_features = get_retail_model_features(
    test_engineered
).copy()

# Ensure identical feature columns and order.
test_features = test_features.reindex(
    columns=train_features.columns
)

# Convert features to numeric values.
train_features = train_features.apply(
    pd.to_numeric, errors="coerce"
)

test_features = test_features.apply(
    pd.to_numeric, errors="coerce"
)

# IMPORTANT:
# Replace infinite values BEFORE calculating medians.
train_features = train_features.replace(
    [np.inf, -np.inf], np.nan
)

test_features = test_features.replace(
    [np.inf, -np.inf], np.nan
)

# Calculate medians using training data only.
medians = train_features.median()

# Remove features that have no usable training median.
valid_columns = medians[
    medians.notna()
].index.tolist()

train_features = train_features[valid_columns]
test_features = test_features[valid_columns]
medians = medians[valid_columns]

# Impute missing values using training medians.
train_features = train_features.fillna(medians)
test_features = test_features.fillna(medians)

# Check for numerical problems before scaling.
if not np.isfinite(train_features.to_numpy()).all():
    raise ValueError(
        "Training features still contain infinite or invalid values."
    )

if not np.isfinite(test_features.to_numpy()).all():
    raise ValueError(
        "Test features still contain infinite or invalid values."
    )

print(f"Features used: {len(valid_columns)}")
print("Feature names:", valid_columns)


# =========================================
# 5. SCALE FEATURES
# =========================================

print("\nScaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(train_features)
X_test = scaler.transform(test_features)

# =========================================
# DIAGNOSE EXTREME FEATURE VALUES
# =========================================

print("\nFEATURE DIAGNOSTICS")
print("----------------------------")

print("\nTraining feature ranges BEFORE scaling:")
print(
    train_features.describe()
    .loc[["min", "50%", "max"]]
    .rename(index={"50%": "median"})
    .T
)
pd.DataFrame(X_train, columns=valid_columns).describe().loc[
    ["min", "50%", "max"]
].rename(index={"50%": "median"}).T

print("\nLargest absolute scaled training value:")
print(np.max(np.abs(X_train)))

print("\nLargest absolute scaled test value:")
print(np.max(np.abs(X_test)))

# Identify the feature responsible for extreme scaled values

print("\nEXTREME FEATURE DIAGNOSTICS")
print("----------------------------")

train_max = pd.Series(
    np.max(np.abs(X_train), axis=0),
    index=valid_columns
)

test_max = pd.Series(
    np.max(np.abs(X_test), axis=0),
    index=valid_columns
)

feature_check = pd.DataFrame({
    "Training_Max_Absolute_Scaled": train_max,
    "Test_Max_Absolute_Scaled": test_max
})

print(
    feature_check.sort_values(
        "Test_Max_Absolute_Scaled",
        ascending=False
    ).to_string()
)

print("\nFEATURE SCALER INFORMATION")
print("----------------------------")

for name, mean, scale in zip(
    valid_columns,
    scaler.mean_,
    scaler.scale_
):
    print(
        f"{name}: mean={mean:.8f}, "
        f"scale={scale:.8f}"
    )

print("\nSmallest LOF training neighbourhood distance:")

from sklearn.neighbors import NearestNeighbors

neighbours = NearestNeighbors(
    n_neighbors=21,
    n_jobs=-1
)

neighbours.fit(X_train)

distances, indices = neighbours.kneighbors(X_train)

print(
    "Minimum distance to a neighbour:",
    distances[:, 1:].min()
)

print(
    "Number of zero-distance neighbours:",
    int(np.sum(distances[:, 1:] == 0))
)

# Check scaled data.
print(
    "Training values finite:",
    np.isfinite(X_train).all()
)

print(
    "Test values finite:",
    np.isfinite(X_test).all()
)

if not np.isfinite(X_train).all():
    raise ValueError(
        "Training data contains invalid values after scaling."
    )

if not np.isfinite(X_test).all():
    raise ValueError(
        "Test data contains invalid values after scaling."
    )


# =========================================
# 6. TRAIN LOCAL OUTLIER FACTOR
# =========================================

print("\nTraining Local Outlier Factor...")

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.02,
    novelty=True,
    n_jobs=-1
)

lof.fit(X_train)

print("LOF training completed.")


# =========================================
# 7. CALCULATE SCORES
# =========================================

# With novelty=True:
# Lower decision_function values are more anomalous.
# Values below zero are classified as anomalies
# using LOF's default decision threshold.

scores = lof.decision_function(X_test)

predictions = lof.predict(X_test)

# sklearn returns -1 for anomalies and 1 for normal records.
predictions = (predictions == -1).astype(int)

diagnostics = pd.DataFrame({
    "Actual_Anomaly": y_true,
    "LOF_Decision_Score": scores,
    "LOF_Prediction": predictions
})

diagnostics["Actual_Status"] = diagnostics[
    "Actual_Anomaly"
].map({
    0: "Normal",
    1: "Injected Anomaly"
})

diagnostics["Predicted_Status"] = diagnostics[
    "LOF_Prediction"
].map({
    0: "Normal",
    1: "Predicted Anomaly"
})


# =========================================
# 8. CHECK SCORE DISTRIBUTION
# =========================================

print("\nLOF SCORE DISTRIBUTION")
print("----------------------------")

print(
    diagnostics.groupby("Actual_Status")[
        "LOF_Decision_Score"
    ].describe()
)

print("\nSCORE RANGE")
print("----------------------------")

print("Minimum score:", np.min(scores))
print("Maximum score:", np.max(scores))
print("Mean score:", np.mean(scores))

print(
    "Number of non-finite scores:",
    int((~np.isfinite(scores)).sum())
)


# =========================================
# 9. EVALUATE PREDICTIONS
# =========================================

print("\nPREDICTION COUNTS")
print("----------------------------")

print(
    pd.crosstab(
        diagnostics["Actual_Status"],
        diagnostics["LOF_Prediction"],
        rownames=["Actual Status"],
        colnames=["Predicted Anomaly"]
    )
)

precision = precision_score(
    y_true, predictions, zero_division=0
)

recall = recall_score(
    y_true, predictions, zero_division=0
)

f1 = f1_score(
    y_true, predictions, zero_division=0
)

print("\nLOF EVALUATION METRICS")
print("----------------------------")

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nCONFUSION MATRIX")
print("----------------------------")

print(
    confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )
)


# =========================================
# 10. TOP 20 MOST ANOMALOUS RECORDS
# =========================================

print("\nTOP 20 MOST ANOMALOUS RECORDS")
print("----------------------------")

print(
    diagnostics.nsmallest(
        20,
        "LOF_Decision_Score"
    )[
        [
            "Actual_Anomaly",
            "LOF_Decision_Score",
            "Actual_Status",
            "LOF_Prediction"
        ]
    ].to_string(index=False)
)


# =========================================
# 11. SAVE RESULTS
# =========================================

os.makedirs("exports", exist_ok=True)

output_path = os.path.join(
    "exports",
    "lof_diagnostic_results.csv"
)

diagnostics.to_csv(
    output_path,
    index=False
)

print(f"\nSaved: {output_path}")
print("\nLOF diagnostic completed!")
