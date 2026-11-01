
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================
# 1. LOAD DATA
# =========================================

file_path = (
    r"C:\Users\sarat\Downloads\online+retail"
    r"\Online Retail.xlsx"
)

print("Loading retail dataset...")

df = pd.read_excel(file_path)

print("Dataset shape:", df.shape)

# =========================================
# 2. SPLIT DATA
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

# =========================================
# 3. INJECT KNOWN ANOMALIES
# =========================================

test_df["Actual_Anomaly"] = 0

rng = np.random.default_rng(42)

n_anomalies = min(400, len(test_df))

indices = rng.choice(
    test_df.index.to_numpy(),
    size=n_anomalies,
    replace=False
)

test_df.loc[indices, "Quantity"] = (
    rng.choice([-1, 1], size=n_anomalies)
    * rng.integers(1000, 5000, size=n_anomalies)
)

test_df.loc[indices, "UnitPrice"] = (
    rng.uniform(500, 2000, size=n_anomalies)
)

test_df.loc[indices, "Actual_Anomaly"] = 1

y_true = test_df["Actual_Anomaly"].to_numpy()

print("Injected anomalies:", int(y_true.sum()))

# =========================================
# 4. CREATE STABLE FEATURES
# =========================================

def create_features(data):
    quantity = pd.to_numeric(
        data["Quantity"], errors="coerce"
    )

    price = pd.to_numeric(
        data["UnitPrice"], errors="coerce"
    )

    result = pd.DataFrame(index=data.index)

    result["LogQuantity"] = np.log1p(
        quantity.abs()
    )

    result["LogUnitPrice"] = np.log1p(
        price.abs()
    )

    result["LogTransactionValue"] = np.log1p(
        (quantity * price).abs()
    )

    result["IsReturn"] = (
        quantity < 0
    ).astype(float)

    result["IsZeroPrice"] = (
        price == 0
    ).astype(float)

    return result.replace(
        [np.inf, -np.inf], np.nan
    )


X_train_df = create_features(train_df)
X_test_df = create_features(test_df)

medians = X_train_df.median()

X_train_df = X_train_df.fillna(medians)
X_test_df = X_test_df.fillna(medians)

# Learn clipping thresholds from training data only.
lower = X_train_df.quantile(0.01)
upper = X_train_df.quantile(0.99)

X_train_df = X_train_df.clip(
    lower=lower,
    upper=upper,
    axis=1
)

X_test_df = X_test_df.clip(
    lower=lower,
    upper=upper,
    axis=1
)

# =========================================
# 5. SCALE FEATURES
# =========================================

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train_df)
X_test = scaler.transform(X_test_df)

print(
    "Maximum absolute scaled training value:",
    np.max(np.abs(X_train))
)

print(
    "Maximum absolute scaled test value:",
    np.max(np.abs(X_test))
)

# =========================================
# 6. TRAIN LOF
# =========================================

print("\nTraining LOF...")

model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.02,
    novelty=True,
    n_jobs=-1
)
# Check for duplicate feature vectors
import numpy as np

print("\nDUPLICATE FEATURE CHECK")
print("Training matrix shape:", X_train.shape)
print("Unique training rows:", np.unique(X_train, axis=0).shape[0])
print("Duplicate training rows:",
      X_train.shape[0] - np.unique(X_train, axis=0).shape[0])

print("Training matrix contains NaN:", np.isnan(X_train).any())
print("Training matrix contains infinity:", np.isinf(X_train).any())

print("First 5 training rows:")
print(X_train[:5])

# Remove duplicate feature vectors from LOF training only
X_train_unique = np.unique(X_train, axis=0)

print("\nLOF TRAINING DATA")
print("Original training rows:", len(X_train))
print("Unique training rows:", len(X_train_unique))
print("Removed duplicate feature vectors:", 
      len(X_train) - len(X_train_unique))

# Train LOF using unique feature vectors
model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.02,
    novelty=True,
    n_jobs=-1
)

model.fit(X_train_unique)
# Diagnose LOF's internal scores
# Inspect nearest-neighbour distances
distances, indices = model.kneighbors(X_train[:100])

print("\nNEIGHBOUR DISTANCE CHECK")
print("Minimum distance:", distances.min())
print("Maximum distance:", distances.max())
print("Mean distance:", distances.mean())
print("Zero-distance neighbours:", (distances == 0).sum())
print("Total neighbour distances:", distances.size)

print("First 5 distance rows:")
print(distances[:5])
import numpy as np

print("\nLOF INTERNAL DIAGNOSTICS")
print("Scikit-learn version:", __import__("sklearn").__version__)

train_scores = model.negative_outlier_factor_

print("Training scores finite:", np.isfinite(train_scores).all())
print("Training score minimum:", np.min(train_scores))
print("Training score maximum:", np.max(train_scores))
print("Training score mean:", np.mean(train_scores))

print("Number of infinite training scores:",
      np.isinf(train_scores).sum())

print("Number of NaN training scores:",
      np.isnan(train_scores).sum())

print("Neighbour count:", model.n_neighbors_)
# =========================================
# 7. PREDICT AND EVALUATE
# =========================================

predictions = (
    model.predict(X_test) == -1
).astype(int)

scores = model.decision_function(X_test)

print("\nLOF SCORE RANGE")
print("Minimum:", np.min(scores))
print("Maximum:", np.max(scores))
print("Mean:", np.mean(scores))

print("\nPREDICTION COUNTS")
print(pd.Series(predictions).value_counts())

print("\nEVALUATION")
print("Precision:", precision_score(
    y_true, predictions, zero_division=0
))

print("Recall:", recall_score(
    y_true, predictions, zero_division=0
))

print("F1-score:", f1_score(
    y_true, predictions, zero_division=0
))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_true, predictions))

# =========================================
# 8. SAVE RESULTS
# =========================================

os.makedirs("exports", exist_ok=True)

results = pd.DataFrame({
    "Actual_Anomaly": y_true,
    "LOF_Score": scores,
    "LOF_Prediction": predictions
})

results.to_csv(
    "exports/lof_clean_results.csv",
    index=False
)

print("\nSaved: exports/lof_clean_results.csv")
print("Experiment completed.")
