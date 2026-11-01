
import pandas as pd
import numpy as np

# ==========================================
# SCIKIT-LEARN IMPORTS
# ==========================================

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)

# Import our FlowForge feature engineering module
from feature_engineering import (
    engineer_retail_features,
    get_retail_model_features
)


print("======================================")
print("FLOWFORGE - MODEL COMPARISON")
print("======================================")


# ==========================================
# 1. LOAD RETAIL DATASET
# ==========================================

DATA_PATH = r"C:\Users\sarat\Downloads\online+retail\Online Retail.xlsx"

print("\nLoading retail dataset...")

df = pd.read_excel(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. TRAIN-TEST SPLIT
# ==========================================

print("\nPreparing training and testing data...")

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Training records:", len(train_df))
print("Testing records:", len(test_df))


# ==========================================
# 3. INJECT ARTIFICIAL ANOMALIES
# ==========================================

print("\nInjecting artificial anomalies...")

rng = np.random.default_rng(42)

test_df["Actual_Anomaly"] = 0

anomaly_count = 400

anomaly_indices = rng.choice(
    test_df.index,
    size=anomaly_count,
    replace=False
)

test_df.loc[anomaly_indices, "Quantity"] = (
    rng.choice([-1, 1], size=anomaly_count)
    * rng.integers(1000, 5001, size=anomaly_count)
)

test_df.loc[anomaly_indices, "UnitPrice"] = rng.uniform(
    500,
    2000,
    size=anomaly_count
)

test_df.loc[anomaly_indices, "Actual_Anomaly"] = 1

print("Injected anomalies:", test_df["Actual_Anomaly"].sum())

print("\nAnomaly distribution:")
print(test_df["Actual_Anomaly"].value_counts())


# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================

print("\nEngineering retail features...")

train_features = engineer_retail_features(train_df)

test_features = engineer_retail_features(test_df)

X_train = get_retail_model_features(train_features)

X_test = get_retail_model_features(test_features)

# Preserve actual anomaly labels
y_test = test_df["Actual_Anomaly"].to_numpy()

print("Training feature shape:", X_train.shape)

print("Testing feature shape:", X_test.shape)

print("\nFeatures used:")
print(X_train.columns.tolist())


# ==========================================
# 5. DATA PREPROCESSING
# ==========================================

print("\nPreparing features for machine learning...")

# Replace infinite values
X_train = X_train.replace(
    [np.inf, -np.inf], np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf], np.nan
)

# Impute missing values using training medians
imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)

X_test_imputed = imputer.transform(X_test)

# Scale features
scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)

X_test_scaled = scaler.transform(X_test_imputed)


print("\nPreprocessing completed!")

print("Training matrix:", X_train_scaled.shape)

print("Testing matrix:", X_test_scaled.shape)

print(
    "Maximum absolute training value:",
    np.abs(X_train_scaled).max()
)

print(
    "Maximum absolute testing value:",
    np.abs(X_test_scaled).max()
)

print(
    "Missing values in training:",
    np.isnan(X_train_scaled).sum()
)

print(
    "Missing values in testing:",
    np.isnan(X_test_scaled).sum()
)


# ==========================================
# 6. DATA PREPARATION SUMMARY
# ==========================================

print("\n======================================")
print("DATA PREPARATION COMPLETED")
print("======================================")

print("Training records:", len(X_train_scaled))

print("Testing records:", len(X_test_scaled))

print("Model features:", X_train_scaled.shape[1])

print("Injected anomalies:", int(y_test.sum()))

print("\nReady for model training!")


# ==========================================
# 7. CREATE COMMON MODEL TRAINING AND TEST SAMPLES
# ==========================================

print("\nCreating model training and evaluation samples...")

rng = np.random.default_rng(42)

# Select 5,000 training records
train_sample_size = 5000

train_indices = rng.choice(
    len(X_train_scaled),
    size=train_sample_size,
    replace=False
)

X_train_sample = X_train_scaled[train_indices]


# Select all 400 injected anomalies
anomaly_test_indices = np.where(y_test == 1)[0]

# Select 5,000 normal test records
normal_test_indices = np.where(y_test == 0)[0]

normal_sample_indices = rng.choice(
    normal_test_indices,
    size=5000,
    replace=False
)

# Combine normal and artificial anomaly records
evaluation_indices = np.concatenate([
    normal_sample_indices,
    anomaly_test_indices
])

# Shuffle evaluation records
rng.shuffle(evaluation_indices)

X_test_sample = X_test_scaled[evaluation_indices]
y_test_sample = y_test[evaluation_indices]


print("Training sample:", X_train_sample.shape)
print("Evaluation sample:", X_test_sample.shape)

print("\nEvaluation label distribution:")
print("Normal records:", np.sum(y_test_sample == 0))
print("Anomalies:", np.sum(y_test_sample == 1))

print("\nCommon model samples are ready!")

# ==========================================
# 8. TRAIN ANOMALY DETECTION MODELS
# ==========================================

print("\n======================================")
print("TRAINING ANOMALY DETECTION MODELS")
print("======================================")


# ------------------------------------------
# MODEL 1: ISOLATION FOREST
# ------------------------------------------

print("\nTraining Isolation Forest...")

isolation_forest = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)

isolation_forest.fit(X_train_sample)

if_predictions = isolation_forest.predict(X_test_sample)

# Convert -1 (anomaly) and 1 (normal) into 1 and 0
if_predictions = np.where(if_predictions == -1, 1, 0)

print("Isolation Forest completed!")


# ------------------------------------------
# MODEL 2: LOCAL OUTLIER FACTOR
# ------------------------------------------

print("\nTraining Local Outlier Factor...")

# Remove duplicate feature vectors for LOF training
X_train_lof = np.unique(
    X_train_sample,
    axis=0
)

print("LOF training records:", len(X_train_lof))

lof_model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.02,
    novelty=True,
    n_jobs=-1
)

lof_model.fit(X_train_lof)

lof_predictions = lof_model.predict(X_test_sample)

lof_predictions = np.where(
    lof_predictions == -1, 1, 0
)

print("Local Outlier Factor completed!")


# ------------------------------------------
# MODEL 3: ONE-CLASS SVM
# ------------------------------------------

print("\nTraining One-Class SVM...")

one_class_svm = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.02
)

one_class_svm.fit(X_train_sample)

svm_predictions = one_class_svm.predict(X_test_sample)

svm_predictions = np.where(
    svm_predictions == -1, 1, 0
)

print("One-Class SVM completed!")


print("\nAll three models have been trained!")

# ==========================================
# 9. MODEL EVALUATION
# ==========================================

print("\n======================================")
print("MODEL PERFORMANCE EVALUATION")
print("======================================")


def evaluate_model(model_name, actual, predicted):

    precision = precision_score(
        actual, predicted, zero_division=0
    )

    recall = recall_score(
        actual, predicted, zero_division=0
    )

    f1 = f1_score(
        actual, predicted, zero_division=0
    )

    accuracy = accuracy_score(
        actual, predicted
    )

    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "True_Positive": tp,
        "False_Positive": fp,
        "False_Negative": fn,
        "True_Negative": tn
    }


# Evaluate each model
results = []

results.append(
    evaluate_model(
        "Isolation Forest",
        y_test_sample,
        if_predictions
    )
)

results.append(
    evaluate_model(
        "Local Outlier Factor",
        y_test_sample,
        lof_predictions
    )
)

results.append(
    evaluate_model(
        "One-Class SVM",
        y_test_sample,
        svm_predictions
    )
)


# Create comparison DataFrame
comparison_df = pd.DataFrame(results)

print("\nMODEL COMPARISON RESULTS")
print(comparison_df.to_string(index=False))


# ==========================================
# 10. SAVE MODEL COMPARISON RESULTS
# ==========================================

comparison_df.to_csv(
    "exports/model_comparison_final.csv",
    index=False
)

print("\nComparison report saved successfully!")
print("File: exports/model_comparison_final.csv")
# ==========================================
# 11. MODEL PERFORMANCE VISUALISATION
# ==========================================

import matplotlib.pyplot as plt

metrics = ["Precision", "Recall", "F1_Score", "Accuracy"]

comparison_plot = comparison_df.set_index("Model")[metrics]

ax = comparison_plot.plot(
    kind="bar",
    figsize=(12, 6),
    ylim=(0, 1.05)
)

plt.title("FlowForge - Anomaly Detection Model Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Performance Score")
plt.xticks(rotation=0)
plt.legend(title="Evaluation Metrics")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "exports/model_performance_comparison.png",
    dpi=300
)

plt.show()

print("Model comparison chart saved!")