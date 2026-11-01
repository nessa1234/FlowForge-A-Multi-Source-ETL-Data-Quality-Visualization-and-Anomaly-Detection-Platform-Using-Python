
import pandas as pd

from anomaly_detection import (
    prepare_anomaly_data,
    run_all_anomaly_models
)

# Load the original retail dataset
df = pd.read_excel(
    r"C:\Users\sarat\Downloads\online+retail\Online Retail.xlsx"
)

# Use a small sample for testing
df_sample = df.sample(
    n=min(5000, len(df)),
    random_state=42
).copy()

print("Original sample shape:", df_sample.shape)

# Prepare features
original_data, X_scaled, feature_names = prepare_anomaly_data(
    df_sample
)

print("Prepared feature shape:", X_scaled.shape)
print("Features:", feature_names)

# Run all three models
results, summary = run_all_anomaly_models(
    original_data,
    X_scaled
)

print("\nModel execution summary:")
print(summary)

print("\nAnomaly counts:")
print(results[
    ["IF_Anomaly", "LOF_Anomaly", "SVM_Anomaly",
     "Combined_Anomaly"]
].sum())

print("\nSample results:")
print(results.head())

# Export results
from anomaly_detection import export_anomaly_results

files = export_anomaly_results(results, summary)

print("\nExported files:")
for name, path in files.items():
    print(name, ":", path)

# Compare model agreement
print("\nMODEL AGREEMENT")
print(
    pd.crosstab(
        results["IF_Anomaly"],
        results["LOF_Anomaly"],
        rownames=["Isolation Forest"],
        colnames=["LOF"]
    )
)

print("\nMODEL FLAG COMBINATIONS")
print(
    results.groupby(
        ["IF_Anomaly", "LOF_Anomaly", "SVM_Anomaly"]
    ).size().reset_index(name="Record_Count")
)

# Inspect records flagged by majority voting
print("\nTOP 10 DETECTED ANOMALIES")

anomaly_review = results[
    results["Combined_Anomaly"] == 1
].copy()

columns_to_review = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "UnitPrice",
    "IF_Anomaly",
    "LOF_Anomaly",
    "SVM_Anomaly",
    "Models_Flagged"
]

print(
    anomaly_review[columns_to_review]
    .head(10)
    .to_string(index=False)
)
