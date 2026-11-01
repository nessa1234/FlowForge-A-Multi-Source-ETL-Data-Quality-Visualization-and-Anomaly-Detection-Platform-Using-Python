
import pandas as pd

from anomaly_detection import (
    detect_anomalies,
    detect_anomalies_lof,
    detect_anomalies_svm
)


# ==========================================
# LOAD DATASET
# ==========================================

file_path = (
    r"C:\Users\sarat\Downloads\online+retail"
    r"\Online Retail.xlsx"
)

df = pd.read_excel(file_path)

print("\nRETAIL DATASET")
print("----------------------------")
print("Total records:", len(df))


# ==========================================
# MODEL 1: ISOLATION FOREST
# ==========================================

print("\nRunning Isolation Forest...")

if_results = detect_anomalies(df)

if_anomalies = (
    if_results["Anomaly"] == -1
).sum()

print("\nISOLATION FOREST RESULTS")
print("----------------------------")
print("Records analysed:", len(if_results))
print("Anomalies detected:", if_anomalies)
print("Normal records:", len(if_results) - if_anomalies)


# ==========================================
# MODEL 2: LOCAL OUTLIER FACTOR
# ==========================================

print("\nRunning Local Outlier Factor...")

lof_results = detect_anomalies_lof(df)

lof_anomalies = (
    lof_results["Anomaly"] == -1
).sum()

print("\nLOCAL OUTLIER FACTOR RESULTS")
print("----------------------------")
print("Records analysed:", len(lof_results))
print("Anomalies detected:", lof_anomalies)
print("Normal records:", len(lof_results) - lof_anomalies)


# ==========================================
# DISPLAY SAMPLE ANOMALIES
# ==========================================

columns_to_display = [
    "InvoiceNo",
    "StockCode",
    "Quantity",
    "UnitPrice",
    "Anomaly_Score",
    "Anomaly_Status"
]

print("\nISOLATION FOREST SAMPLE ANOMALIES")

print(
    if_results[
        if_results["Anomaly_Status"] == "Anomaly"
    ][columns_to_display].head(10)
)

print("\nLOF SAMPLE ANOMALIES")

print(
    lof_results[
        lof_results["Anomaly_Status"] == "Anomaly"
    ][columns_to_display].head(10)
)

print("\nBoth anomaly detection models completed successfully!")

# ==========================================
# MODEL 3: ONE-CLASS SVM
# ==========================================

print("\nRunning One-Class SVM...")

svm_results = detect_anomalies_svm(df)

svm_anomalies = (
    svm_results["Anomaly"] == -1
).sum()

print("\nONE-CLASS SVM RESULTS")
print("----------------------------")

print("Records analysed:", len(svm_results))
print("Anomalies detected:", svm_anomalies)
print("Normal records:", len(svm_results) - svm_anomalies)

print("\nONE-CLASS SVM SAMPLE ANOMALIES")

print(
    svm_results[
        svm_results["Anomaly_Status"] == "Anomaly"
    ][
        ["InvoiceNo", "StockCode", "Quantity",
         "UnitPrice", "Anomaly_Score", "Anomaly_Status"]
    ].head(10)
)