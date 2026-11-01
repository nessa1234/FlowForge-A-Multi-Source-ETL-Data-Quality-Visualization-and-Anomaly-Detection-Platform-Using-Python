
import pandas as pd

from feature_engineering import (
    engineer_retail_features,
    get_retail_model_features
)

file_path = r"C:\Users\sarat\Downloads\online+retail\Online Retail.xlsx"

df = pd.read_excel(file_path)

print("ORIGINAL DATASET")
print("----------------------------")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# Engineer new features
df_engineered = engineer_retail_features(df)

print("\nENGINEERED DATASET")
print("----------------------------")
print("Rows:", len(df_engineered))
print("Columns:", len(df_engineered.columns))

new_features = [
    "TransactionValue",
    "AbsoluteQuantity",
    "LogQuantity",
    "LogUnitPrice",
    "LogTransactionValue",
    "IsReturn",
    "IsZeroPrice"
]

print("\nNEW FEATURES")
print("----------------------------")
print(df_engineered[new_features].head(10))

# Prepare model input
model_data = get_retail_model_features(
    df_engineered
)

print("\nMODEL INPUT")
print("----------------------------")
print("Shape:", model_data.shape)
print(model_data.head())

# Verify original dataset is unchanged
print(
    "\nOriginal dataset unchanged:",
    list(df.columns) == [
        col for col in df_engineered.columns
        if col in df.columns
    ]
    and df.equals(df_engineered[df.columns])
)

print("\nFeature engineering completed successfully!")
