
import pandas as pd
import numpy as np


def engineer_retail_features(df):

    required_columns = ["Quantity", "UnitPrice"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    result = df.copy()

    result["Quantity"] = pd.to_numeric(
        result["Quantity"], errors="coerce"
    )

    result["UnitPrice"] = pd.to_numeric(
        result["UnitPrice"], errors="coerce"
    )

    result[required_columns] = result[
        required_columns
    ].replace([np.inf, -np.inf], np.nan)

    # Transaction features
    result["TransactionValue"] = (
        result["Quantity"] * result["UnitPrice"]
    )

    result["AbsoluteQuantity"] = result["Quantity"].abs()

    # Log transformations
    result["LogQuantity"] = np.log1p(
        result["AbsoluteQuantity"]
    )

    result["LogUnitPrice"] = np.log1p(
        result["UnitPrice"].abs()
    )

    result["LogTransactionValue"] = np.log1p(
        result["TransactionValue"].abs()
    )

    # Business indicators
    result["IsReturn"] = (
        result["Quantity"] < 0
    ).fillna(False).astype("int8")

    result["IsZeroPrice"] = (
        result["UnitPrice"] == 0
    ).fillna(False).astype("int8")

    return result


def get_retail_model_features(df):

    feature_columns = [
        "Quantity",
        "UnitPrice",
        "TransactionValue",
        "LogQuantity",
        "LogUnitPrice",
        "LogTransactionValue",
        "IsReturn",
        "IsZeroPrice"
    ]

    missing_columns = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Engineer features first. Missing: {missing_columns}"
        )

    model_data = df[feature_columns].copy()

    model_data = model_data.replace(
        [np.inf, -np.inf], np.nan
    )

    return model_data
