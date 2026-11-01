
import pandas as pd


def run_data_quality_analysis(df):
    """Profile the uploaded dataset without modifying it."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing_cells = int(df.isna().sum().sum())
    missing_rows = int(df.isna().any(axis=1).sum())

    duplicate_mask = df.duplicated(keep="first")
    duplicate_records = df.loc[duplicate_mask].copy()

    total_cells = df.shape[0] * df.shape[1]

    completeness = (
        round((1 - missing_cells / total_cells) * 100, 2)
        if total_cells else 100.0
    )

    null_report = pd.DataFrame({
        "Column": df.columns,
        "Missing_Count": [int(df[c].isna().sum()) for c in df.columns],
        "Missing_Percentage": [
            round(df[c].isna().mean() * 100, 2)
            for c in df.columns
        ]
    })

    data_types = pd.DataFrame({
        "Column": df.columns,
        "Data_Type": [str(df[c].dtype) for c in df.columns],
        "Non_Null_Count": [int(df[c].notna().sum()) for c in df.columns],
        "Missing_Count": [int(df[c].isna().sum()) for c in df.columns]
    })

    retail_rule_findings = {}

    if "Quantity" in df.columns:
        quantity = pd.to_numeric(df["Quantity"], errors="coerce")
        retail_rule_findings["negative_quantity_returns"] = (
            df.loc[quantity < 0].copy()
        )

    if "UnitPrice" in df.columns:
        price = pd.to_numeric(df["UnitPrice"], errors="coerce")
        retail_rule_findings["zero_or_negative_unit_price"] = (
            df.loc[price <= 0].copy()
        )

    if {"Quantity", "UnitPrice"}.issubset(df.columns):
        quantity = pd.to_numeric(df["Quantity"], errors="coerce")
        price = pd.to_numeric(df["UnitPrice"], errors="coerce")

        retail_rule_findings["high_transaction_value"] = (
            df.loc[(quantity * price).abs() > 1000].copy()
        )

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_cells": missing_cells,
        "missing_rows": missing_rows,
        "duplicate_count": int(duplicate_mask.sum()),
        "completeness": completeness,
        "null_report": null_report,
        "duplicate_records": duplicate_records,
        "data_types": data_types,
        "retail_rule_findings": retail_rule_findings,
        "dataset_preview": df.head(10).copy()
    }


def build_cleaning_preview(
    df,
    duplicate_action="keep",
    missing_action="keep"
):
    """Create a proposed cleaning result without changing the original."""

    valid_actions = {"keep", "remove", "quarantine"}

    if duplicate_action not in valid_actions:
        raise ValueError("Invalid duplicate action.")

    if missing_action not in valid_actions:
        raise ValueError("Invalid missing-value action.")

    original = df.copy()
    working = original.copy()

    quarantine_parts = []
    removed_count = 0

    # Handle duplicate records
    duplicate_mask = working.duplicated(keep="first")

    if duplicate_action == "remove":
        removed_count += int(duplicate_mask.sum())
        working = working.loc[~duplicate_mask].copy()

    elif duplicate_action == "quarantine":
        duplicate_rows = working.loc[duplicate_mask].copy()
        duplicate_rows["Quarantine_Reason"] = "Duplicate record"

        if not duplicate_rows.empty:
            quarantine_parts.append(duplicate_rows)

        working = working.loc[~duplicate_mask].copy()

    # Handle missing values
    missing_mask = working.isna().any(axis=1)

    if missing_action == "remove":
        removed_count += int(missing_mask.sum())
        working = working.loc[~missing_mask].copy()

    elif missing_action == "quarantine":
        missing_rows = working.loc[missing_mask].copy()
        missing_rows["Quarantine_Reason"] = "Missing values"

        if not missing_rows.empty:
            quarantine_parts.append(missing_rows)

        working = working.loc[~missing_mask].copy()

    if quarantine_parts:
        quarantine = pd.concat(
            quarantine_parts,
            ignore_index=True,
            sort=False
        )
    else:
        quarantine = pd.DataFrame(
            columns=list(original.columns) + ["Quarantine_Reason"]
        )

    summary = {
        "original_records": len(original),
        "records_removed": removed_count,
        "records_quarantined": len(quarantine),
        "records_after_preview": len(working),
        "duplicate_action": duplicate_action,
        "missing_action": missing_action
    }

    return {
        "cleaned_preview": working.copy(),
        "quarantine_preview": quarantine.copy(),
        "summary": summary
    }
