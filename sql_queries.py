
import pandas as pd
from sqlalchemy import text

from sql_pipeline import engine


# --------------------------------------------------
# 1. DATASET OVERVIEW
# --------------------------------------------------

def get_dataset_overview():

    query = """
        SELECT
            COUNT(*) AS total_records,
            COUNT(DISTINCT InvoiceNo) AS total_invoices,
            COUNT(DISTINCT StockCode) AS unique_products,
            COUNT(DISTINCT CustomerID) AS unique_customers,
            COUNT(DISTINCT Country) AS countries
        FROM retail_transactions
    """

    with engine.connect() as connection:
        result = pd.read_sql(text(query), connection)

    return result


# --------------------------------------------------
# 2. DATA QUALITY SUMMARY
# --------------------------------------------------

def get_data_quality_summary():

    query = """
        SELECT
            COUNT(*) AS total_records,

            SUM(CASE
                WHEN InvoiceNo IS NULL THEN 1 ELSE 0
            END) AS missing_invoice,

            SUM(CASE
                WHEN StockCode IS NULL THEN 1 ELSE 0
            END) AS missing_stock_code,

            SUM(CASE
                WHEN Description IS NULL THEN 1 ELSE 0
            END) AS missing_description,

            SUM(CASE
                WHEN CustomerID IS NULL THEN 1 ELSE 0
            END) AS missing_customer_id,

            SUM(CASE
                WHEN Quantity IS NULL THEN 1 ELSE 0
            END) AS missing_quantity,

            SUM(CASE
                WHEN UnitPrice IS NULL THEN 1 ELSE 0
            END) AS missing_unit_price,

            SUM(CASE
                WHEN Quantity < 0 THEN 1 ELSE 0
            END) AS return_records,

            SUM(CASE
                WHEN UnitPrice <= 0 THEN 1 ELSE 0
            END) AS invalid_price_records

        FROM retail_transactions
    """

    with engine.connect() as connection:
        result = pd.read_sql(text(query), connection)

    return result


# --------------------------------------------------
# 3. DUPLICATE RECORD ANALYSIS
# --------------------------------------------------

def get_duplicate_summary():

    query = """
        SELECT
            COALESCE(SUM(record_count - 1), 0)
            AS duplicate_records
        FROM (
            SELECT COUNT(*) AS record_count
            FROM retail_transactions
            GROUP BY
                InvoiceNo,
                StockCode,
                Description,
                Quantity,
                InvoiceDate,
                UnitPrice,
                CustomerID,
                Country
            HAVING COUNT(*) > 1
        )
    """

    with engine.connect() as connection:
        result = pd.read_sql(text(query), connection)

    return result
    with engine.connect() as connection:
        result = pd.read_sql(text(query), connection)

    return result


# --------------------------------------------------
# 4. MONTHLY SALES TREND
# --------------------------------------------------

def get_monthly_sales():

    query = """
        SELECT
            SUBSTR(InvoiceDate, 1, 7) AS sales_month,

            SUM(Quantity * UnitPrice) AS total_sales,

            COUNT(DISTINCT InvoiceNo) AS total_invoices,

            SUM(Quantity) AS total_quantity

        FROM retail_transactions

        WHERE InvoiceDate IS NOT NULL

        GROUP BY SUBSTR(InvoiceDate, 1, 7)

        ORDER BY sales_month
    """

    with engine.connect() as connection:
        result = pd.read_sql(text(query), connection)

    return result


# --------------------------------------------------
# 5. COUNTRY-WISE SALES ANALYSIS
# --------------------------------------------------

def get_country_sales():

    query = """
        SELECT
            Country,

            SUM(Quantity * UnitPrice) AS total_sales,

            COUNT(DISTINCT InvoiceNo) AS total_invoices,

            SUM(Quantity) AS total_quantity

        FROM retail_transactions

        WHERE Country IS NOT NULL

        GROUP BY Country

        ORDER BY total_sales DESC
    """

    with engine.connect() as connection:
        result = pd.read_sql(text(query), connection)

    return result


# --------------------------------------------------
# 6. TOP 10 PRODUCTS
# --------------------------------------------------

def get_top_products():

    query = """
        SELECT
            StockCode,
            Description,

            SUM(Quantity) AS total_quantity,

            SUM(Quantity * UnitPrice) AS total_sales

        FROM retail_transactions

        WHERE StockCode IS NOT NULL

        GROUP BY StockCode, Description

        ORDER BY total_sales DESC

        LIMIT 10
    """

    with engine.connect() as connection:
        result = pd.read_sql(text(query), connection)

    return result


# --------------------------------------------------
# 7. SAVE SQL ANALYSIS OUTPUTS
# --------------------------------------------------

def export_sql_analysis():

    import os

    export_dir = "exports"
    os.makedirs(export_dir, exist_ok=True)

    analyses = {
        "sql_dataset_overview.csv": get_dataset_overview(),
        "sql_data_quality_summary.csv": get_data_quality_summary(),
        "sql_duplicate_summary.csv": get_duplicate_summary(),
        "sql_monthly_sales.csv": get_monthly_sales(),
        "sql_country_sales.csv": get_country_sales(),
        "sql_top_products.csv": get_top_products()
    }

    for filename, dataframe in analyses.items():

        output_path = os.path.join(export_dir, filename)

        dataframe.to_csv(output_path, index=False)

        print(f"Exported: {output_path}")

    return analyses


# --------------------------------------------------
# 8. MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    print("\nFLOWFORGE SQL ANALYSIS")
    print("=" * 50)

    results = export_sql_analysis()

    for name, dataframe in results.items():

        print(f"\n{name}")
        print("-" * 50)

        print(dataframe.head(10).to_string(index=False))

    print("\nSQL analysis completed successfully.")
