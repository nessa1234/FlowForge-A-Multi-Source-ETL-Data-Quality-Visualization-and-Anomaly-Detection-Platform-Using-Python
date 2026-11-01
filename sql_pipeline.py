
import os
import uuid
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text

# --------------------------------------------------
# 1. DATABASE CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "flowforge.db")
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

engine = create_engine(DATABASE_URL, echo=False)


# --------------------------------------------------
# 2. CREATE DATABASE TABLES
# --------------------------------------------------

def create_database_tables():

    with engine.begin() as connection:

        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS data_quality_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT,
                source_file TEXT,
                total_rows INTEGER,
                missing_cells INTEGER,
                duplicate_rows INTEGER,
                staging_rows INTEGER,
                final_rows INTEGER,
                status TEXT,
                message TEXT,
                created_at TEXT
            )
        """))

    print("Database tables initialized successfully.")


# --------------------------------------------------
# 3. LOAD DATA INTO STAGING TABLE
# --------------------------------------------------

def load_to_staging(df, source_file="Unknown"):

    if df is None or df.empty:
        raise ValueError("Cannot stage an empty dataset.")

    batch_id = str(uuid.uuid4())

    staging_df = df.copy()
    staging_df["Batch_ID"] = batch_id
    staging_df["Source_File"] = source_file

    staging_df.to_sql(
        name="retail_staging",
        con=engine,
        if_exists="append",
        index=False
    )

    print(f"Staging completed: {len(staging_df)} records")

    return batch_id


# --------------------------------------------------
# 4. LOAD STAGING DATA INTO FINAL TABLE
# --------------------------------------------------

def load_staging_to_final(batch_id):

    # Retrieve the selected batch from staging
    query = text("""
        SELECT *
        FROM retail_staging
        WHERE Batch_ID = :batch_id
    """)

    with engine.connect() as connection:
        staging_df = pd.read_sql(
            query,
            connection,
            params={"batch_id": batch_id}
        )

    if staging_df.empty:
        raise ValueError("No staging records found for this batch.")

    # Create final table using the staging table structure.
    # This handles the first-ever batch correctly.
    with engine.begin() as connection:

        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS retail_transactions
            AS SELECT * FROM retail_staging WHERE 1=0
        """))

        existing = connection.execute(text("""
            SELECT COUNT(*)
            FROM retail_transactions
            WHERE Batch_ID = :batch_id
        """), {"batch_id": batch_id}).scalar()

    if existing and existing > 0:
        raise ValueError(
            "This batch has already been loaded into the final table."
        )

    staging_df.to_sql(
        name="retail_transactions",
        con=engine,
        if_exists="append",
        index=False
    )

    print(f"Final loading completed: {len(staging_df)} records")

    return len(staging_df)


# --------------------------------------------------
# 5. DATA QUALITY LOGGING
# --------------------------------------------------

def log_data_quality(
    df,
    batch_id,
    source_file,
    staging_rows,
    final_rows,
    status="SUCCESS",
    message="Data loading completed"
):

    total_rows = len(df)
    missing_cells = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    with engine.begin() as connection:

        connection.execute(text("""
            INSERT INTO data_quality_logs (
                batch_id,
                source_file,
                total_rows,
                missing_cells,
                duplicate_rows,
                staging_rows,
                final_rows,
                status,
                message,
                created_at
            )
            VALUES (
                :batch_id,
                :source_file,
                :total_rows,
                :missing_cells,
                :duplicate_rows,
                :staging_rows,
                :final_rows,
                :status,
                :message,
                :created_at
            )
        """), {
            "batch_id": batch_id,
            "source_file": source_file,
            "total_rows": total_rows,
            "missing_cells": missing_cells,
            "duplicate_rows": duplicate_rows,
            "staging_rows": staging_rows,
            "final_rows": final_rows,
            "status": status,
            "message": message,
            "created_at": datetime.now().isoformat()
        })

    print("Data quality log saved successfully.")


# --------------------------------------------------
# 6. COMPLETE SQL PIPELINE
# --------------------------------------------------

def run_sql_pipeline(df, source_file="Unknown"):

    create_database_tables()

    batch_id = load_to_staging(
        df,
        source_file
    )

    final_rows = load_staging_to_final(batch_id)

    log_data_quality(
        df=df,
        batch_id=batch_id,
        source_file=source_file,
        staging_rows=len(df),
        final_rows=final_rows
    )

    result = {
        "batch_id": batch_id,
        "staging_rows": len(df),
        "final_rows": final_rows,
        "database_path": DATABASE_PATH
    }

    return result


# --------------------------------------------------
# 7. READ SQL TABLES
# --------------------------------------------------

def get_sql_table(table_name):

    allowed_tables = [
        "retail_staging",
        "retail_transactions",
        "data_quality_logs"
    ]

    if table_name not in allowed_tables:
        raise ValueError("Invalid table name.")

    query = text(f"SELECT * FROM {table_name}")

    with engine.connect() as connection:
        result_df = pd.read_sql(query, connection)

    return result_df


# --------------------------------------------------
# 8. MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    create_database_tables()

    print("\nFlowForge SQL Pipeline is ready.")
    print(f"Database location: {DATABASE_PATH}")
