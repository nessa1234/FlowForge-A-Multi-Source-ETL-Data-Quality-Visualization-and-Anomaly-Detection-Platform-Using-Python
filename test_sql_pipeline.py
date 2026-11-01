import pandas as pd
from sql_pipeline import run_sql_pipeline, get_sql_table

DATASET_PATH = "uploads/03bbdd2a_Online_Retail.xlsx"

print("Reading Online Retail dataset...")

df = pd.read_excel(DATASET_PATH)

print("Dataset shape:", df.shape)

print("\nRunning SQL pipeline...")

result = run_sql_pipeline(
    df,
    source_file="03bbdd2a_Online_Retail.xlsx"
)

print("\nSQL PIPELINE RESULTS")
print(result)

print("\nChecking final SQL table...")

final_df = get_sql_table("retail_transactions")

print("Final table shape:", final_df.shape)

print("\nChecking data quality logs...")

logs = get_sql_table("data_quality_logs")

print(logs.tail(1))