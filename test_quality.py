
import pandas as pd
from etl_engine import run_data_quality_analysis

df = pd.read_excel(
    r"C:\Users\sarat\Downloads\online+retail\Online Retail.xlsx"
)

results = run_data_quality_analysis(df)

print("\nFLOWFORGE DATA QUALITY REPORT")
print("--------------------------------")

print("Total Records:", results["total_rows"])
print("Total Columns:", results["total_columns"])
print("Missing Cells:", results["missing_cells"])
print("Records with Missing Values:", results["missing_rows"])
print("Duplicate Records:", results["duplicate_count"])
print("Data Completeness:", results["completeness"], "%")

print("\nMISSING VALUES BY COLUMN")
print(results["null_report"])

print("\nDATA TYPES")
print(results["data_types"])

print("\nRETAIL BUSINESS RULE FINDINGS")

for rule, records in results["retail_rule_findings"].items():
    print(rule, ":", len(records), "records")

print("\nOriginal dataset remains unchanged:",
      len(df) == results["total_rows"])
