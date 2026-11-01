import pandas as pd

from etl_engine import (
    remove_duplicates,
    handle_missing_values
)

# Create sample data
data = {
    "Customer": ["John", "Mary", "John", "Alex"],
    "Amount": [100, 200, 100, None]
}

df = pd.DataFrame(data)

print("ORIGINAL DATA")
print(df)

# Test duplicate removal
cleaned_df, duplicates_removed = remove_duplicates(df)

print("\nAFTER DUPLICATE REMOVAL")
print(cleaned_df)
print("Duplicates removed:", duplicates_removed)

# Test missing value handling
#final_df = handle_missing_values(
 #   cleaned_df,
  #  strategy="fill_mean"
#)

#print("\nAFTER MISSING VALUE HANDLING")
#print(final_df)

#print("\nRemaining missing values:")
#print(final_df.isna().sum())
from etl_engine import detect_null_values

# Test null detection using original dataset
null_results = detect_null_values(cleaned_df)

print("\nNULL VALUE REPORT")
print(null_results["report"])

print("\nRECORDS WITH NULL VALUES")
print(null_results["rows_with_nulls"])

print("\nCOMPLETE RECORDS")
print(null_results["complete_rows"])

print(
    "\nTotal affected records:",
    null_results["total_affected_rows"]
)