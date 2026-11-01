
import pandas as pd

from etl_engine import build_cleaning_preview


# Small sample dataset to test cleaning safely
data = {
    "Customer": ["John", "Mary", "John", "Alex", "David"],
    "Amount": [100, 200, 100, None, 500],
    "Country": ["India", "UK", "India", "India", None]
}

df = pd.DataFrame(data)

print("ORIGINAL DATA")
print(df)

# Prepare a cleaning preview
results = build_cleaning_preview(
    df,
    duplicate_action="quarantine",
    missing_action="quarantine"
)

print("\nPROPOSED CLEAN DATA")
print(results["cleaned_preview"])

print("\nPROPOSED QUARANTINE DATA")
print(results["quarantine_preview"])

print("\nCLEANING SUMMARY")
for key, value in results["summary"].items():
    print(f"{key}: {value}")

print("\nORIGINAL DATA AFTER PREVIEW")
print(df)

print(
    "\nOriginal data unchanged:",
    len(df) == 5 and df.isna().sum().sum() == 2
)
