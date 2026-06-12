import pandas as pd
import numpy as np

df = pd.read_excel("finaccess2024_datasprint.xlsx")

print("=" * 60)
print("SHAPE")
print("=" * 60)
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n" + "=" * 60)
print("COLUMNS & DTYPES")
print("=" * 60)
print(df.dtypes.to_string())

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head().to_string())

print("\n" + "=" * 60)
print("BASIC STATS (NUMERIC)")
print("=" * 60)
print(df.describe().to_string())

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Count": missing, "Percent": missing_pct})
print(missing_df[missing_df["Count"] > 0].sort_values("Percent", ascending=False).to_string())
if missing_df[missing_df["Count"] > 0].empty:
    print("No missing values!")

print("\n" + "=" * 60)
print("TARGET VARIABLE: financial_status")
print("=" * 60)
if "financial_status" in df.columns:
    print(df["financial_status"].value_counts())
    print()
    print(df["financial_status"].value_counts(normalize=True).round(4) * 100)
else:
    print("Column 'financial_status' not found!")
    print("Available columns:", list(df.columns))

print("\n" + "=" * 60)
print("CATEGORICAL COLUMNS - UNIQUE VALUES")
print("=" * 60)
for col in df.select_dtypes(include=["object", "category"]).columns:
    print(f"\n--- {col} ({df[col].nunique()} unique) ---")
    print(df[col].value_counts().head(10).to_string())

print("\n" + "=" * 60)
print("NUMERIC COLUMNS - DISTRIBUTIONS")
print("=" * 60)
for col in df.select_dtypes(include=[np.number]).columns:
    print(f"\n--- {col} ---")
    print(f"  Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}, Std: {df[col].std():.2f}")
