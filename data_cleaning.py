import pandas as pd
import numpy as np


df = pd.read_csv('finaccess2024_datasprint.csv')
print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")


dupes_before = df.duplicated().sum()
df = df.drop_duplicates()
print(f"\nSTEP 1 — Duplicates: Found {dupes_before}, dropped them. Rows now: {df.shape[0]}")


df['education_level'] = df['education_level'].str.replace('"', '', regex=False).str.strip()


junk_education = [
    'Refused to Answer (DO NOT READ OUT)',
    "Don't know (DO NOT READ OUT)", 
    'Other (Specify)',
    '95'
]
rows_before = len(df)
df = df[~df['education_level'].isin(junk_education)]
rows_dropped = rows_before - len(df)
print(f"\nSTEP 2 — education_level: Stripped quotes/whitespace. Dropped {rows_dropped} junk rows.")
print(f"  Clean values: {sorted(df['education_level'].unique())}")


junk_marital = [
    "Don't know   (DO NOT READ OUT)",
    "Refused to Answer(DO NOT READ OUT)"
]
rows_before = len(df)
df = df[~df['marital_status'].isin(junk_marital)]
rows_dropped = rows_before - len(df)
print(f"\nSTEP 3 — marital_status: Dropped {rows_dropped} junk rows.")
print(f"  Clean values: {sorted(df['marital_status'].unique())}")

missing_before = df['barriers_bank'].isnull().sum()
df['barriers_bank'] = df['barriers_bank'].fillna('No barrier')
print(f"\nSTEP 4 — barriers_bank: Filled {missing_before} NaN values with 'No barrier'.")


count_zero = (df['barriers_mobile_money'] == '0').sum()
df['barriers_mobile_money'] = df['barriers_mobile_money'].replace('0', 'No barrier')
print(f"\nSTEP 5 — barriers_mobile_money: Renamed {count_zero} '0' values to 'No barrier'.")


text_cols = df.select_dtypes(include='object').columns
for col in text_cols:
    df[col] = df[col].str.strip()
print(f"\nSTEP 6 — Stripped whitespace from all {len(text_cols)} text columns.")

print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

missing = df.isnull().sum()
missing_total = missing.sum()
print(f"\n✓ Missing values: {missing_total}")
if missing_total > 0:
    print("  WARNING — still have missing values:")
    print(missing[missing > 0])


dupes = df.duplicated().sum()
print(f"✓ Duplicate rows: {dupes}")


print(f"✓ Final shape: {df.shape[0]} rows, {df.shape[1]} columns")


print(f"\n✓ Target variable (financial_status):")
print(df['financial_status'].value_counts())
print()
print((df['financial_status'].value_counts(normalize=True) * 100).round(2))

print(f"\n✓ All columns — unique value counts:")
for col in df.columns:
    print(f"  {col}: {df[col].nunique()} unique")

print(f"\n✓ education_level clean values:")
for v in sorted(df['education_level'].unique()):
    print(f"  {repr(v):60s} count={int((df['education_level']==v).sum())}")

output_path = 'finaccess2024_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"\n{'=' * 70}")
print(f"SAVED cleaned data to: {output_path}")
print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"{'=' * 70}")
