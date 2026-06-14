import pandas as pd
df = pd.read_csv('finaccess2024_datasprint.csv')

for col in df.select_dtypes(include='object').columns:
    vals = df[col].dropna().unique()
    print(f"\n=== {col} ({len(vals)} unique) ===")
    for v in sorted(vals):
        print(f"  {repr(v):65s} count={int((df[col]==v).sum())}")

print(f"\n=== MISSING VALUES ===")
print(df.isnull().sum()[df.isnull().sum() > 0])

print(f"\n=== DUPLICATES ===")
print(f"Duplicate rows: {df.duplicated().sum()}")
