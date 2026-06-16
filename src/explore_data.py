import pandas as pd
import numpy as np

# So here we are loading the data with pd.read_excel() that reads the .xlsx file into a dataframe.
# umm think of a dataframe like a giant Excel table but inside python
df = pd.read_excel("finaccess2024_datasprint.xlsx")


print("=" * 60)
print("SHAPE")
print("=" * 60)
# df.shape tells us how big our dataset is. 
# df.shape[0] is the number of rows (people surveyed) and df.shape[1] is the now the number of columns (questions asked).
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n" + "=" * 60)
print("COLUMNS & DTYPES")
print("=" * 60)
# dtypes stands for data types so it tells us if a column has text or numbers
print(df.dtypes.to_string())

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
# df.head() is like a sneak peek so It shows just the first 5 rows 
# so we can see what the actual data looks like without crashing our screen with alot of stuff.
print(df.head().to_string())

print("\n" + "=" * 60)
print("BASIC STATS (NUMERIC)")
print("=" * 60)
# df.describe() does some math for us on the number columns. 
# it finds the avg mean, the median, the smallest and largest values.
print(df.describe().to_string())

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
# isnull().sum() checks every single column and counts how many blank or missing answers there are
# thats how i found out that barries_bank had missing values.
missing = df.isnull().sum()
#now here we calculate what % of the data is missing to see how bad the problem is.
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
# we put these counts into a mini table.
missing_df = pd.DataFrame({"Count": missing, "Percent": missing_pct})
# and also we only want to print the columns that have missing data so that is (> 0).
print(missing_df[missing_df["Count"] > 0].sort_values("Percent", ascending=False).to_string())
if missing_df[missing_df["Count"] > 0].empty:
    print("No missing values!")

print("\n" + "=" * 60)
print("TARGET VARIABLE: financial_status")
print("=" * 60)
# the 'target variable' is what we want our machine learning model to predict here which is the financial status.
if "financial_status" in df.columns:
    # value_counts() counts how many people said their status worsened, improved or stayed the same.
    print(df["financial_status"].value_counts())
    print()
    # normalize=True turns those counts into percentages so we can see the class imbalance easily.
    print(df["financial_status"].value_counts(normalize=True).round(4) * 100)
else:
    print("Column 'financial_status' not found!")
    print("Available columns:", list(df.columns))

print("\n" + "=" * 60)
print("CATEGORICAL COLUMNS - UNIQUE VALUES")
print("=" * 60)
# categorical columns are the ones with text like Male/Female, Urban/Rural and so on.
# we use select_dtypes to grab only those text columns then loop through them.
for col in df.select_dtypes(include=["object", "category"]).columns:
    print(f"\n--- {col} ({df[col].nunique()} unique answers) ---")
    # for each text column we print the top 10 most common answers
    print(df[col].value_counts().head(10).to_string())

print("\n" + "=" * 60)
print("NUMERIC COLUMNS - DISTRIBUTIONS")
print("=" * 60)
# now we loop through the numeric columns like age, income, household size
for col in df.select_dtypes(include=[np.number]).columns:
    print(f"\n--- {col} ---")
    # for each numeric column, print out some stats to understand the range of numbers.
    print(f"  Smallest: {df[col].min()}, Largest: {df[col].max()}, Average: {df[col].mean():.2f}, Middle (Median): {df[col].median():.2f}")
