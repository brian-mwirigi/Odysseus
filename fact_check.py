"""
FACT-CHECK: Verify the cleaned dataset is 100% accurate.
Compares cleaned data against the raw data to ensure nothing was lost or corrupted.
"""
import pandas as pd
import numpy as np

raw = pd.read_csv('finaccess2024_datasprint.csv')
clean = pd.read_csv('finaccess2024_cleaned.csv')

errors = []

print("=" * 70)
print("FACT-CHECK REPORT")
print("=" * 70)

# ── CHECK 1: Row count math ──
print("\n[1] ROW COUNT MATH")
raw_rows = len(raw)
dupes_in_raw = raw.duplicated().sum()
# Count junk rows in raw (after deduplication)
raw_deduped = raw.drop_duplicates()

# education junk (need to strip quotes first to match)
edu_vals = raw_deduped['education_level'].str.replace('"', '', regex=False).str.strip()
edu_junk = edu_vals.isin(['Refused to Answer (DO NOT READ OUT)', "Don't know (DO NOT READ OUT)", 'Other (Specify)', '95']).sum()

# marital junk
mar_junk = raw_deduped['marital_status'].isin(["Don't know   (DO NOT READ OUT)", "Refused to Answer(DO NOT READ OUT)"]).sum()

expected = raw_rows - dupes_in_raw - edu_junk - mar_junk
actual = len(clean)

print(f"  Raw rows:           {raw_rows}")
print(f"  Duplicates dropped: {dupes_in_raw}")
print(f"  Education junk:     {edu_junk}")
print(f"  Marital junk:       {mar_junk}")
print(f"  Expected clean:     {expected}")
print(f"  Actual clean:       {actual}")
if expected == actual:
    print("  >> PASS")
else:
    msg = f"Row count mismatch: expected {expected}, got {actual}"
    errors.append(msg)
    print(f"  >> FAIL: {msg}")

# ── CHECK 2: No missing values ──
print("\n[2] MISSING VALUES")
missing = clean.isnull().sum()
total_missing = missing.sum()
print(f"  Total NaN cells: {total_missing}")
if total_missing == 0:
    print("  >> PASS")
else:
    for col in missing[missing > 0].index:
        msg = f"{col} has {missing[col]} NaN values"
        errors.append(msg)
        print(f"  >> FAIL: {msg}")

# ── CHECK 3: No duplicates ──
print("\n[3] DUPLICATES")
dupes = clean.duplicated().sum()
print(f"  Duplicate rows: {dupes}")
if dupes == 0:
    print("  >> PASS")
else:
    errors.append(f"{dupes} duplicate rows remain")
    print(f"  >> FAIL")

# ── CHECK 4: Column count preserved ──
print("\n[4] COLUMN COUNT")
print(f"  Raw columns:   {raw.shape[1]}")
print(f"  Clean columns: {clean.shape[1]}")
if raw.shape[1] == clean.shape[1]:
    print("  >> PASS")
else:
    msg = f"Column count changed: {raw.shape[1]} -> {clean.shape[1]}"
    errors.append(msg)
    print(f"  >> FAIL: {msg}")

# ── CHECK 5: Column names match ──
print("\n[5] COLUMN NAMES")
if list(raw.columns) == list(clean.columns):
    print("  All column names match exactly.")
    print("  >> PASS")
else:
    msg = f"Column names differ"
    errors.append(msg)
    print(f"  >> FAIL: {msg}")

# ── CHECK 6: education_level is clean ──
print("\n[6] EDUCATION_LEVEL VALUES")
edu_clean = sorted(clean['education_level'].unique())
expected_edu = [
    'Completed technical training after secondary school',
    'No formal education',
    'Primary completed',
    'Secondary completed',
    'Some primary',
    'Some secondary',
    'Some technical training after secondary school',
    'Some university',
    'University completed'
]
print(f"  Expected: {len(expected_edu)} values")
print(f"  Actual:   {len(edu_clean)} values")
if edu_clean == expected_edu:
    print("  >> PASS")
else:
    extra = set(edu_clean) - set(expected_edu)
    missing_vals = set(expected_edu) - set(edu_clean)
    if extra:
        errors.append(f"Unexpected education values: {extra}")
        print(f"  >> FAIL: Unexpected values: {extra}")
    if missing_vals:
        errors.append(f"Missing education values: {missing_vals}")
        print(f"  >> FAIL: Missing values: {missing_vals}")

# Check no quotes or trailing spaces in education values
for v in edu_clean:
    if '"' in v or v != v.strip():
        msg = f"education_level has dirty value: {repr(v)}"
        errors.append(msg)
        print(f"  >> FAIL: {msg}")

# ── CHECK 7: marital_status is clean ──
print("\n[7] MARITAL_STATUS VALUES")
mar_clean = sorted(clean['marital_status'].unique())
expected_mar = ['Divorced/separated', 'Married/Living with partner', 'Single/Never Married', 'Widowed']
print(f"  Values: {mar_clean}")
if mar_clean == expected_mar:
    print("  >> PASS")
else:
    errors.append(f"marital_status has unexpected values: {set(mar_clean) - set(expected_mar)}")
    print(f"  >> FAIL")

# ── CHECK 8: barriers_bank has no NaN, has 'No barrier' ──
print("\n[8] BARRIERS_BANK")
bb_nulls = clean['barriers_bank'].isnull().sum()
bb_nobarrier = (clean['barriers_bank'] == 'No barrier').sum()
print(f"  NaN count: {bb_nulls}")
print(f"  'No barrier' count: {bb_nobarrier}")
# The raw data had ~5730 NaN in barriers_bank — these should now be 'No barrier'
raw_bb_nulls = raw.drop_duplicates()['barriers_bank'].isnull().sum()
# Account for rows we dropped (edu + marital junk)
print(f"  Raw NaN count (after dedup): {raw_bb_nulls}")
if bb_nulls == 0 and bb_nobarrier > 0:
    print("  >> PASS")
else:
    errors.append("barriers_bank still has issues")
    print("  >> FAIL")

# ── CHECK 9: barriers_mobile_money has no '0' values ──
print("\n[9] BARRIERS_MOBILE_MONEY")
zero_count = (clean['barriers_mobile_money'] == '0').sum()
nobarrier_count = (clean['barriers_mobile_money'] == 'No barrier').sum()
print(f"  '0' values remaining: {zero_count}")
print(f"  'No barrier' count: {nobarrier_count}")
if zero_count == 0 and nobarrier_count > 0:
    print("  >> PASS")
else:
    errors.append("barriers_mobile_money still has '0' values")
    print("  >> FAIL")

# ── CHECK 10: All text columns are clean (no quotes, no trailing spaces) ──
print("\n[10] TEXT COLUMN CLEANLINESS (quotes & whitespace)")
dirty_found = False
for col in clean.select_dtypes(include='object').columns:
    for v in clean[col].dropna().unique():
        if '"' in v:
            msg = f"{col} has value with quotes: {repr(v)}"
            errors.append(msg)
            print(f"  >> FAIL: {msg}")
            dirty_found = True
        if v != v.strip():
            msg = f"{col} has value with whitespace: {repr(v)}"
            errors.append(msg)
            print(f"  >> FAIL: {msg}")
            dirty_found = True
if not dirty_found:
    print("  All text values are clean (no embedded quotes, no trailing spaces).")
    print("  >> PASS")

# ── CHECK 11: Target variable preserved correctly ──
print("\n[11] TARGET VARIABLE (financial_status)")
target_vals = sorted(clean['financial_status'].unique())
expected_target = ['Improved', 'Stayed the same', 'Worsened']
print(f"  Values: {target_vals}")
total_target = clean['financial_status'].value_counts().sum()
print(f"  Total: {total_target} (should equal {len(clean)})")
if target_vals == expected_target and total_target == len(clean):
    print("  >> PASS")
else:
    errors.append("Target variable has issues")
    print("  >> FAIL")

# ── CHECK 12: Numeric columns — no corruption ──
print("\n[12] NUMERIC COLUMNS INTEGRITY")
for col in ['household_size', 'monthly_income', 'prodsum1']:
    raw_min = raw[col].min()
    raw_max = raw[col].max()
    clean_min = clean[col].min()
    clean_max = clean[col].max()
    # Clean min/max should be within raw range
    ok = clean_min >= raw_min and clean_max <= raw_max
    print(f"  {col}: raw[{raw_min}-{raw_max}] clean[{clean_min}-{clean_max}] {'PASS' if ok else 'FAIL'}")
    if not ok:
        errors.append(f"{col} values outside raw range")

# ── CHECK 13: Binary columns only have expected values ──
print("\n[13] BINARY COLUMN VALUES")
binary_checks = {
    'location_type': ['Rural', 'Urban'],
    'Sex': ['Female', 'Male'],
    'Savings_formal': ['Non-usage', 'Usage'],
    'Savings_informal': ['Non-usage', 'Usage'],
    'Loan_formal': ['Non-usage', 'Usage'],
    'Loan_informal': ['Non-usage', 'Usage'],
    'defaulted': ['No', 'Yes'],
    'formal_service_use': ['Non-usage', 'Usage'],
    'mobile_money_access': ['No', 'Yes'],
    'mobile_ownership_1': ['No', 'Yes'],
    'experienced_shock': ['No', 'Yes'],
    'nfhi_11': ['No', 'Yes'],
    'nfhi_12': ['No', 'Yes'],
    'nfhi_13': ['No', 'Yes'],
    'accessto_13k_1month': ['No', 'Yes'],
    'not_difficult': ['No', 'Yes'],
    'has_disability': ['With Disability', 'Without Disability'],
}
all_binary_ok = True
for col, expected_vals in binary_checks.items():
    actual_vals = sorted(clean[col].unique())
    if actual_vals != expected_vals:
        msg = f"{col}: expected {expected_vals}, got {actual_vals}"
        errors.append(msg)
        print(f"  >> FAIL: {msg}")
        all_binary_ok = False
if all_binary_ok:
    print(f"  All {len(binary_checks)} binary columns have exactly the expected values.")
    print("  >> PASS")

# ── CHECK 14: fl_score values ──
print("\n[14] FL_SCORE VALUES")
fl_vals = sorted(clean['fl_score'].unique())
expected_fl = ['All correct', 'None correct', 'One correct', 'Two correct']
print(f"  Values: {fl_vals}")
if fl_vals == expected_fl:
    print("  >> PASS")
else:
    errors.append(f"fl_score unexpected values: {fl_vals}")
    print("  >> FAIL")

# ── CHECK 15: Age values ──
print("\n[15] AGE VALUES")
age_vals = sorted(clean['Age'].unique())
expected_age = ['16-17', '18-25', '26-35', '36-45', '46-55', 'Above 55']
print(f"  Values: {age_vals}")
if age_vals == expected_age:
    print("  >> PASS")
else:
    errors.append(f"Age unexpected values: {age_vals}")
    print("  >> FAIL")

# ── CHECK 16: County count ──
print("\n[16] COUNTY COUNT")
county_count = clean['county'].nunique()
print(f"  Unique counties: {county_count}")
if county_count == 47:
    print("  >> PASS (all 47 Kenyan counties present)")
else:
    errors.append(f"Expected 47 counties, got {county_count}")
    print(f"  >> FAIL")

# ── FINAL VERDICT ──
print("\n" + "=" * 70)
if len(errors) == 0:
    print("VERDICT: ALL 16 CHECKS PASSED. DATA IS CLEAN.")
else:
    print(f"VERDICT: {len(errors)} ISSUES FOUND:")
    for i, e in enumerate(errors, 1):
        print(f"  {i}. {e}")
print("=" * 70)
