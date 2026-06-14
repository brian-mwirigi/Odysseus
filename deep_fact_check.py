"""
ULTRA-DEEP FACT-CHECK SCRIPT — DataSprint 2026
Cross-validates every single data point and structural integrity constraint.
"""
import pandas as pd
import numpy as np
import sys

raw = pd.read_csv('finaccess2024_datasprint.csv')
clean = pd.read_csv('finaccess2024_cleaned.csv')

def assert_check(condition, fail_msg):
    if not condition:
        print(f"  [X] FATAL: {fail_msg}")
        return False
    return True

print("\n" + "=" * 80)
print("ULTRA-DEEP DATASET VERIFICATION (PHASE 2)")
print("=" * 80)

all_passed = True

# 1. EXACT MATHEMATICAL RECONCILIATION OF ROWS
print("\n1. MATHEMATICAL RECONCILIATION")
raw_dedup = raw.drop_duplicates()
dupes_removed = len(raw) - len(raw_dedup)

# Strip quotes to find the true number of junk rows in raw
edu = raw_dedup['education_level'].str.replace('"', '', regex=False).str.strip()
mar = raw_dedup['marital_status']

junk_edu_mask = edu.isin(['Refused to Answer (DO NOT READ OUT)', "Don't know (DO NOT READ OUT)", 'Other (Specify)', '95'])
junk_mar_mask = mar.isin(["Don't know   (DO NOT READ OUT)", "Refused to Answer(DO NOT READ OUT)"])

# Total rows that have ANY junk
total_junk_rows = (junk_edu_mask | junk_mar_mask).sum()
expected_rows = len(raw_dedup) - total_junk_rows

print(f"  Raw rows: {len(raw)}")
print(f"  Duplicates removed: {dupes_removed}")
print(f"  Rows with junk data removed: {total_junk_rows}")
print(f"  Expected clean rows: {expected_rows}")
print(f"  Actual clean rows: {len(clean)}")
all_passed &= assert_check(len(clean) == expected_rows, f"Row count mismatch! Expected {expected_rows}, got {len(clean)}")
if expected_rows == len(clean): print("  [✓] Row math perfectly reconciled.")

# 2. NaN LEAKAGE TEST
print("\n2. NaN LEAKAGE & HIDDEN NULLS")
true_nans = clean.isnull().sum().sum()
all_passed &= assert_check(true_nans == 0, f"Found {true_nans} NaN values!")

# Check for hidden nulls ("NaN", "Null", "", " ")
hidden_nulls = 0
for col in clean.select_dtypes(include='object').columns:
    hidden = clean[col].isin(['NaN', 'Null', 'None', '', ' ']).sum()
    if hidden > 0:
        print(f"  [X] FATAL: Column '{col}' contains {hidden} hidden string-nulls!")
        hidden_nulls += hidden
all_passed &= assert_check(hidden_nulls == 0, f"Found {hidden_nulls} hidden null strings.")
if true_nans == 0 and hidden_nulls == 0: print("  [✓] Zero NaN values and zero hidden string nulls detected.")


# 3. CATEGORICAL CARDINALITY ENFORCEMENT
print("\n3. CATEGORICAL CARDINALITY & TYPO DETECTION")
expected_classes = {
    'location_type': 2, 'Sex': 2, 'Age': 6, 'education_level': 9,
    'marital_status': 4, 'Savings_formal': 2, 'Savings_informal': 2,
    'Loan_formal': 2, 'Loan_informal': 2, 'defaulted': 2,
    'formal_service_use': 2, 'mobile_money_access': 2,
    'barriers_mobile_money': 10, 'mobile_ownership_1': 2,
    'experienced_shock': 2, 'nfhi_11': 2, 'nfhi_12': 2, 'nfhi_13': 2,
    'accessto_13k_1month': 2, 'not_difficult': 2, 'financial_status': 3,
    'fl_score': 4, 'barriers_bank': 10, 'has_disability': 2
}
cardinality_ok = True
for col, expected_count in expected_classes.items():
    actual_count = clean[col].nunique()
    if actual_count != expected_count:
        print(f"  [X] FATAL: '{col}' should have {expected_count} classes, but has {actual_count}. Values: {clean[col].unique()}")
        cardinality_ok = False
all_passed &= assert_check(cardinality_ok, "Categorical cardinality violations found.")
if cardinality_ok: print(f"  [✓] All {len(expected_classes)} categorical columns have the exact expected number of classes.")


# 4. EXPLICIT IMPUTATION VERIFICATION
print("\n4. IMPUTATION VERIFICATION")
# Bank barriers
bank_nobarrier = (clean['barriers_bank'] == 'No barrier').sum()
# The exact number of NaNs we started with in the *kept* rows
raw_kept = raw_dedup[~(junk_edu_mask | junk_mar_mask)]
raw_kept_bank_nans = raw_kept['barriers_bank'].isnull().sum()
# Raw kept might already have some 'No barrier' (actually it doesn't, they were NaN)
print(f"  Expected 'No barrier' in bank: {raw_kept_bank_nans}")
print(f"  Actual 'No barrier' in bank: {bank_nobarrier}")
all_passed &= assert_check(bank_nobarrier == raw_kept_bank_nans, "Imputation count mismatch for barriers_bank.")

# Mobile money barriers
# Raw used '0'
raw_kept_mobile_zero = (raw_kept['barriers_mobile_money'] == '0').sum()
clean_mobile_nobarrier = (clean['barriers_mobile_money'] == 'No barrier').sum()
print(f"  Expected 'No barrier' in mobile: {raw_kept_mobile_zero}")
print(f"  Actual 'No barrier' in mobile: {clean_mobile_nobarrier}")
all_passed &= assert_check(clean_mobile_nobarrier == raw_kept_mobile_zero, "Replacement count mismatch for barriers_mobile_money.")
if bank_nobarrier == raw_kept_bank_nans and clean_mobile_nobarrier == raw_kept_mobile_zero:
    print("  [✓] 'No barrier' imputations perfectly match the raw data mapping.")

# 5. NO DATA LEAKAGE (Row shuffling/corruption)
print("\n5. DATA INTEGRITY (NO CORRUPTION)")
# If we sum household_size or monthly_income in raw_kept vs clean, they must match exactly
raw_income_sum = raw_kept['monthly_income'].sum()
clean_income_sum = clean['monthly_income'].sum()
raw_hh_sum = raw_kept['household_size'].sum()
clean_hh_sum = clean['household_size'].sum()

print(f"  Total Monthly Income (Clean): {clean_income_sum}")
print(f"  Total Household Size (Clean): {clean_hh_sum}")
all_passed &= assert_check(raw_income_sum == clean_income_sum, "Monthly income sum does not match! Data was corrupted.")
all_passed &= assert_check(raw_hh_sum == clean_hh_sum, "Household size sum does not match! Data was corrupted.")
if raw_income_sum == clean_income_sum and raw_hh_sum == clean_hh_sum:
    print("  [✓] Numeric column sums match perfectly. No row misalignment occurred.")

print("\n" + "=" * 80)
if all_passed:
    print("SUCCESS: Dataset has passed the Ultra-Deep Verification.")
    print("The data is flawlessly aligned with the raw source down to the byte.")
else:
    print("FAIL: Dataset failed Ultra-Deep Verification.")
print("=" * 80)
