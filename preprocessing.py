"""
Data preprocessing and feature engineering for the ML pipeline.
Run this first. Other scripts import from here.
"""
import pandas as pd
import numpy as np


def load_and_preprocess():
    df = pd.read_csv('finaccess2024_cleaned.csv')
    assert df.shape[0] == 20848 and df.isnull().sum().sum() == 0

    y = df['financial_status']
    X = df.drop(columns=['financial_status'])

    for col in ['defaulted', 'mobile_money_access', 'mobile_ownership_1',
                'experienced_shock', 'nfhi_11', 'nfhi_12', 'nfhi_13',
                'accessto_13k_1month', 'not_difficult']:
        X[col] = (X[col] == 'Yes').astype(int)

    for col in ['Savings_formal', 'Savings_informal', 'Loan_formal',
                'Loan_informal', 'formal_service_use']:
        X[col] = (X[col] == 'Usage').astype(int)

    X['location_type'] = (X['location_type'] == 'Urban').astype(int)
    X['Sex'] = (X['Sex'] == 'Male').astype(int)
    X['has_disability'] = (X['has_disability'] == 'With Disability').astype(int)

    X['Age'] = X['Age'].map({v: i for i, v in enumerate(
        ['16-17', '18-25', '26-35', '36-45', '46-55', 'Above 55'])}).fillna(-1)
    X['education_level'] = X['education_level'].map({v: i for i, v in enumerate([
        'No formal education', 'Some primary', 'Primary completed', 'Some secondary',
        'Secondary completed', 'Some technical training after secondary school',
        'Completed technical training after secondary school',
        'Some university', 'University completed'])}).fillna(-1)
    X['fl_score'] = X['fl_score'].map({v: i for i, v in enumerate(
        ['None correct', 'One correct', 'Two correct', 'All correct'])}).fillna(-1)

    X['log_income'] = np.log1p(X['monthly_income'])
    X['income_per_person'] = X['monthly_income'] / X['household_size'].clip(lower=1)
    X['log_income_per_person'] = np.log1p(X['income_per_person'])
    X['nfhi_composite'] = X['nfhi_11'] + X['nfhi_12'] + X['nfhi_13']
    X['total_formal_products'] = X['Savings_formal'] + X['Loan_formal'] + X['formal_service_use']
    X['total_informal_products'] = X['Savings_informal'] + X['Loan_informal']
    X['total_products'] = X['total_formal_products'] + X['total_informal_products']
    X['resilience_score'] = X['accessto_13k_1month'] + X['not_difficult'] + (1 - X['defaulted'])
    X['shock_vulnerable'] = ((X['experienced_shock'] == 1) & (X['resilience_score'] <= 1)).astype(int)
    X['edu_income_ratio'] = X['education_level'] / (X['log_income'] + 1)
    X['prodsum1_per_person'] = X['prodsum1'] / X['household_size'].clip(lower=1)
    X['age_x_shock'] = X['Age'] * X['experienced_shock']
    X['urban_x_formal'] = X['location_type'] * X['total_formal_products']
    X['digital_access'] = X['mobile_money_access'] * X['mobile_ownership_1']
    X['financial_capability'] = X['education_level'] * X['fl_score']

    X['shock_no_resilience'] = X['experienced_shock'] * (1 - X['accessto_13k_1month'])
    X['income_quantile'] = pd.qcut(X['monthly_income'], q=5, labels=False, duplicates='drop')

    age_midpoint_map = {0: 16.5, 1: 21.5, 2: 30.5, 3: 40.5, 4: 50.5, 5: 60}
    X['dependency_ratio'] = X['household_size'] / (1 + X['Age'].map(age_midpoint_map).fillna(30.5))
    X['no_savings_no_access'] = ((X['Savings_formal'] == 0) & (X['accessto_13k_1month'] == 0)).astype(int)

    X['shock_x_defaulted'] = X['experienced_shock'] * X['defaulted']
    X['income_x_formal'] = X['log_income'] * X['total_formal_products']
    X['age_x_education'] = X['Age'] * X['education_level']
    X['disability_x_shock'] = X['has_disability'] * X['experienced_shock']
    X['urban_x_income'] = X['location_type'] * X['log_income']
    X['shock_x_no_savings'] = X['experienced_shock'] * X['no_savings_no_access']
    X['shock_x_income'] = X['experienced_shock'] * X['log_income']
    X['defaulted_x_resilience'] = X['defaulted'] * X['resilience_score']
    X['formal_x_resilience'] = X['total_formal_products'] * X['resilience_score']
    X['income_x_resilience'] = X['log_income_per_person'] * X['resilience_score']
    X['age_x_shock_defaulted'] = X['Age'] * X['shock_x_defaulted']
    X['education_x_income'] = X['education_level'] * X['log_income']
    X['shock_x_disability_x_income'] = X['shock_vulnerable'] * X['log_income']
    X['urban_x_formal_x_income'] = X['urban_x_formal'] * X['log_income']
    X['household_burden'] = X['household_size'] / (X['log_income'] + 1)
    X['savings_gap'] = 3 - X['total_formal_products']
    X['access_gap'] = (1 - X['accessto_13k_1month']) + (1 - X['not_difficult'])
    X['vulnerability_composite'] = X['experienced_shock'] + X['defaulted'] + (1 - X['accessto_13k_1month']) + X['no_savings_no_access']

    for col in ['marital_status', 'barriers_mobile_money', 'barriers_bank']:
        freq = X[col].value_counts(normalize=True)
        X[col + '_freq'] = X[col].map(freq).fillna(0)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    cat_cols_to_te = ['county', 'marital_status', 'barriers_mobile_money', 'barriers_bank']

    return X, y_encoded, list(le.classes_), le, cat_cols_to_te


if __name__ == '__main__':
    X, y, classes, _, cat_cols = load_and_preprocess()
    print(f"Preprocessing complete: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Classes: {classes}")
    print(f"Cat cols for TE: {cat_cols}")
    print(f"No NaNs: {X.isnull().sum().sum() == 0}")
