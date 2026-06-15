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

    # Binary encoding
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

    # Ordinal encoding
    X['Age'] = X['Age'].map({v: i for i, v in enumerate(
        ['16-17', '18-25', '26-35', '36-45', '46-55', 'Above 55'])})
    X['education_level'] = X['education_level'].map({v: i for i, v in enumerate([
        'No formal education', 'Some primary', 'Primary completed', 'Some secondary',
        'Secondary completed', 'Some technical training after secondary school',
        'Completed technical training after secondary school',
        'Some university', 'University completed'])})
    X['fl_score'] = X['fl_score'].map({v: i for i, v in enumerate(
        ['None correct', 'One correct', 'Two correct', 'All correct'])})

    # Engineered features
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

    # Target encoding for county
    county_worsened_rate = (y == 'Worsened').groupby(X['county']).mean()
    county_improved_rate = (y == 'Improved').groupby(X['county']).mean()
    X['county_worsened_rate'] = X['county'].map(county_worsened_rate)
    X['county_improved_rate'] = X['county'].map(county_improved_rate)
    X['county_freq'] = X['county'].map(X['county'].value_counts(normalize=True))

    # One-hot for remaining nominal columns
    X = pd.get_dummies(X, columns=['marital_status', 'barriers_mobile_money', 'barriers_bank', 'county'], drop_first=True)

    # Encode target
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, list(le.classes_), le


if __name__ == '__main__':
    X, y, classes, _ = load_and_preprocess()
    print(f"Preprocessing complete: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Classes: {classes}")
    print(f"No NaNs: {X.isnull().sum().sum() == 0}")
