import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier

print("=" * 70)
print("LOADING & PREPROCESSING DATA")
print("=" * 70)

df = pd.read_csv('finaccess2024_cleaned.csv')

target_col = 'financial_status'
y = df[target_col]
X = df.drop(columns=[target_col])

X['log_income'] = np.log1p(X['monthly_income'])

binary_yes_no = ['defaulted', 'mobile_money_access', 'mobile_ownership_1', 'experienced_shock', 'nfhi_11', 'nfhi_12', 'nfhi_13', 'accessto_13k_1month', 'not_difficult']
for col in binary_yes_no: X[col] = (X[col] == 'Yes').astype(int)

binary_usage = ['Savings_formal', 'Savings_informal', 'Loan_formal', 'Loan_informal', 'formal_service_use']
for col in binary_usage: X[col] = (X[col] == 'Usage').astype(int)

X['location_type'] = (X['location_type'] == 'Urban').astype(int)
X['Sex'] = (X['Sex'] == 'Male').astype(int)
X['has_disability'] = (X['has_disability'] == 'With Disability').astype(int)

age_order = ['16-17', '18-25', '26-35', '36-45', '46-55', 'Above 55']
X['Age'] = X['Age'].map({v: i for i, v in enumerate(age_order)})

edu_order = [
    'No formal education', 'Some primary', 'Primary completed', 'Some secondary',
    'Secondary completed', 'Some technical training after secondary school',
    'Completed technical training after secondary school',
    'Some university', 'University completed'
]
X['education_level'] = X['education_level'].map({v: i for i, v in enumerate(edu_order)})

fl_order = ['None correct', 'One correct', 'Two correct', 'All correct']
X['fl_score'] = X['fl_score'].map({v: i for i, v in enumerate(fl_order)})

nominal_cols = ['marital_status', 'barriers_mobile_money', 'barriers_bank', 'county']
X = pd.get_dummies(X, columns=nominal_cols, drop_first=True)

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

print("Preprocessing complete. Total features:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("\nOriginal Training Set Target Distribution:")
train_dist = pd.Series(y_train).value_counts()
for idx, count in train_dist.items():
    print(f"  {le_target.classes_[idx]}: {count}")

print("\nApplying SMOTE (Synthetic Minority Over-sampling Technique)...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nResampled Training Set Target Distribution:")
resampled_dist = pd.Series(y_train_resampled).value_counts()
for idx, count in resampled_dist.items():
    print(f"  {le_target.classes_[idx]}: {count}")

print("\n" + "=" * 70)
print("ADVANCED MODEL TRAINING & EVALUATION")
print("=" * 70)

models = {
    'XGBoost': XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6, 
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
    ),
    'Gradient Boosting (Baseline)': GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\n--- {name} ---")
    
    if 'Baseline' in name:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train_resampled, y_train_resampled)
    
    y_pred = model.predict(X_test)
    
    f1_w = f1_score(y_test, y_pred, average='weighted')
    results[name] = f1_w
    
    print(f"Weighted F1-Score: {f1_w:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))

print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)
for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    bar = "#" * int(score * 50)
    print(f"  {name:30s}  F1={score:.4f}  {bar}")

best_name = max(results, key=results.get)
best_model = models[best_name]
print(f"\n🏆 Best model: {best_name} (F1={results[best_name]:.4f})")

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    
    print(f"\nTop 10 Most Important Features ({best_name}):")
    for i, (feat, imp) in enumerate(feat_imp.head(10).items(), 1):
        bar = "#" * int(imp * 200)
        print(f"  {i:2d}. {feat:40s} {imp:.4f}  {bar}")
        
print("\nAdvanced Pipeline Finished.")
