"""
ML Pipeline — Team Odysseus (DataSprint 2026)
Predicts financial_status (Worsened / Stayed the same / Improved)
Evaluated on Weighted F1-Score
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# STEP 1: Load the cleaned data
# =============================================================================
df = pd.read_csv('finaccess2024_cleaned.csv')
print(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")

# =============================================================================
# STEP 2: Separate features (X) and target (y)
# =============================================================================
# the target is what we want to predict
target_col = 'financial_status'
y = df[target_col]
X = df.drop(columns=[target_col])

print(f"\nTarget distribution:")
print(y.value_counts())

# =============================================================================
# STEP 3: Feature Engineering — Log transform monthly_income
# =============================================================================
# monthly_income is extremely skewed (mean 9703 vs median 5000)
# log transform compresses the range so the model treats it more fairly
X['log_income'] = np.log1p(X['monthly_income'])  # log1p = log(1+x), safe for 0s
# we keep the original column too — the model can decide which is more useful

# =============================================================================
# STEP 4: Encode categorical columns
# =============================================================================
# We have 3 types of categorical columns and each needs different encoding:

# --- TYPE 1: Binary columns (Yes/No or Usage/Non-usage) ---
# these are simple: just map to 1 and 0
binary_yes_no = [
    'defaulted', 'mobile_money_access', 'mobile_ownership_1',
    'experienced_shock', 'nfhi_11', 'nfhi_12', 'nfhi_13',
    'accessto_13k_1month', 'not_difficult'
]
for col in binary_yes_no:
    X[col] = (X[col] == 'Yes').astype(int)

binary_usage = [
    'Savings_formal', 'Savings_informal', 'Loan_formal',
    'Loan_informal', 'formal_service_use'
]
for col in binary_usage:
    X[col] = (X[col] == 'Usage').astype(int)

# location_type: Urban=1, Rural=0
X['location_type'] = (X['location_type'] == 'Urban').astype(int)

# Sex: Male=1, Female=0
X['Sex'] = (X['Sex'] == 'Male').astype(int)

# has_disability: With Disability=1, Without=0
X['has_disability'] = (X['has_disability'] == 'With Disability').astype(int)

print("\nBinary columns encoded.")

# --- TYPE 2: Ordinal columns (have a natural order) ---
# Age groups: ordered from youngest to oldest
age_order = ['16-17', '18-25', '26-35', '36-45', '46-55', 'Above 55']
X['Age'] = X['Age'].map({v: i for i, v in enumerate(age_order)})

# Education: ordered from lowest to highest
edu_order = [
    'No formal education', 'Some primary', 'Primary completed', 'Some secondary',
    'Secondary completed', 'Some technical training after secondary school',
    'Completed technical training after secondary school',
    'Some university', 'University completed'
]
X['education_level'] = X['education_level'].map({v: i for i, v in enumerate(edu_order)})

# Financial literacy score: ordered
fl_order = ['None correct', 'One correct', 'Two correct', 'All correct']
X['fl_score'] = X['fl_score'].map({v: i for i, v in enumerate(fl_order)})

print("Ordinal columns encoded.")

# --- TYPE 3: Nominal columns (no natural order — use one-hot encoding) ---
# marital_status (4 categories), barriers_mobile_money (10), barriers_bank (10)
# county (47 — high cardinality but we'll use one-hot and let the model sort it out)
nominal_cols = ['marital_status', 'barriers_mobile_money', 'barriers_bank', 'county']
X = pd.get_dummies(X, columns=nominal_cols, drop_first=True)

print(f"One-hot encoded nominal columns. Total features now: {X.shape[1]}")

# --- Verify: no object columns remaining ---
obj_cols = X.select_dtypes(include='object').columns.tolist()
if obj_cols:
    print(f"WARNING: Still have text columns: {obj_cols}")
else:
    print("All columns are numeric. Ready for modeling.")

# =============================================================================
# STEP 5: Encode the target variable
# =============================================================================
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print(f"\nTarget classes: {list(le_target.classes_)}")
print(f"Encoded as:     {list(range(len(le_target.classes_)))}")

# =============================================================================
# STEP 6: Train/Test Split (80/20, stratified)
# =============================================================================
# stratify=y_encoded makes sure each split has the same class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTrain set: {X_train.shape[0]} rows")
print(f"Test set:  {X_test.shape[0]} rows")

# =============================================================================
# STEP 7: Train Models
# =============================================================================
# We train 3 models and compare them. All use class_weight='balanced'
# to handle the class imbalance (Worsened is 52.6% of the data).

models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    ),
}

results = {}
print("\n" + "=" * 70)
print("MODEL TRAINING & EVALUATION")
print("=" * 70)

for name, model in models.items():
    print(f"\n--- {name} ---")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Weighted F1 Score (the competition metric)
    f1_w = f1_score(y_test, y_pred, average='weighted')
    results[name] = f1_w
    
    print(f"Weighted F1-Score: {f1_w:.4f}")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=le_target.classes_
    ))

# =============================================================================
# STEP 8: Results Comparison
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)
for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    bar = "#" * int(score * 50)
    print(f"  {name:25s}  F1={score:.4f}  {bar}")

best_name = max(results, key=results.get)
best_model = models[best_name]
print(f"\nBest model: {best_name} (F1={results[best_name]:.4f})")

# =============================================================================
# STEP 9: Confusion Matrix (for best model)
# =============================================================================
y_pred_best = best_model.predict(X_test)

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_best,
    display_labels=le_target.classes_,
    cmap='Blues',
    ax=ax
)
ax.set_title(f'Confusion Matrix — {best_name}')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("\nSaved: confusion_matrix.png")

# =============================================================================
# STEP 10: Feature Importance (for best model)
# =============================================================================
# This answers the Guiding Question: "Which factors most strongly predict
# financial deterioration among Kenyan adults?"

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    
    print(f"\nTop 15 Most Important Features ({best_name}):")
    for i, (feat, imp) in enumerate(feat_imp.head(15).items(), 1):
        bar = "#" * int(imp * 200)
        print(f"  {i:2d}. {feat:45s} {imp:.4f}  {bar}")
    
    # Plot top 20
    fig, ax = plt.subplots(figsize=(10, 8))
    feat_imp.head(20).plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title(f'Top 20 Feature Importances — {best_name}')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.close()
    print("Saved: feature_importance.png")
else:
    print("(Best model does not support feature_importances_)")

# =============================================================================
# STEP 11: Cross-Validation (to confirm the score is not a fluke)
# =============================================================================
print(f"\n{'='*70}")
print(f"5-FOLD CROSS-VALIDATION — {best_name}")
print(f"{'='*70}")
cv_scores = cross_val_score(best_model, X, y_encoded, cv=5, scoring='f1_weighted', n_jobs=-1)
print(f"  Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Mean F1:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print(f"\n{'='*70}")
print("PIPELINE COMPLETE")
print(f"{'='*70}")
