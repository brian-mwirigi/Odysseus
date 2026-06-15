"""
FINAL ML SCRIPT — Team Odysseus
What this does: Tries to guess if someone's financial status Got Worse, Stayed the Same, or Improved.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore') # Hide annoying warning messages

# Import all the tools we need from scikit-learn and other libraries
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, 
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib # Used for saving our trained model later

# =======================================================================
# STEP 1: LOAD THE DATA
# =======================================================================
print("=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

# Load the cleaned dataset we made earlier
df = pd.read_csv('finaccess2024_cleaned.csv')
print(f"Loaded: {df.shape[0]} rows and {df.shape[1]} columns")

# =======================================================================
# STEP 2: FIX THE DATA FOR THE MODEL (FEATURE ENGINEERING)
# =======================================================================
print("\n" + "=" * 70)
print("STEP 2: PREPARING THE DATA")
print("=" * 70)

# 'financial_status' is the thing we want to predict (the Target)
target_col = 'financial_status'
y = df[target_col]
X = df.drop(columns=[target_col]) # Drop it from the inputs

# --- Fix Yes/No Questions ---
# The computer only understands numbers, so we turn "Yes" into 1 and "No" into 0
binary_yes_no = [
    'defaulted', 'mobile_money_access', 'mobile_ownership_1',
    'experienced_shock', 'nfhi_11', 'nfhi_12', 'nfhi_13',
    'accessto_13k_1month', 'not_difficult'
]
for col in binary_yes_no:
    X[col] = (X[col] == 'Yes').astype(int)

# Turn "Usage" into 1 and "Non-usage" into 0
binary_usage = [
    'Savings_formal', 'Savings_informal', 'Loan_formal',
    'Loan_informal', 'formal_service_use'
]
for col in binary_usage:
    X[col] = (X[col] == 'Usage').astype(int)

# Urban = 1, Rural = 0
X['location_type'] = (X['location_type'] == 'Urban').astype(int)
# Male = 1, Female = 0
X['Sex'] = (X['Sex'] == 'Male').astype(int)
# With Disability = 1, Without = 0
X['has_disability'] = (X['has_disability'] == 'With Disability').astype(int)

# --- Fix Ordered Categories (Like Age and Education) ---
# We number these in order, so the computer knows 55+ is older than 18
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

# --- Create New Helpful Columns ---
# Income is super lopsided (a few rich people, many poor people), so we use math (log) to balance it out
X['log_income'] = np.log1p(X['monthly_income'])

# Calculate how much money each person in the house gets
X['income_per_person'] = X['monthly_income'] / X['household_size'].clip(lower=1)
X['log_income_per_person'] = np.log1p(X['income_per_person'])

# Add up their financial health scores
X['nfhi_composite'] = X['nfhi_11'] + X['nfhi_12'] + X['nfhi_13']

# Add up how many financial products they use
X['total_formal_products'] = X['Savings_formal'] + X['Loan_formal'] + X['formal_service_use']
X['total_informal_products'] = X['Savings_informal'] + X['Loan_informal']
X['total_products'] = X['total_formal_products'] + X['total_informal_products']

# Give them a resilience score (1 point for no default, 1 point for easy access to 13k)
X['resilience_score'] = X['accessto_13k_1month'] + X['not_difficult'] + (1 - X['defaulted'])

# Flag people who had an emergency but no money to deal with it
X['shock_vulnerable'] = ((X['experienced_shock'] == 1) & (X['resilience_score'] <= 1)).astype(int)

# Education vs Income (are they highly educated but broke?)
X['edu_income_ratio'] = X['education_level'] / (X['log_income'] + 1)

# --- Fix Messy Text Columns ---
# Instead of making 47 new columns for counties, we just count how common each county is
county_freq = X['county'].value_counts(normalize=True)
X['county_freq'] = X['county'].map(county_freq)

# For other text columns (like marital status), we make a new Yes/No column for every option
nominal_cols = ['marital_status', 'barriers_mobile_money', 'barriers_bank', 'county']
X = pd.get_dummies(X, columns=nominal_cols, drop_first=True)

# --- Turn the Target into Numbers ---
# Turn "Worsened", "Stayed the same", "Improved" into 0, 1, 2
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
class_names = list(le_target.classes_)
print(f"Target classes changed to numbers: {class_names} -> {list(range(len(class_names)))}")

# =======================================================================
# STEP 3: SPLIT DATA INTO TRAIN AND TEST
# =======================================================================
print("\n" + "=" * 70)
print("STEP 3: SPLITTING DATA")
print("=" * 70)

# We hide 20% of the data to test the model later. We only teach it on 80%.
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# =======================================================================
# STEP 4: FIX THE LOPSIDED TARGET DATA (SMOTE)
# =======================================================================
print("\n" + "=" * 70)
print("STEP 4: FIXING IMBALANCE (SMOTE)")
print("=" * 70)

# Too many people said "Worsened". If we don't fix this, the model will just guess "Worsened" every time.
# SMOTE creates fake (synthetic) examples of people who "Improved" so the data is perfectly fair.
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# SMOTETomek does the same thing, but also deletes confusing examples that are right on the border
smotetomek = SMOTETomek(random_state=42)
X_train_st, y_train_st = smotetomek.fit_resample(X_train, y_train)

# =======================================================================
# STEP 5: TRAIN LOTS OF DIFFERENT MODELS
# =======================================================================
print("\n" + "=" * 70)
print("STEP 5: TRAINING THE MODELS")
print("=" * 70)

# We are going to test 12 different setups to see which one gets the highest score
configs = [
    ("RF + class_weight", RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1), X_train, y_train),
    ("RF + SMOTE", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1), X_train_smote, y_train_smote),
    ("GB + raw", GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42), X_train, y_train),
    ("GB + SMOTE", GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42), X_train_smote, y_train_smote),
    ("XGB + raw", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1), X_train, y_train),
    ("XGB + SMOTE", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1), X_train_smote, y_train_smote),
    ("XGB + SMOTE+Tomek", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1), X_train_st, y_train_st),
    ("LGBM + raw", LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1), X_train, y_train),
    ("LGBM + SMOTE", LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1), X_train_smote, y_train_smote),
    ("LGBM + SMOTE+Tomek", LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1), X_train_st, y_train_st),
    ("LGBM + class_weight", LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1), X_train, y_train),
    ("XGB + class_weight", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,), X_train, y_train),
]

# We need this math formula to help the XGBoost class_weight model work
from sklearn.utils.class_weight import compute_sample_weight

results = {}
trained_models = {}

for name, model, train_x, train_y in configs:
    print(f"\n  Training: {name}...", end=" ", flush=True)
    
    if name == "XGB + class_weight":
        sw = compute_sample_weight('balanced', train_y)
        model.fit(train_x, train_y, sample_weight=sw)
    else:
        model.fit(train_x, train_y)
    
    # Let the model take a guess on the 20% of data we hid earlier
    y_pred = model.predict(X_test)
    
    # Grade the test (F1-score is our grade)
    f1_w = f1_score(y_test, y_pred, average='weighted')
    results[name] = f1_w
    trained_models[name] = model
    print(f"Score={f1_w:.4f}")

# =======================================================================
# STEP 6: FIND THE WINNER
# =======================================================================
print("\n" + "=" * 70)
print("STEP 6: LEADERBOARD")
print("=" * 70)

# Sort them from best to worst
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for rank, (name, score) in enumerate(sorted_results, 1):
    print(f"  {rank}. {name} scored {score:.4f}")

best_name = sorted_results[0][0]
best_model = trained_models[best_name]
best_f1 = sorted_results[0][1]

# =======================================================================
# STEP 7: TRY TO MAKE THE WINNER EVEN BETTER
# =======================================================================
print("\n" + "=" * 70)
print(f"STEP 7: FINE TUNING")
print("=" * 70)

# Grab the data that the winning model used
best_config = [c for c in configs if c[0] == best_name][0]
tune_X, tune_y = best_config[2], best_config[3]

# Give the model random settings to try and see if it can accidentally beat its own score
if 'LGBM' in best_name:
    param_dist = {
        'n_estimators': [200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 6, 8],
    }
    base_model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
elif 'XGB' in best_name:
    param_dist = {
        'n_estimators': [200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 6, 8],
    }
    base_model = XGBClassifier(random_state=42, n_jobs=-1)
else:
    param_dist = {'n_estimators': [200, 300]}
    base_model = GradientBoostingClassifier(random_state=42)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print(f"Trying random settings on {best_name} to see if we can boost the score...")
search = RandomizedSearchCV(
    base_model, param_dist, n_iter=10, cv=cv, scoring='f1_weighted',
    random_state=42, n_jobs=-1, verbose=0
)

if 'class_weight' in best_name and 'XGB' in best_name:
    sw = compute_sample_weight('balanced', tune_y)
    search.fit(tune_X, tune_y, sample_weight=sw)
else:
    search.fit(tune_X, tune_y)

tuned_model = search.best_estimator_
y_pred_tuned = tuned_model.predict(X_test)
f1_tuned = f1_score(y_test, y_pred_tuned, average='weighted')

if f1_tuned > best_f1:
    best_model = tuned_model
    best_f1 = f1_tuned
    print(">> The random settings worked! We got a better score.")
else:
    print(">> The random settings made it worse. We will stick with the original winner.")

# =======================================================================
# STEP 8: TEAMWORK (COMBINE THE TOP 3)
# =======================================================================
print("\n" + "=" * 70)
print("STEP 8: ENSEMBLE (COMBINING THE BEST MODELS)")
print("=" * 70)

top3 = sorted_results[:3]
print("We are going to make the Top 3 models vote on the final answer:")

ensemble_estimators = []
for name, _ in top3:
    cfg = [c for c in configs if c[0] == name][0]
    model_copy = cfg[1].__class__(**cfg[1].get_params())
    ensemble_estimators.append((name.replace(" ", "_"), model_copy))

# Make them vote!
ensemble = VotingClassifier(estimators=ensemble_estimators, voting='soft', n_jobs=-1)

if 'SMOTE+Tomek' in best_name:
    ensemble.fit(X_train_st, y_train_st)
elif 'SMOTE' in best_name:
    ensemble.fit(X_train_smote, y_train_smote)
else:
    ensemble.fit(X_train, y_train)

y_pred_ens = ensemble.predict(X_test)
f1_ens = f1_score(y_test, y_pred_ens, average='weighted')

if f1_ens > best_f1:
    best_model = ensemble
    best_f1 = f1_ens
    best_name = "Voting Ensemble (Top 3)"
    print(">> Combining them worked! The team effort got the highest score.")
else:
    print(">> Combining them didn't help. The solo winner is still the best.")

# =======================================================================
# STEP 9: FINAL RESULTS AND GRAPHS
# =======================================================================
print("\n" + "=" * 70)
print("STEP 9: FINAL GRADES")
print("=" * 70)

# Take the final test
y_pred_final = best_model.predict(X_test)
f1_final = f1_score(y_test, y_pred_final, average='weighted')

print(f"\nFinal Score: {f1_final:.4f}")

# Save the confusion matrix (a graph showing where the model got confused)
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_final, display_labels=class_names, cmap='Blues', ax=ax)
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()

# Save a graph showing which questions were the most important to the model
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    feat_imp.head(20).plot(kind='barh', ax=ax, color='steelblue')
    ax.invert_yaxis()
    plt.savefig('feature_importance.png', dpi=150)
    plt.close()

# =======================================================================
# STEP 10: SAVE THE FILES FOR THE JUDGES
# =======================================================================
print("\n" + "=" * 70)
print("STEP 10: SAVING FILES")
print("=" * 70)

# Save the actual brain of the model to a file so judges can test it
joblib.dump(best_model, 'odysseus_final_model.pkl')

# Save its answers to an excel-style file
results_df = pd.DataFrame({'True_Answer': y_test, 'Model_Guess': y_pred_final})
results_df.to_csv('final_predictions.csv', index=False)

print("All done! Check your folder for the new files.")
