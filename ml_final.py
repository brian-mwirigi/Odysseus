"""
ML Pipeline - Team Odysseus (DataSprint 2026)
Trains, evaluates, and exports the best model.
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from preprocessing import load_and_preprocess

# --- Load ---
print("=" * 70)
print("LOADING DATA")
print("=" * 70)
X, y, class_names, le_target = load_and_preprocess()
print(f"{X.shape[0]} rows, {X.shape[1]} features")

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Resample ---
print("\nResampling...")
X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)
X_st, y_st = SMOTETomek(random_state=42).fit_resample(X_train, y_train)

# --- Train all models ---
print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)

configs = [
    ("XGB+ST", XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.7, min_child_weight=3, reg_lambda=1.0, gamma=0.1, random_state=42, n_jobs=-1), X_st, y_st),
    ("XGB+SMOTE", XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.7, min_child_weight=3, reg_lambda=1.0, gamma=0.1, random_state=42, n_jobs=-1), X_smote, y_smote),
    ("XGB+raw", XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.7, min_child_weight=3, reg_lambda=1.0, gamma=0.1, random_state=42, n_jobs=-1), X_train, y_train),
    ("LGBM+ST", LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31, subsample=0.8, colsample_bytree=0.7, min_child_samples=20, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1), X_st, y_st),
    ("LGBM+SMOTE", LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31, subsample=0.8, colsample_bytree=0.7, min_child_samples=20, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1), X_smote, y_smote),
    ("GB+ST", GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42), X_st, y_st),
    ("RF+balanced", RandomForestClassifier(n_estimators=500, class_weight='balanced', max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1), X_train, y_train),
    ("CatBoost+ST", CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, random_seed=42, verbose=0, auto_class_weights='Balanced'), X_st, y_st),
    ("CatBoost+raw", CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, random_seed=42, verbose=0, auto_class_weights='Balanced'), X_train, y_train),
]

results = {}
trained = {}

for name, model, tx, ty in configs:
    print(f"  {name}...", end=" ", flush=True)
    model.fit(tx, ty)
    yp = model.predict(X_test)
    f1 = f1_score(y_test, yp, average='weighted')
    results[name] = f1
    trained[name] = model
    print(f"F1={f1:.4f}")

# --- Ensembles ---
print("\n" + "=" * 70)
print("ENSEMBLES")
print("=" * 70)

top3 = sorted(results.items(), key=lambda x: x[1], reverse=True)[:3]

# Voting
vote_est = []
for name, _ in top3:
    cfg = [c for c in configs if c[0] == name][0]
    vote_est.append((name.replace("+", "_"), cfg[1].__class__(**cfg[1].get_params())))
voting = VotingClassifier(estimators=vote_est, voting='soft', n_jobs=-1)

# Train ensemble on the data the best model used
best_cfg = [c for c in configs if c[0] == top3[0][0]][0]
voting.fit(best_cfg[2], best_cfg[3])
f1_vote = f1_score(y_test, voting.predict(X_test), average='weighted')
results["Voting Ensemble"] = f1_vote
trained["Voting Ensemble"] = voting
print(f"  Voting Ensemble: F1={f1_vote:.4f}")

# Stacking
stack_est = []
for name, _ in top3:
    cfg = [c for c in configs if c[0] == name][0]
    stack_est.append((name.replace("+", "_"), cfg[1].__class__(**cfg[1].get_params())))
stacking = StackingClassifier(
    estimators=stack_est,
    final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
    cv=5, n_jobs=-1
)
stacking.fit(best_cfg[2], best_cfg[3])
f1_stack = f1_score(y_test, stacking.predict(X_test), average='weighted')
results["Stacking Ensemble"] = f1_stack
trained["Stacking Ensemble"] = stacking
print(f"  Stacking Ensemble: F1={f1_stack:.4f}")

# --- Final leaderboard ---
print("\n" + "=" * 70)
print("LEADERBOARD")
print("=" * 70)
final = sorted(results.items(), key=lambda x: x[1], reverse=True)
for rank, (name, score) in enumerate(final, 1):
    marker = " <<<< WINNER" if rank == 1 else ""
    print(f"  {rank:2d}. {name:25s} F1={score:.4f}{marker}")

best_name = final[0][0]
best_model = trained[best_name]
y_pred = best_model.predict(X_test)
f1_final = f1_score(y_test, y_pred, average='weighted')

# --- Report ---
print(f"\n{'='*70}")
print(f"FINAL: {best_name} — F1={f1_final:.4f}")
print(f"{'='*70}")
print(classification_report(y_test, y_pred, target_names=class_names))

# --- Save confusion matrix ---
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names, cmap='Blues', ax=ax)
ax.set_title(f'{best_name} (F1={f1_final:.4f})')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()

# --- Export ---
joblib.dump(best_model, 'odysseus_final_model.pkl')
joblib.dump((X_test, y_test, y_pred, class_names, trained, results, X, y), 'pipeline_state.pkl')

import pandas as pd
pd.DataFrame({'True': np.ravel(y_test), 'Predicted': np.ravel(y_pred)}).to_csv('final_predictions.csv', index=False)
pd.DataFrame(final, columns=['Model', 'F1']).to_csv('model_leaderboard.csv', index=False)

print("\nSaved: odysseus_final_model.pkl, final_predictions.csv, confusion_matrix.png")
print("Run explainability.py next for SHAP + Vulnerability Index.")
