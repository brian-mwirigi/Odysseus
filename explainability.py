"""
Explainability & Policy Insights - Team Odysseus (DataSprint 2026)
Run AFTER ml_final.py. Generates:
  1. SHAP beeswarm plots (what drives each prediction)
  2. Financial Vulnerability Index (0-100 score per person)
  3. County-level policy map (which counties need intervention)
"""
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load the saved pipeline state from ml_final.py
X_test, y_test, y_pred, class_names, trained, results, X_full, y_full = joblib.load('pipeline_state.pkl')

# ===================================================================
# 1. SHAP EXPLAINABILITY
# ===================================================================
print("=" * 70)
print("SHAP EXPLAINABILITY")
print("=" * 70)

# Pick the best CatBoost model for SHAP (it handles multiclass perfectly and won the leaderboard)
cat_models = {n: s for n, s in results.items() if 'CatBoost' in n}
shap_name = max(cat_models, key=cat_models.get)
shap_model = trained[shap_name]
print(f"Using {shap_name} (F1={cat_models[shap_name]:.4f})")

# Use 500 random test samples for speed
X_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
explainer = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(X_sample)

# Ensure shap_values is handled correctly whether it's a list or a 3D array
if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
    shap_list = [shap_values[:, :, i] for i in range(len(class_names))]
else:
    shap_list = shap_values

# Beeswarm plot for each class
for i, cls in enumerate(class_names):
    print(f"  Generating SHAP for '{cls}'...")
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_list[i], X_sample, max_display=15, show=False)
    plt.title(f'What drives "{cls}" predictions', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'shap_{cls.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close()

# Combined bar chart across all classes
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_list, X_sample, class_names=class_names,
                  max_display=15, show=False, plot_type='bar')
plt.title('Feature Importance Across All Classes (SHAP)', fontsize=14)
plt.tight_layout()
plt.savefig('shap_all_classes.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: shap_improved.png, shap_stayed_the_same.png, shap_worsened.png, shap_all_classes.png")

# ===================================================================
# 2. FINANCIAL VULNERABILITY INDEX (0-100)
# ===================================================================
print("\n" + "=" * 70)
print("FINANCIAL VULNERABILITY INDEX")
print("=" * 70)

# Get the probability of "Worsened" for EVERY person in the dataset
# This turns our classifier into a continuous vulnerability score
from preprocessing import load_and_preprocess
X_all, y_all, _, _ = load_and_preprocess()

# Use the best XGBoost model to get probabilities
proba = shap_model.predict_proba(X_all)
worsened_idx = class_names.index('Worsened')

# Scale the "Worsened" probability to 0-100
vulnerability = (proba[:, worsened_idx] * 100).round(1)

# Load original data for county names
df_orig = pd.read_csv('finaccess2024_cleaned.csv')
df_orig['vulnerability_score'] = vulnerability

print(f"  Score range: {vulnerability.min():.1f} - {vulnerability.max():.1f}")
print(f"  Mean score: {vulnerability.mean():.1f}")
print(f"  People above 70 (High Risk): {(vulnerability > 70).sum()} ({(vulnerability > 70).mean()*100:.1f}%)")
print(f"  People below 30 (Low Risk): {(vulnerability < 30).sum()} ({(vulnerability < 30).mean()*100:.1f}%)")

# Distribution plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(vulnerability, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
ax.axvline(vulnerability.mean(), color='red', linestyle='--', label=f'Mean: {vulnerability.mean():.1f}')
ax.axvline(70, color='orange', linestyle='--', label='High Risk Threshold (70)')
ax.set_xlabel('Vulnerability Score (0-100)', fontsize=12)
ax.set_ylabel('Number of People', fontsize=12)
ax.set_title('Financial Vulnerability Distribution Across Kenya', fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('vulnerability_distribution.png', dpi=150)
plt.close()
print("  Saved: vulnerability_distribution.png")

# ===================================================================
# 3. COUNTY-LEVEL POLICY MAP
# ===================================================================
print("\n" + "=" * 70)
print("COUNTY-LEVEL POLICY ANALYSIS")
print("=" * 70)

county_stats = df_orig.groupby('county').agg(
    population=('vulnerability_score', 'count'),
    avg_vulnerability=('vulnerability_score', 'mean'),
    high_risk_count=('vulnerability_score', lambda x: (x > 70).sum()),
    worsened_rate=('financial_status', lambda x: (x == 'Worsened').mean()),
).round(2)

county_stats['high_risk_pct'] = (county_stats['high_risk_count'] / county_stats['population'] * 100).round(1)
county_stats = county_stats.sort_values('avg_vulnerability', ascending=False)

print("\nTop 10 Most Vulnerable Counties:")
print(f"  {'County':<20} {'Avg Score':>10} {'High Risk %':>12} {'Worsened %':>12} {'Population':>12}")
print("  " + "-" * 68)
for county, row in county_stats.head(10).iterrows():
    print(f"  {county:<20} {row['avg_vulnerability']:>10.1f} {row['high_risk_pct']:>11.1f}% {row['worsened_rate']*100:>11.1f}% {int(row['population']):>12}")

print("\nTop 10 Least Vulnerable Counties:")
for county, row in county_stats.tail(10).iterrows():
    print(f"  {county:<20} {row['avg_vulnerability']:>10.1f} {row['high_risk_pct']:>11.1f}% {row['worsened_rate']*100:>11.1f}% {int(row['population']):>12}")

# Bar chart of county vulnerability
fig, ax = plt.subplots(figsize=(14, 10))
colors = ['#d32f2f' if v > 60 else '#ff9800' if v > 50 else '#4caf50'
          for v in county_stats['avg_vulnerability']]
county_stats['avg_vulnerability'].plot(kind='barh', ax=ax, color=colors)
ax.set_xlabel('Average Vulnerability Score', fontsize=12)
ax.set_title('Financial Vulnerability by County', fontsize=14)
ax.axvline(county_stats['avg_vulnerability'].mean(), color='black', linestyle='--', alpha=0.5, label='National Average')
ax.legend()
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('county_vulnerability.png', dpi=150)
plt.close()

# Save full county data
county_stats.to_csv('county_vulnerability.csv')
df_orig[['county', 'vulnerability_score', 'financial_status']].to_csv('vulnerability_scores.csv', index=False)

print("\n  Saved: county_vulnerability.png, county_vulnerability.csv, vulnerability_scores.csv")

print(f"\n{'='*70}")
print("EXPLAINABILITY COMPLETE")
print(f"{'='*70}")
