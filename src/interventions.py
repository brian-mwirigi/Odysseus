"""
Policy Intervention Simulator & Persona Profiles - Team Odysseus

It answers: "If we change X, what happens to financial vulnerability?"

FIXED: Uses X_all_final from pipeline_state.pkl (already has county_worsened_rate,
       county dropped, all preprocessing done) instead of load_and_preprocess()
       which would return data incompatible with the trained models.
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

state = joblib.load('models/pipeline_state.pkl')
X_test_final = state['X_test_final']
y_test = state['y_test']
class_names = state['class_names']
trained = state['trained']
test_scores = state['test_scores']
X_all = state['X_all_final']
y_full = state['y_full']
te_maps = state['te_maps']
cat_cols_to_te = state['cat_cols_to_te']
worsened_cls = state['worsened_cls']
threshold_weights = state['threshold_weights']
best_name = state['best_name']

county_means = te_maps.get('county', te_maps.get('county', {})).get('means', {})
county_overall_mean = te_maps.get('county', {}).get('overall', 0.52)

df_orig = pd.read_csv('data/finaccess2024_cleaned.csv')

shap_model_name = "CatBoost"
if shap_model_name not in trained:
    shap_model_name = best_name.replace("+Threshold", "")
model = trained[shap_model_name]
worsened_idx = class_names.index('Worsened')
print(f"Using {shap_model_name} (Test F1={test_scores.get(shap_model_name, 0):.4f})")


def get_vulnerability(X_data):
    proba = model.predict_proba(X_data)
    return proba[:, worsened_idx] * 100


baseline_scores = get_vulnerability(X_all)
df_orig['baseline_vulnerability'] = baseline_scores

# ===================================================================
# INTERVENTION SIMULATIONS
# ===================================================================
print("=" * 70)
print("POLICY INTERVENTION SIMULATIONS")
print("=" * 70)

interventions = []

# --- Mobile Money ---
X_sim = X_all.copy()
affected_mask = df_orig['mobile_money_access'] == 'No'
affected_count = affected_mask.sum()
X_sim.loc[affected_mask, 'mobile_money_access'] = 1
X_sim.loc[affected_mask, 'digital_access'] = X_sim.loc[affected_mask, 'mobile_ownership_1']
sim_scores = get_vulnerability(X_sim)
before = baseline_scores[affected_mask].mean()
after = sim_scores[affected_mask].mean()
high_risk_before = (baseline_scores[affected_mask] > 70).sum()
high_risk_after = (sim_scores[affected_mask] > 70).sum()

print(f"\n1. GIVE MOBILE MONEY ACCESS TO ALL EXCLUDED ADULTS")
print(f"   People affected: {affected_count}")
print(f"   Avg vulnerability: {before:.1f} -> {after:.1f} (delta = {after-before:+.1f})")
print(f"   High-risk count:   {high_risk_before} -> {high_risk_after} (delta = {high_risk_after-high_risk_before:+d})")
interventions.append(("Mobile Money\nfor All", before, after, affected_count, high_risk_before, high_risk_after))

# --- Financial Literacy ---
X_sim = X_all.copy()
affected_mask = df_orig['fl_score'].isin(['None correct', 'One correct'])
affected_count = affected_mask.sum()
X_sim.loc[affected_mask, 'fl_score'] = 3
X_sim.loc[affected_mask, 'financial_capability'] = X_sim.loc[affected_mask, 'education_level'] * 3
sim_scores = get_vulnerability(X_sim)
before = baseline_scores[affected_mask].mean()
after = sim_scores[affected_mask].mean()
high_risk_before = (baseline_scores[affected_mask] > 70).sum()
high_risk_after = (sim_scores[affected_mask] > 70).sum()

print(f"\n2. IMPROVE FINANCIAL LITERACY (low-scoring -> all correct)")
print(f"   People affected: {affected_count}")
print(f"   Avg vulnerability: {before:.1f} -> {after:.1f} (delta = {after-before:+.1f})")
print(f"   High-risk count:   {high_risk_before} -> {high_risk_after} (delta = {high_risk_after-high_risk_before:+d})")
interventions.append(("Financial Literacy\nProgram", before, after, affected_count, high_risk_before, high_risk_after))

# --- Formal Savings ---
X_sim = X_all.copy()
affected_mask = df_orig['Savings_formal'] == 'Non-usage'
affected_count = affected_mask.sum()
X_sim.loc[affected_mask, 'Savings_formal'] = 1
X_sim.loc[affected_mask, 'total_formal_products'] = X_sim.loc[affected_mask, 'total_formal_products'] + 1
X_sim.loc[affected_mask, 'total_products'] = X_sim.loc[affected_mask, 'total_products'] + 1
X_sim.loc[affected_mask, 'no_savings_no_access'] = 0
X_sim.loc[affected_mask, 'urban_x_formal'] = X_sim.loc[affected_mask, 'location_type'] * X_sim.loc[affected_mask, 'total_formal_products']
sim_scores = get_vulnerability(X_sim)
before = baseline_scores[affected_mask].mean()
after = sim_scores[affected_mask].mean()
high_risk_before = (baseline_scores[affected_mask] > 70).sum()
high_risk_after = (sim_scores[affected_mask] > 70).sum()

print(f"\n3. PROVIDE FORMAL SAVINGS ACCESS (bank/SACCO/MFI)")
print(f"   People affected: {affected_count}")
print(f"   Avg vulnerability: {before:.1f} -> {after:.1f} (delta = {after-before:+.1f})")
print(f"   High-risk count:   {high_risk_before} -> {high_risk_after} (delta = {high_risk_after-high_risk_before:+d})")
interventions.append(("Formal Savings\nAccess", before, after, affected_count, high_risk_before, high_risk_after))

# --- Shock Protection (safety net) ---
X_sim = X_all.copy()
affected_mask = df_orig['experienced_shock'] == 'Yes'
affected_count = affected_mask.sum()
X_sim.loc[affected_mask, 'experienced_shock'] = 0
X_sim.loc[affected_mask, 'shock_vulnerable'] = 0
X_sim.loc[affected_mask, 'age_x_shock'] = 0
X_sim.loc[affected_mask, 'shock_x_defaulted'] = 0
X_sim.loc[affected_mask, 'shock_no_resilience'] = 0
X_sim.loc[affected_mask, 'shock_x_no_savings'] = 0
X_sim.loc[affected_mask, 'disability_x_shock'] = 0
sim_scores = get_vulnerability(X_sim)
before = baseline_scores[affected_mask].mean()
after = sim_scores[affected_mask].mean()
high_risk_before = (baseline_scores[affected_mask] > 70).sum()
high_risk_after = (sim_scores[affected_mask] > 70).sum()

print(f"\n4. SHOCK PROTECTION (safety net / insurance for those who experienced shocks)")
print(f"   People affected: {affected_count}")
print(f"   Avg vulnerability: {before:.1f} -> {after:.1f} (delta = {after-before:+.1f})")
print(f"   High-risk count:   {high_risk_before} -> {high_risk_after} (delta = {high_risk_after-high_risk_before:+d})")
interventions.append(("Shock Protection\n/ Insurance", before, after, affected_count, high_risk_before, high_risk_after))

# --- Emergency Fund Access ---
X_sim = X_all.copy()
affected_mask = df_orig['accessto_13k_1month'] == 'No'
affected_count = affected_mask.sum()
X_sim.loc[affected_mask, 'accessto_13k_1month'] = 1
X_sim.loc[affected_mask, 'resilience_score'] = X_sim.loc[affected_mask, 'resilience_score'] + 1
X_sim.loc[affected_mask, 'shock_no_resilience'] = 0
X_sim.loc[affected_mask, 'no_savings_no_access'] = (
    (X_sim.loc[affected_mask, 'Savings_formal'] == 0) &
    (X_sim.loc[affected_mask, 'accessto_13k_1month'] == 0)
).astype(int)
X_sim.loc[affected_mask, 'shock_vulnerable'] = (
    (X_sim.loc[affected_mask, 'experienced_shock'] == 1) &
    (X_sim.loc[affected_mask, 'resilience_score'] <= 1)
).astype(int)
sim_scores = get_vulnerability(X_sim)
before = baseline_scores[affected_mask].mean()
after = sim_scores[affected_mask].mean()
high_risk_before = (baseline_scores[affected_mask] > 70).sum()
high_risk_after = (sim_scores[affected_mask] > 70).sum()

print(f"\n5. EMERGENCY FUND ACCESS (ensure everyone can access KES 13k in 1 month)")
print(f"   People affected: {affected_count}")
print(f"   Avg vulnerability: {before:.1f} -> {after:.1f} (delta = {after-before:+.1f})")
print(f"   High-risk count:   {high_risk_before} -> {high_risk_after} (delta = {high_risk_after-high_risk_before:+d})")
interventions.append(("Emergency Fund\nAccess", before, after, affected_count, high_risk_before, high_risk_after))

# --- Plot ---
fig, ax = plt.subplots(figsize=(14, 7))
names = [i[0] for i in interventions]
befores = [i[1] for i in interventions]
afters = [i[2] for i in interventions]
x = np.arange(len(names))
w = 0.35

bars1 = ax.bar(x - w/2, befores, w, label='Before Intervention', color='#e53935', alpha=0.85)
bars2 = ax.bar(x + w/2, afters, w, label='After Intervention', color='#43a047', alpha=0.85)

for bar, val in zip(bars1, befores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
for bar, val in zip(bars2, afters):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel('Average Vulnerability Score', fontsize=12)
ax.set_title('Impact of Policy Interventions on Financial Vulnerability', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10)
ax.legend(fontsize=11)
ax.set_ylim(0, max(befores) + 10)
plt.tight_layout()
plt.savefig('visualizations/intervention_impact.png', dpi=150)
plt.close()
print("\nSaved: intervention_impact.png")

# ===================================================================
# PERSONA PROFILES
# ===================================================================
print(f"\n{'='*70}")
print("PERSONA PROFILES")
print("=" * 70)

personas = [
    ("Worsened", "Those whose financial status deteriorated"),
    ("Improved", "Those whose financial status got better"),
    ("Stayed the same", "Those with no change"),
]

for label, desc in personas:
    subset = df_orig[df_orig['financial_status'] == label]
    print(f"\n--- TYPICAL PROFILE: '{label}' ({desc}) ---")
    print(f"  Most common age group:     {subset['Age'].mode().iloc[0]}")
    print(f"  Most common sex:           {subset['Sex'].mode().iloc[0]}")
    print(f"  Most common marital status:{subset['marital_status'].mode().iloc[0]}")
    print(f"  Most common location:      {subset['location_type'].mode().iloc[0]}")
    print(f"  Median monthly income:     KES {subset['monthly_income'].median():,.0f}")
    print(f"  Experienced shock:         {(subset['experienced_shock']=='Yes').mean()*100:.1f}%")
    print(f"  Has mobile money:          {(subset['mobile_money_access']=='Yes').mean()*100:.1f}%")
    print(f"  Uses formal savings:       {(subset['Savings_formal']=='Usage').mean()*100:.1f}%")
    print(f"  Defaulted on loan:         {(subset['defaulted']=='Yes').mean()*100:.1f}%")
    print(f"  Can access KES 13k:        {(subset['accessto_13k_1month']=='Yes').mean()*100:.1f}%")
    print(f"  Avg vulnerability score:   {subset['baseline_vulnerability'].mean():.1f}")
    top_counties = subset['county'].value_counts().head(3)
    print(f"  Top 3 counties:            {', '.join(top_counties.index)}")

# --- County-specific intervention ---
print(f"\n{'='*70}")
print("TARGETED INTERVENTION: MOST VULNERABLE COUNTY")
print("=" * 70)

county_vuln = df_orig.groupby('county')['baseline_vulnerability'].mean().sort_values(ascending=False)
most_vulnerable_county = county_vuln.index[0]
print(f"  Most vulnerable county: {most_vulnerable_county} (avg score={county_vuln.iloc[0]:.1f})")

target = df_orig[df_orig['county'] == most_vulnerable_county]
print(f"  Population in data:       {len(target)}")
print(f"  % Worsened:               {(target['financial_status']=='Worsened').mean()*100:.1f}%")
print(f"  % No mobile money:        {(target['mobile_money_access']=='No').mean()*100:.1f}%")
print(f"  % No formal savings:      {(target['Savings_formal']=='Non-usage').mean()*100:.1f}%")
print(f"  % Experienced shock:      {(target['experienced_shock']=='Yes').mean()*100:.1f}%")
print(f"  % Cannot access 13k:      {(target['accessto_13k_1month']=='No').mean()*100:.1f}%")
print(f"  Avg vulnerability score:  {target['baseline_vulnerability'].mean():.1f}")

X_target = X_all.copy()
target_mask = df_orig['county'] == most_vulnerable_county
X_target.loc[target_mask, 'mobile_money_access'] = 1
X_target.loc[target_mask, 'Savings_formal'] = 1
X_target.loc[target_mask, 'accessto_13k_1month'] = 1
X_target.loc[target_mask, 'resilience_score'] = 3
X_target.loc[target_mask, 'shock_vulnerable'] = 0
X_target.loc[target_mask, 'total_formal_products'] = X_target.loc[target_mask, 'total_formal_products'] + 1
X_target.loc[target_mask, 'total_products'] = X_target.loc[target_mask, 'total_products'] + 1
X_target.loc[target_mask, 'digital_access'] = 1
X_target.loc[target_mask, 'no_savings_no_access'] = 0
X_target.loc[target_mask, 'shock_no_resilience'] = 0

target_after = get_vulnerability(X_target)
before_target = baseline_scores[target_mask].mean()
after_target = target_after[target_mask].mean()
hr_before = (baseline_scores[target_mask] > 70).sum()
hr_after = (target_after[target_mask] > 70).sum()

print(f"\n  COMBINED INTERVENTION (mobile money + savings + emergency fund):")
print(f"  Avg vulnerability: {before_target:.1f} -> {after_target:.1f} (delta = {after_target-before_target:+.1f})")
if hr_before > 0:
    print(f"  High-risk people:  {hr_before} -> {hr_after} (delta = {hr_after-hr_before:+d})")
    print(f"  High-risk reduction: {(1 - hr_after/hr_before)*100:.1f}%")
else:
    print(f"  High-risk people:  {hr_before} -> {hr_after}")

print(f"\n{'='*70}")
print("SIMULATION COMPLETE")
print(f"{'='*70}")
