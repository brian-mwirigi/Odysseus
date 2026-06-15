"""
Policy Intervention Simulator & Persona Profiles - Team Odysseus
This is the weapon nobody else has.

It answers: "If we change X, what happens to financial vulnerability?"
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preprocessing import load_and_preprocess

# Load everything
X_test, y_test, y_pred, class_names, trained, results, X_full, y_full = joblib.load('pipeline_state.pkl')
df_orig = pd.read_csv('finaccess2024_cleaned.csv')

# Use the winning CatBoost model
cat_models = {n: s for n, s in results.items() if 'CatBoost' in n}
model_name = max(cat_models, key=cat_models.get)
model = trained[model_name]
worsened_idx = class_names.index('Worsened')

X_all, y_all, _, _ = load_and_preprocess()

def get_vulnerability(X_data):
    proba = model.predict_proba(X_data)
    return proba[:, worsened_idx] * 100

baseline_scores = get_vulnerability(X_all)
df_orig['baseline_vulnerability'] = baseline_scores

# ===================================================================
# INTERVENTION 1: Give mobile money to everyone who doesn't have it
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
print(f"   Avg vulnerability: {before:.1f} → {after:.1f} (Δ = {after-before:+.1f})")
print(f"   High-risk count:   {high_risk_before} → {high_risk_after} (Δ = {high_risk_after-high_risk_before:+d})")
interventions.append(("Mobile Money\nfor All", before, after, affected_count, high_risk_before, high_risk_after))

# --- Financial Literacy ---
X_sim = X_all.copy()
affected_mask = df_orig['fl_score'].isin(['None correct', 'One correct'])
affected_count = affected_mask.sum()
X_sim.loc[affected_mask, 'fl_score'] = 3  # All correct
X_sim.loc[affected_mask, 'financial_capability'] = X_sim.loc[affected_mask, 'education_level'] * 3
sim_scores = get_vulnerability(X_sim)
before = baseline_scores[affected_mask].mean()
after = sim_scores[affected_mask].mean()
high_risk_before = (baseline_scores[affected_mask] > 70).sum()
high_risk_after = (sim_scores[affected_mask] > 70).sum()

print(f"\n2. IMPROVE FINANCIAL LITERACY (low-scoring → all correct)")
print(f"   People affected: {affected_count}")
print(f"   Avg vulnerability: {before:.1f} → {after:.1f} (Δ = {after-before:+.1f})")
print(f"   High-risk count:   {high_risk_before} → {high_risk_after} (Δ = {high_risk_after-high_risk_before:+d})")
interventions.append(("Financial Literacy\nProgram", before, after, affected_count, high_risk_before, high_risk_after))

# --- Formal Savings ---
X_sim = X_all.copy()
affected_mask = df_orig['Savings_formal'] == 'Non-usage'
affected_count = affected_mask.sum()
X_sim.loc[affected_mask, 'Savings_formal'] = 1
X_sim.loc[affected_mask, 'total_formal_products'] = X_sim.loc[affected_mask, 'total_formal_products'] + 1
X_sim.loc[affected_mask, 'total_products'] = X_sim.loc[affected_mask, 'total_products'] + 1
X_sim.loc[affected_mask, 'urban_x_formal'] = X_sim.loc[affected_mask, 'location_type'] * X_sim.loc[affected_mask, 'total_formal_products']
sim_scores = get_vulnerability(X_sim)
before = baseline_scores[affected_mask].mean()
after = sim_scores[affected_mask].mean()
high_risk_before = (baseline_scores[affected_mask] > 70).sum()
high_risk_after = (sim_scores[affected_mask] > 70).sum()

print(f"\n3. PROVIDE FORMAL SAVINGS ACCESS (bank/SACCO/MFI)")
print(f"   People affected: {affected_count}")
print(f"   Avg vulnerability: {before:.1f} → {after:.1f} (Δ = {after-before:+.1f})")
print(f"   High-risk count:   {high_risk_before} → {high_risk_after} (Δ = {high_risk_after-high_risk_before:+d})")
interventions.append(("Formal Savings\nAccess", before, after, affected_count, high_risk_before, high_risk_after))

# --- Shock Protection (safety net) ---
X_sim = X_all.copy()
affected_mask = df_orig['experienced_shock'] == 'Yes'
affected_count = affected_mask.sum()
X_sim.loc[affected_mask, 'experienced_shock'] = 0
X_sim.loc[affected_mask, 'shock_vulnerable'] = 0
X_sim.loc[affected_mask, 'age_x_shock'] = 0
sim_scores = get_vulnerability(X_sim)
before = baseline_scores[affected_mask].mean()
after = sim_scores[affected_mask].mean()
high_risk_before = (baseline_scores[affected_mask] > 70).sum()
high_risk_after = (sim_scores[affected_mask] > 70).sum()

print(f"\n4. SHOCK PROTECTION (safety net / insurance for those who experienced shocks)")
print(f"   People affected: {affected_count}")
print(f"   Avg vulnerability: {before:.1f} → {after:.1f} (Δ = {after-before:+.1f})")
print(f"   High-risk count:   {high_risk_before} → {high_risk_after} (Δ = {high_risk_after-high_risk_before:+d})")
interventions.append(("Shock Protection\n/ Insurance", before, after, affected_count, high_risk_before, high_risk_after))

# --- Emergency Fund Access ---
X_sim = X_all.copy()
affected_mask = df_orig['accessto_13k_1month'] == 'No'
affected_count = affected_mask.sum()
X_sim.loc[affected_mask, 'accessto_13k_1month'] = 1
X_sim.loc[affected_mask, 'resilience_score'] = X_sim.loc[affected_mask, 'resilience_score'] + 1
X_sim.loc[affected_mask, 'shock_vulnerable'] = ((X_sim.loc[affected_mask, 'experienced_shock'] == 1) & (X_sim.loc[affected_mask, 'resilience_score'] <= 1)).astype(int)
sim_scores = get_vulnerability(X_sim)
before = baseline_scores[affected_mask].mean()
after = sim_scores[affected_mask].mean()
high_risk_before = (baseline_scores[affected_mask] > 70).sum()
high_risk_after = (sim_scores[affected_mask] > 70).sum()

print(f"\n5. EMERGENCY FUND ACCESS (ensure everyone can access KES 13k in 1 month)")
print(f"   People affected: {affected_count}")
print(f"   Avg vulnerability: {before:.1f} → {after:.1f} (Δ = {after-before:+.1f})")
print(f"   High-risk count:   {high_risk_before} → {high_risk_after} (Δ = {high_risk_after-high_risk_before:+d})")
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
plt.savefig('intervention_impact.png', dpi=150)
plt.close()
print("\nSaved: intervention_impact.png")

# ===================================================================
# PERSONA PROFILES
# ===================================================================
print(f"\n{'='*70}")
print("PERSONA PROFILES")
print("=" * 70)

personas = [
    ("Worsened", 2, "Those whose financial status deteriorated"),
    ("Improved", 0, "Those whose financial status got better"),
    ("Stayed the same", 1, "Those with no change"),
]

for label, encoded, desc in personas:
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
print("TARGETED INTERVENTION: TANA RIVER COUNTY (Most Vulnerable)")
print("=" * 70)

tana = df_orig[df_orig['county'] == 'Tana River']
print(f"  Population in data:       {len(tana)}")
print(f"  % Worsened:               {(tana['financial_status']=='Worsened').mean()*100:.1f}%")
print(f"  % No mobile money:        {(tana['mobile_money_access']=='No').mean()*100:.1f}%")
print(f"  % No formal savings:      {(tana['Savings_formal']=='Non-usage').mean()*100:.1f}%")
print(f"  % Experienced shock:      {(tana['experienced_shock']=='Yes').mean()*100:.1f}%")
print(f"  % Cannot access 13k:      {(tana['accessto_13k_1month']=='No').mean()*100:.1f}%")
print(f"  Avg vulnerability score:  {tana['baseline_vulnerability'].mean():.1f}")

# Simulate: give Tana River everything
X_tana = X_all.copy()
tana_mask = df_orig['county'] == 'Tana River'
X_tana.loc[tana_mask, 'mobile_money_access'] = 1
X_tana.loc[tana_mask, 'Savings_formal'] = 1
X_tana.loc[tana_mask, 'accessto_13k_1month'] = 1
X_tana.loc[tana_mask, 'resilience_score'] = 3
X_tana.loc[tana_mask, 'shock_vulnerable'] = 0
X_tana.loc[tana_mask, 'total_formal_products'] = X_tana.loc[tana_mask, 'total_formal_products'] + 1
X_tana.loc[tana_mask, 'total_products'] = X_tana.loc[tana_mask, 'total_products'] + 1
X_tana.loc[tana_mask, 'digital_access'] = 1

tana_after = get_vulnerability(X_tana)
before_tana = baseline_scores[tana_mask].mean()
after_tana = tana_after[tana_mask].mean()
hr_before = (baseline_scores[tana_mask] > 70).sum()
hr_after = (tana_after[tana_mask] > 70).sum()

print(f"\n  COMBINED INTERVENTION (mobile money + savings + emergency fund):")
print(f"  Avg vulnerability: {before_tana:.1f} → {after_tana:.1f} (Δ = {after_tana-before_tana:+.1f})")
print(f"  High-risk people:  {hr_before} → {hr_after} (Δ = {hr_after-hr_before:+d})")
print(f"  High-risk reduction: {(1 - hr_after/hr_before)*100:.1f}%")

print(f"\n{'='*70}")
print("SIMULATION COMPLETE")
print(f"{'='*70}")
