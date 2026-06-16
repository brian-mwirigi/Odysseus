"""
Verify findings - Team Odysseus
Uses pipeline_state.pkl to avoid data leakage.
"""
import pandas as pd
import numpy as np
import joblib

state = joblib.load('pipeline_state.pkl')
X_test_final = state['X_test_final']
y_test = state['y_test']
class_names = state['class_names']
trained = state['trained']
test_scores = state['test_scores']
X_all = state['X_all_final']
y_full = state['y_full']
threshold_weights = state['threshold_weights']

df = pd.read_csv('finaccess2024_cleaned.csv')

model = trained["CatBoost"]
worsened_idx = class_names.index('Worsened')
print(f"Using CatBoost (Test F1={test_scores.get('CatBoost', 0):.4f})")

proba = model.predict_proba(X_all)
vuln = proba[:, worsened_idx] * 100
df['vuln'] = vuln

print('=== VULNERABILITY INDEX ===')
print(f'Range: {vuln.min():.1f} - {vuln.max():.1f}')
print(f'Mean: {vuln.mean():.1f}')
print(f'Median: {np.median(vuln):.1f}')
print(f'High risk (>70): {(vuln>70).sum()} ({(vuln>70).mean()*100:.1f}%)')
print(f'Low risk (<30): {(vuln<30).sum()} ({(vuln<30).mean()*100:.1f}%)')

print('\n=== COUNTY RANKINGS (TOP 10 MOST VULNERABLE) ===')
cs = df.groupby('county').agg(
    n=('vuln','count'),
    avg_vuln=('vuln','mean'),
    hr=('vuln', lambda x: (x>70).sum()),
    worsened=('financial_status', lambda x: (x=='Worsened').mean())
).sort_values('avg_vuln', ascending=False)
cs['hr_pct'] = (cs['hr']/cs['n']*100).round(1)
for c, r in cs.head(10).iterrows():
    print(f"  {c:20s} avg={r['avg_vuln']:.1f}  high_risk={r['hr_pct']:.1f}%  worsened={r['worsened']*100:.1f}%  n={int(r['n'])}")

print('\n=== COUNTY RANKINGS (TOP 10 LEAST VULNERABLE) ===')
for c, r in cs.tail(10).iterrows():
    print(f"  {c:20s} avg={r['avg_vuln']:.1f}  high_risk={r['hr_pct']:.1f}%  worsened={r['worsened']*100:.1f}%  n={int(r['n'])}")

print('\n=== PERSONA: WORSENED ===')
w = df[df['financial_status']=='Worsened']
print(f"  Age: {w['Age'].mode().iloc[0]}")
print(f"  Sex: {w['Sex'].mode().iloc[0]}")
print(f"  Marital: {w['marital_status'].mode().iloc[0]}")
print(f"  Location: {w['location_type'].mode().iloc[0]}")
print(f"  Median income: KES {w['monthly_income'].median():,.0f}")
print(f"  Shock: {(w['experienced_shock']=='Yes').mean()*100:.1f}%")
print(f"  Mobile money: {(w['mobile_money_access']=='Yes').mean()*100:.1f}%")
print(f"  Formal savings: {(w['Savings_formal']=='Usage').mean()*100:.1f}%")
print(f"  Defaulted: {(w['defaulted']=='Yes').mean()*100:.1f}%")
print(f"  Can access 13k: {(w['accessto_13k_1month']=='Yes').mean()*100:.1f}%")

print('\n=== PERSONA: IMPROVED ===')
im = df[df['financial_status']=='Improved']
print(f"  Age: {im['Age'].mode().iloc[0]}")
print(f"  Sex: {im['Sex'].mode().iloc[0]}")
print(f"  Marital: {im['marital_status'].mode().iloc[0]}")
print(f"  Location: {im['location_type'].mode().iloc[0]}")
print(f"  Median income: KES {im['monthly_income'].median():,.0f}")
print(f"  Shock: {(im['experienced_shock']=='Yes').mean()*100:.1f}%")
print(f"  Mobile money: {(im['mobile_money_access']=='Yes').mean()*100:.1f}%")
print(f"  Formal savings: {(im['Savings_formal']=='Usage').mean()*100:.1f}%")
print(f"  Defaulted: {(im['defaulted']=='Yes').mean()*100:.1f}%")
print(f"  Can access 13k: {(im['accessto_13k_1month']=='Yes').mean()*100:.1f}%")

print('\n=== PERSONA: STAYED THE SAME ===')
st = df[df['financial_status']=='Stayed the same']
print(f"  Age: {st['Age'].mode().iloc[0]}")
print(f"  Sex: {st['Sex'].mode().iloc[0]}")
print(f"  Marital: {st['marital_status'].mode().iloc[0]}")
print(f"  Location: {st['location_type'].mode().iloc[0]}")
print(f"  Median income: KES {st['monthly_income'].median():,.0f}")
print(f"  Shock: {(st['experienced_shock']=='Yes').mean()*100:.1f}%")
print(f"  Mobile money: {(st['mobile_money_access']=='Yes').mean()*100:.1f}%")
print(f"  Formal savings: {(st['Savings_formal']=='Usage').mean()*100:.1f}%")
print(f"  Defaulted: {(st['defaulted']=='Yes').mean()*100:.1f}%")
print(f"  Can access 13k: {(st['accessto_13k_1month']=='Yes').mean()*100:.1f}%")

print('\n=== KEY DATASET STATS ===')
print(f"Observations: {len(df)}")
print(f"Counties: {df['county'].nunique()}")
print(f"Mobile money: {(df['mobile_money_access']=='Yes').mean()*100:.1f}%")
print(f"Experienced shock: {(df['experienced_shock']=='Yes').mean()*100:.1f}%")
print(f"Defaulted: {(df['defaulted']=='Yes').mean()*100:.1f}%")
print(f"Formal savings: {(df['Savings_formal']=='Usage').mean()*100:.1f}%")
print(f"Can access 13k: {(df['accessto_13k_1month']=='Yes').mean()*100:.1f}%")
print(f"Urban: {(df['location_type']=='Urban').mean()*100:.1f}%")
print(f"Female: {(df['Sex']=='Female').mean()*100:.1f}%")
print(f"Median income: KES {df['monthly_income'].median():,.0f}")
