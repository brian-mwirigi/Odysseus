# STRATHMORE DATA COMMUNITY
## DataSprint 2026 — Participant Brief
*What You Need to Know to Get Started*

---

## Welcome

You are about to spend one week working with real data about real Kenyans. The dataset you will use comes from the 2024 FinAccess Household Survey — the most comprehensive survey of financial inclusion in Kenya, published in December 2024 by the Central Bank of Kenya, KNBS, and FSD Kenya.

Your task is to build a machine learning model that predicts whether a Kenyan adult's financial situation has **Improved**, **Stayed the same**, or **Worsened**, and to explain what drives the difference. This is a multiclass classification problem.

---

## Dataset Access

| Source | Link |
|---|---|
| Kaggle (recommended — run notebooks directly in browser) | https://www.kaggle.com/datasets/davidpbriggs/kenya-finaccess-household-survey-2024 |
| Official FinAccess site (full report + data manual) | https://finaccess.knbs.or.ke/reports-and-datasets |

**Download the curated file: `finaccess2024_datasprint.csv` (20,871 rows × 28 columns)**

---

## The Data — Quick Reference

| Category | Columns Included |
|---|---|
| Demographics | `county`, `location_type`, `Sex`, `Age`, `household_size`, `education_level`, `marital_status`, `has_disability` |
| Livelihood | `monthly_income`, `experienced_shock` |
| Mobile & digital | `mobile_ownership_1`, `mobile_money_access`, `barriers_mobile_money` |
| Financial behaviour | `Savings_formal`, `Savings_informal`, `Loan_formal`, `Loan_informal`, `defaulted`, `formal_service_use`, `barriers_bank`, `prodsum1` |
| Financial health | `fl_score`, `nfhi_11`, `nfhi_12`, `nfhi_13`, `accessto_13k_1month`, `not_difficult` |
| **TARGET** | `financial_status` → Worsened (52.6%) \| Stayed the same (26.9%) \| Improved (20.5%) |

---

## Missing Values

| Column | Missing | What to Do |
|---|---|---|
| `barriers_bank` | 27.5% | These are people who have a bank — no barrier. Fill missing values with `'No barrier'` before encoding. |
| `monthly_income` | ~0% | Already imputed with median in the curated file. No action needed. |

---

## Suggested Workflow

| Step | Task | Tools / Tips |
|---|---|---|
| 1 | Load and inspect the data | `pd.read_csv()`, `df.shape`, `df.dtypes`, `df.isnull().sum()` |
| 2 | Explore — EDA | `df.value_counts()`, `sns.countplot()`, `sns.heatmap()` for correlations |
| 3 | Clean and preprocess | Fill missing values, `LabelEncoder` or `pd.get_dummies()` for categoricals |
| 4 | Split into train and test | `train_test_split(df, test_size=0.2, random_state=42)` |
| 5 | Train a model | Start with `LogisticRegression` or `DecisionTreeClassifier` |
| 6 | Evaluate | `f1_score(y_test, y_pred, average='weighted')` — not just accuracy |
| 7 | Interpret | `feature_importances_`, coefficients, or SHAP values |
| 8 | Build your slides | 7 slides, tell the story, keep it visual |

---

## Starter Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

df = pd.read_csv('finaccess2024_datasprint.csv')
print(df.shape)  # (20871, 28)
print(df['financial_status'].value_counts())

# Always evaluate with weighted F1 - not accuracy!
score = f1_score(y_test, y_pred, average='weighted')
print(f'Weighted F1: {score:.3f}')
```

---

## Required Deliverables

| Item | Required | Notes |
|---|---|---|
| Jupyter Notebook (`.ipynb`) | Yes | Clean, commented, reproducible. Includes full pipeline from loading to evaluation. |
| Data Visualisations | Yes | Minimum 5 labelled charts in the notebook. Tell a visual story about Kenya. |
| 7-Slide PowerPoint (`.pptx`) | Yes | Follow the required slide structure below. Keep slides visual — not text-heavy. |

---

## Required Slide Structure

| Slide | Title | What to Include |
|---|---|---|
| 1 | Problem Statement | What problem are you solving? Why does it matter for Kenya? |
| 2 | Dataset Overview | Key statistics, feature categories, target distribution chart |
| 3 | Key EDA Finding | Your most interesting insight from data exploration |
| 4 | Modelling Approach | Which model? Why? How did you handle preprocessing and class imbalance? |
| 5 | Model Performance | Weighted F1-score, classification report, confusion matrix |
| 6 | Key Drivers | Which features predict financial status most strongly? |
| 7 | Recommendations | What should policymakers, banks, or NGOs do based on your findings? |

---

## Tips From iLab Africa

- Start with a simple model first — understand it before trying something complex.
- Visualise before you model — charts will tell you more about the data than any algorithm.
- Think about what the features mean in real life. A `monthly_income` of KES 1,000 is very different from KES 50,000.
- Class imbalance is real — over half the respondents said 'Worsened'. Make sure your model learns all three classes.
- The guiding question matters as much as your F1 score — judges will ask you what your model tells policymakers. Know your answer.
- `barriers_bank` missing values are not random — they represent people who already have bank accounts. Think about how to encode this meaningfully.

---

## Key Dates

| Date | Milestone |
|---|---|
| Thursday, 11th June 2026 \| 4:00 PM | Registration closes |
| Friday, 12th June 2026 \| 2:00 – 4:30 PM | Kickoff & Info Session \| MSB 1, Strathmore University |
| Tuesday, 16th June 2026 \| 12:00 Midday | Submission deadline |
| Wednesday – Friday, 17th – 19th June 2026 | Review period — iLab Africa reviews submissions and provides feedback |
| Friday, 19th June 2026 \| 2:00 – 5:00 PM | Workshop, Final Presentations, Awards & Networking \| MSB 1, Strathmore University |

---

## Contact

**Philip Tait** | Chairperson, Strathmore Data Community  
📧 datacommunity@strathmore.edu | 📞 0716 914 156 | 🔗 linktr.ee/strathmoredatacommunity

---

*Learn. Build. Launch.*

*The Strathmore Data Community is an officially recognised student community at Strathmore University, Nairobi, supported by SCES and SIMS.*
