# STRATHMORE DATA COMMUNITY
## DataSprint 2026 — Problem Statement
*One Week. Real Data. Real Impact. | SDC × iLab Africa*

---

## Background

Kenya has made significant strides in financial inclusion over the past decade. Mobile money has transformed how millions of Kenyans save, borrow, and transact, yet the picture on the ground remains uneven. According to the 2024 FinAccess Household Survey, conducted by the Central Bank of Kenya, KNBS, and FSD Kenya, 9.9% of Kenyan adults remain fully excluded from financial services. More significantly, over half of those surveyed (52.6%) reported that their financial situation has worsened compared to the previous year.

Behind every data point is a real person; a smallholder farmer who could not repay a loan after a drought, a young woman in Turkana with no national ID and therefore no bank account, a household in Nairobi spending more than it earns. Understanding who is struggling financially and why — is one of the most important questions facing Kenya's policymakers, banks, NGOs, and development partners today.

---

## The Challenge

Using data from 20,871 Kenyan adults surveyed in 2024, build a machine learning model that predicts whether a person's financial situation has **Improved**, **Stayed the same**, or **Worsened**, and identify the key factors that drive financial outcomes in Kenya.

This is a multiclass classification problem. Your model will predict one of three outcomes for each individual:

- **Improved** — the person's financial situation is better than the previous year
- **Stayed the same** — no significant change in financial situation
- **Worsened** — the person's financial situation has deteriorated

---

## The Dataset

The dataset has been curated by iLabAfrica (Kevin Obote) from the 2024 FinAccess Household Survey — the most recent and comprehensive survey of financial inclusion in Kenya, published in December 2024. It covers 20,871 individuals across all 47 counties.

### Demographics

| Column | Description | Example Values |
|---|---|---|
| `county` | Kenya county of respondent | Nairobi City, Mombasa, Kisumu, Turkana |
| `location_type` | Urban or rural setting | Urban, Rural |
| `Sex` | Gender of respondent | Male, Female |
| `Age` | Age group | 18–25, 26–35, 36–45, 46–55, Above 55 |
| `household_size` | Number of people in household | 1 to 15+ |
| `education_level` | Highest education completed | None, Primary, Secondary, Tertiary |
| `marital_status` | Marital status | Married, Single, Widowed, Divorced |
| `has_disability` | Disability status | With Disability, Without Disability |

### Livelihood & Income

| Column | Description | Example Values |
|---|---|---|
| `monthly_income` | Total income received in past month (KES) | 1000, 5000, 30000 |
| `experienced_shock` | Experienced a financial shock in past year | Yes, No |

### Mobile & Digital Access

| Column | Description | Example Values |
|---|---|---|
| `mobile_ownership_1` | Owns a mobile phone | Yes, No |
| `mobile_money_access` | Has ever accessed mobile money | Yes, No |
| `barriers_mobile_money` | Main barrier to mobile money access | 0 (none), Phone ownership, Eligibility, Affordability |

### Financial Behaviour

| Column | Description | Example Values |
|---|---|---|
| `Savings_formal` | Uses formal savings (bank, SACCO, MFI) | Usage, Non-usage |
| `Savings_informal` | Uses informal savings (chama, group) | Usage, Non-usage |
| `Loan_formal` | Has a formal loan | Usage, Non-usage |
| `Loan_informal` | Has an informal loan | Usage, Non-usage |
| `defaulted` | Has defaulted on a loan or debt | Yes, No |
| `formal_service_use` | Uses any formal financial service | Usage, Non-usage |
| `barriers_bank` | Main barrier to bank access | Affordability, Eligibility, Awareness, Access, Trust |
| `prodsum1` | Number of financial services used (count) | 0 to 10+ |

### Financial Health & Literacy

| Column | Description | Example Values |
|---|---|---|
| `fl_score` | Financial literacy score | None correct, One correct, Two correct, All correct |
| `nfhi_11` | Was food secure in past 12 months | Yes, No |
| `nfhi_12` | Managed non-food spending adequately | Yes, No |
| `nfhi_13` | No debt stress in past 3 months | Yes, No |
| `accessto_13k_1month` | Can access KES 13,000 emergency funds in 1 month | Yes, No |
| `not_difficult` | Emergency funds not difficult to access in 30 days | Yes, No |

### Target Variable

| Column | Description | Class Distribution |
|---|---|---|
| `financial_status` | Financial situation vs. previous year | Worsened: 52.6% \| Stayed the same: 26.9% \| Improved: 20.5% |

---

## The Guiding Question

Beyond building a model, your team must answer this question:

> **Which factors most strongly predict financial deterioration among Kenyan adults, and what does your model suggest policymakers, banks, or NGOs should prioritise to improve financial wellbeing?**

---

## What You Are Expected To Do

| Step | Task |
|---|---|
| 1. Data Cleaning | Handle missing values, check for inconsistencies, understand each feature |
| 2. EDA | Visualise distributions, explore relationships between features and the target, generate insights |
| 3. Preprocessing | Encode categorical variables, scale numerical features, prepare data for modelling |
| 4. Modelling | Train at least one classification model that predicts `financial_status` |
| 5. Evaluation | Use weighted F1-score as your primary metric, report performance on a held-out test set |
| 6. Interpretation | Identify important features and translate your findings into actionable insights |

---

## Deliverables

| Deliverable | Requirements |
|---|---|
| Jupyter Notebook (`.ipynb`) | Full solution — data cleaning, EDA, preprocessing, model training, evaluation, and interpretation. Must be clean, well-commented, and reproducible. |
| Data Visualisations | Minimum 5 meaningful charts included in the notebook. Well-labelled and interpretable by a non-technical audience. |
| 7-Slide PowerPoint (`.pptx`) | Slides: (1) Problem, (2) Dataset & EDA, (3) Key EDA finding, (4) Modelling approach, (5) Model performance, (6) Key drivers — answer to guiding question, (7) Recommendations. |

---

## Evaluation Metric & Dataset Access

Your model is evaluated using the **Weighted F1-Score**, which accounts for class imbalance. A model that only predicts 'Worsened' will score poorly because it fails on 'Improved' and 'Stayed the same'.

- **Dataset (Kaggle — recommended):** https://www.kaggle.com/datasets/davidpbriggs/kenya-finaccess-household-survey-2024
- **Official FinAccess Report & Manual:** https://finaccess.knbs.or.ke/reports-and-datasets

---

## Note on Missing Values

| Column | Missing % | How to Handle |
|---|---|---|
| `barriers_bank` | 27.5% | These are respondents who have a bank — no barrier applies. Impute with a new category: `'No barrier'` or `0`. |
| `monthly_income` | ~0% (pre-imputed) | Already imputed with median in the curated dataset. |

---

## Contact

**Philip Tait** | Chairperson, Strathmore Data Community  
📧 datacommunity@strathmore.edu | 📞 0716 914 156 | 🔗 linktr.ee/strathmoredatacommunity

---

*Learn. Build. Launch.*

*The Strathmore Data Community is an officially recognised student community at Strathmore University, Nairobi.*
