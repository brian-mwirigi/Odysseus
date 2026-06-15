# FinAccess 2024: Advanced Predictive Modeling & Policy Simulation Report
**Team Odysseus — Strathmore Data Community DataSprint 2026**

---

## 1. Executive Summary
This report details the end-to-end machine learning architecture developed by Team Odysseus to predict financial deterioration among Kenyan adults. Moving beyond standard multiclass classification, we engineered a **Financial Vulnerability Index (FVI)** to quantify individual risk and built a **Counterfactual Policy Simulator** to measure the exact impact of proposed interventions. 

**The primary finding:** Exposure to economic shocks without adequate safety nets is the overwhelming driver of financial deterioration in Kenya. Our simulations prove that expanding mere access to credit or mobile money without providing shock protection (insurance/safety nets) paradoxically *increases* vulnerability by pushing populations into debt traps.

---

## 2. Dataset Architecture & Feature Engineering
The 2024 FinAccess Household Survey dataset (20,848 observations, 28 raw features) required extensive preprocessing to extract actionable signals from subjective survey responses. 

### 2.1 Handling Structural Missingness
As noted in the dataset manual, missing values in `barriers_bank` (27.5%) were not missing at random (MNAR); they represented individuals who already possessed bank accounts. We explicitly imputed these with `'No barrier'` prior to encoding to preserve this structural signal.

### 2.2 Engineered Features (Domain Knowledge Injection)
To help the model understand complex socioeconomic interactions, we engineered several composite indicators:
*   **`nfhi_composite`**: A continuous scale of basic needs fulfillment, aggregating `nfhi_11` (food security), `nfhi_12` (non-food spending), and `nfhi_13` (debt stress).
*   **`resilience_score`**: A 0-3 index combining `accessto_13k_1month`, `not_difficult`, and `(1 - defaulted)`. This measures liquidity vs. liability.
*   **`shock_vulnerable`**: A binary interaction feature triggering `True` only if an individual experienced an economic shock *and* has a `resilience_score` $\leq 1$. This separates those who face shocks from those who are *destroyed* by them.
*   **`total_formal_products` / `total_informal_products`**: Count vectors to measure the depth of financial inclusion, rather than binary access.
*   **Target Encoding**: Replaced `county` strings with the historical `worsened_rate` of that specific county, allowing the tree models to split on geographic risk rather than alphabetical strings.

### 2.3 Class Imbalance Resolution (SMOTE + Tomek Links)
The target variable `financial_status` was severely imbalanced: Worsened (52.6%), Stayed the same (26.9%), Improved (20.5%).
Standard SMOTE creates synthetic points by interpolating between minority examples, which often introduces noise in overlapping class boundaries (common in survey data). We applied **SMOTE-Tomek**, which first synthesizes minority samples and then removes Tomek links (pairs of nearest neighbors belonging to different classes). This cleans the decision boundary, leading to cleaner splits for the tree models.

---

## 3. Modeling Strategy & Evaluation
We trained and evaluated 11 discrete model architectures, including Random Forests, LightGBM, XGBoost, and ensemble methods (Voting/Stacking). 

### 3.1 The Winning Algorithm: CatBoost Classifier
**CatBoost+ST** (CatBoost with SMOTE-Tomek data) achieved the highest **Weighted F1-Score of 0.5524**.
*   *Why CatBoost?* Unlike XGBoost or LightGBM, CatBoost uses oblivious decision trees (where the same splitting criterion is used across an entire level of the tree). This serves as a powerful regularizer, making it exceptionally resistant to overfitting on the noisy, highly categorical data typical of household surveys.

### 3.2 Evaluation Metrics
*   **Global Weighted F1:** 0.5524
*   **Class 'Worsened' (The critical class):** Precision: 0.64 | Recall: 0.78 | F1-Score: 0.70
*   The model prioritizes high recall (78%) on the 'Worsened' class, ensuring that the vast majority of vulnerable individuals are correctly identified by the algorithm, minimizing false negatives in risk assessment.

---

## 4. The Financial Vulnerability Index (FVI)
Instead of returning a hard classification label, we extracted the raw probability outputs from the CatBoost model. We define the **FVI** as the scaled probability ($P \times 100$) that an individual belongs to the 'Worsened' class.

*   **FVI Range:** 1.3 (Highly secure) to 93.6 (Critical danger)
*   **Mean National Score:** 47.9
*   **High-Risk Threshold (>70):** 16.4% of the Kenyan population (3,417 individuals in the sample) are currently living in the high-risk zone. 

### 4.1 County-Level Vulnerability Matrix
By aggregating the FVI, we can direct policy intervention to specific geographies.

**The Crisis Zone (Top 5 Most Vulnerable):**
1.  **Tana River:** Avg FVI = 72.0 | 66.3% of population in High-Risk zone.
2.  **Turkana:** Avg FVI = 69.9 | 59.5% in High-Risk zone.
3.  **Kisumu:** Avg FVI = 68.3 | 52.7% in High-Risk zone.
4.  **Homabay:** Avg FVI = 67.8 | 52.2% in High-Risk zone.
5.  **Garissa:** Avg FVI = 62.0 | 33.8% in High-Risk zone.

*Geographic Insight:* The most vulnerable counties are predominantly arid/semi-arid lands (ASAL) or regions heavily dependent on climate-sensitive agriculture/fishing (Lake Victoria basin), directly linking environmental shocks to financial deterioration.

---

## 5. SHAP Interpretability: What Drives Deterioration?
We utilized SHapley Additive exPlanations (SHAP) to deconstruct the CatBoost model's decision-making process. 

**Top Drivers of the "Worsened" Classification:**
1.  **`marital_status` (Married/Living with partner):** Counterintuitively, being married strongly pushes the model toward predicting 'Worsened', while being Single pushes toward 'Improved' or 'Stayed the same'. This indicates that the burden of household dependents during an economic crisis severely outweighs the benefits of dual-income structures in the current Kenyan economy.
2.  **`Age` (Older populations):** Older age brackets show a strong positive SHAP correlation with deterioration, highlighting the lack of pension/retirement safety nets.
3.  **`county_worsened_rate`:** Geographic location remains a massive structural barrier.
4.  **`barriers_bank_Affordability`:** High transaction costs and minimum balances actively trap populations in the worsening cycle.

---

## 6. Counterfactual Policy Simulator
To answer the Guiding Question, we built a simulator that alters specific features in the dataset (mimicking a policy intervention) and re-runs the FVI scoring to observe the delta.

### 6.1 Intervention A: Universal Shock Protection (The Winner)
*   *Mechanism:* For all 9,073 individuals who experienced a shock, we set `experienced_shock = 0` and `shock_vulnerable = 0`, simulating a perfect insurance/safety net payout.
*   *Result:* National average FVI dropped from 56.8 to 48.6. 
*   *Impact:* **996 individuals were rescued from the High-Risk zone.** This is the single most effective intervention possible.

### 6.2 Intervention B: Emergency Credit Access (The Paradox)
*   *Mechanism:* For the 10,618 people lacking emergency funds, we simulated providing access to KES 13k within 30 days.
*   *Result:* The High-Risk population actually **increased** from 2,461 to 2,578.
*   *Impact:* Providing liquidity without shock protection leads directly to defaulting. The model learned that populations taking emergency loans without baseline resilience end up worse off than before. 

---

## 7. Final Policy Recommendations (Answer to the Guiding Question)
Based on predictive feature importance and simulated counterfactuals, we advise the following priority shifts for Kenyan policymakers and NGOs:

1.  **Pivot from "Access" to "Protection":** The era of pushing basic mobile money and bank accounts is over (85.6% already have mobile money). The new frontier is micro-insurance and climate-indexed safety nets. Shock protection is mathematically proven by our model to be the most effective intervention.
2.  **Targeted ASAL and Lake Basin Interventions:** Blanket national policies are inefficient. Relief funds and subsidized insurance must be aggressively routed to Tana River, Turkana, Kisumu, and Homabay, which currently exhibit FVI scores indicating localized economic collapse.
3.  **Caution on Unsecured Emergency Credit:** Expanding emergency loan facilities (like Hustler Fund or digital lenders) without corresponding financial resilience training will actively increase vulnerability by triggering debt defaults among the poorest quartiles.
