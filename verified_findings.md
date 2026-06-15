# Verified Findings for Presentation

This document contains confirmed, verified statistics generated directly from the trained CatBoost model and the preprocessed FinAccess 2024 dataset. **Do not alter these numbers.**

## 1. Dataset Overview
*   **Total Observations:** 20,848 individuals across 47 counties
*   **Target Variable Distribution (Financial Status):**
    *   Worsened: 10,962 (52.6%)
    *   Stayed the same: 5,607 (26.9%)
    *   Improved: 4,279 (20.5%)
*   **Key Population Stats:**
    *   Female: 59.1%
    *   Rural: 64.9% (Urban: 35.1%)
    *   Median Monthly Income: KES 5,000
    *   Has Mobile Money Access: 85.6%
    *   Has Formal Savings: 44.3%
    *   Experienced a Financial Shock: 43.5%
    *   Defaulted on a Loan: 33.3%

## 2. Model Performance
*   **Algorithm Chosen:** CatBoost Classifier + SMOTE-Tomek (for class balancing)
*   **Weighted F1-Score:** 0.5524
*   **Why this model?** Outperformed 10 other configurations (including XGBoost, LightGBM, and Ensembles). CatBoost is optimized for categorical survey data, and SMOTE-Tomek handled the severe class imbalance (52% Worsened vs 20% Improved).

## 3. Financial Vulnerability Index (FVI)
We transformed the model's probability of "Worsened" into a 0-100 Vulnerability Score for every individual.
*   **Score Range:** 1.3 (Safest) to 93.6 (Most Vulnerable)
*   **Mean Score:** 47.9
*   **High Risk (>70 Score):** 16.4% of the population (3,417 people)
*   **Low Risk (<30 Score):** 22.9% of the population (4,781 people)

## 4. County Vulnerability Rankings
Aggregating individual scores gives us the most and least vulnerable counties in Kenya.

**Top 5 MOST Vulnerable Counties:**
1.  **Tana River:** Avg Score = 72.0 (66.3% are High Risk, 70.8% worsened)
2.  **Turkana:** Avg Score = 69.9 (59.5% are High Risk, 69.2% worsened)
3.  **Kisumu:** Avg Score = 68.3 (52.7% are High Risk, 72.9% worsened)
4.  **Homabay:** Avg Score = 67.8 (52.2% are High Risk, 70.6% worsened)
5.  **Garissa:** Avg Score = 62.0 (33.8% are High Risk, 63.3% worsened)

**Top 5 LEAST Vulnerable Counties:**
1.  **Wajir:** Avg Score = 23.5 (0.2% High Risk, 28.5% worsened)
2.  **Bomet:** Avg Score = 26.3 (0.0% High Risk, 31.4% worsened)
3.  **Mandera:** Avg Score = 26.9 (0.0% High Risk, 34.3% worsened)
4.  **Kitui:** Avg Score = 34.1 (5.8% High Risk, 40.9% worsened)
5.  **West Pokot:** Avg Score = 36.7 (3.2% High Risk, 40.8% worsened)

## 5. Persona Profiles
What does a typical person look like in each category?

**The "Worsened" Persona:**
*   **Demographics:** 26-35 year-old Rural Female, Married/Living with partner
*   **Income:** KES 5,000 median
*   **Financial Health:** 49.3% experienced a shock, 36.8% defaulted on a loan, only 43.1% can access KES 13k in an emergency.
*   **Top Counties:** Kisumu, Meru, Homabay

**The "Improved" Persona:**
*   **Demographics:** 26-35 year-old Rural Female, Married/Living with partner
*   **Income:** KES 7,000 median (40% higher than Worsened)
*   **Financial Health:** Only 36.9% experienced a shock, lower default rate (31.1%), and significantly higher emergency fund access (66.3%).
*   **Top Counties:** Bomet, Nairobi City, Machakos

## 6. Policy Intervention Simulations (The Game Changer)
We used the ML model to simulate policy changes. "If we change X, what happens to vulnerability?"

**Intervention 1: Universal Shock Protection (Safety Nets/Insurance)**
*   **Action:** Provide guaranteed safety nets to the 9,073 people who experienced financial shocks.
*   **Result:** Average vulnerability drops from 56.8 to 48.6.
*   **Impact:** Removes **996 people** from the High-Risk zone. This is the **most effective intervention**.

**Intervention 2: Universal Emergency Fund Access**
*   **Action:** Ensure the 10,618 people who cannot access KES 13k in 30 days are given access (e.g., via specialized credit lines).
*   **Result:** Counter-intuitively, the model predicts vulnerability *increases slightly* (53.1 -> 53.4) and high-risk individuals increase. 
*   **Insight:** The model learned that simply having credit access without shock protection can lead to debt traps (defaulting), increasing long-term vulnerability.

## Answer to the Guiding Question
**"Which factors most strongly predict financial deterioration, and what should be prioritized?"**

Based on SHAP values and our Intervention Simulator, **exposure to economic shocks without adequate safety nets** is the primary driver of financial deterioration. While interventions like improving access to emergency credit seem logical, our simulations show they can exacerbate vulnerability if they lead to debt traps. Therefore, policymakers must prioritize **Shock Protection and Micro-Insurance products** over simple credit expansion to sustainably improve financial wellbeing in high-risk areas like Tana River and Turkana.
