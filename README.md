# 🏦 Loan Default Prediction System

> **Predicting loan default risk for Nigerian microfinance banks and fintechs — 
> powered by XGBoost and explained by SHAP**

## 🚀 Live Demo
**[👉 Open the Live App](https://rotech-loan-default-predictor.streamlit.app/)**

Enter a borrower's details and get an instant ML-powered credit risk 
assessment with a plain-English SHAP explanation — no code required.

![App Screenshot](reports/app_page.png)

---

## 🎯 The Business Problem

Every year, Nigerian microfinance banks and fintechs lose billions of 
naira to loan defaults, not because of bad judgement, but because 
manual credit assessment cannot process enough signals fast enough.

A loan officer reviewing 40 applications a day using a 2019 spreadsheet 
cannot simultaneously weigh bureau scores, debt burden, employment 
stability, and household income for every applicant.

This system does.

---

## 📊 Project Overview

| | |
|---|---|
| **Dataset** | Home Credit Default Risk (Kaggle) |
| **Records** | 307,511 loan applications |
| **Features** | 122 original → 101 engineered |
| **Target** | Binary — loan default (1) or repaid (0) |
| **Class imbalance** | 8.07% default rate |
| **Best model** | XGBoost |
| **AUC-ROC** | 0.7679 |
| **Recall** | 65.8% of defaults caught |

---

## 🔍 Key Findings

**1. Class imbalance is the core challenge**
Only 8.07% of borrowers defaulted. A naive model predicting "no default" 
for everyone achieves 91.9% accuracy, but catches zero bad loans. 
AUC-ROC and Recall are the only metrics that matter here.

**2. External credit bureau scores dominate**
EXT_SOURCE_MEAN (average of three bureau scores) is the single strongest 
predictor of default, confirming what credit officers have always known 
intuitively about the importance of credit history.

**3. Debt burden matters more than loan size**
ANNUITY_INCOME_RATIO (monthly repayment relative to income) is a stronger 
predictor than the raw loan amount. A borrower's financial stretch before 
the loan is disbursed predicts their ability to repay after.

**4. Longer loan terms carry higher risk**
CREDIT_TERM_YEARS ranks consistently in the top 5 SHAP features. 
Borrowers on extended repayment schedules default at higher rates 
independent of income or loan size.

---

## 🛠 Technical Stack

| Layer | Tools |
|-------|-------|
| Data processing | Python, pandas, NumPy |
| ML modelling | scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualisation | matplotlib, seaborn, Plotly |
| Deployment | Streamlit, Streamlit Cloud |
| Version control | Git, GitHub |

---

## 📁 Project Structure