# SBA Loan Default Prediction

**Course:** Applied Machine Learning (BUAN 6341) · UT Dallas · Spring 2026  
**Tools:** Python · Scikit-learn · H2O · Pandas · NumPy

---

## Business Problem
The U.S. Small Business Administration (SBA) guarantees loans to small businesses. 
Predicting which loans will default helps lenders make smarter decisions and reduces 
financial risk. This project builds an end-to-end binary classification pipeline on 
800K+ real SBA loan records.

## What I Built
- Cleaned and prepared 800K+ mixed-type loan records
- Engineered 10+ custom features including loan-to-SBA ratios, job creation metrics, 
  urban/franchise indicators, and WoE encodings
- Built leakage-safe preprocessing pipeline (fit on train only)
- Trained and tuned Logistic Regression (sklearn) and GLM + GBM (H2O)
- Optimized classification threshold using F1 score for imbalanced classes (82/18 split)

## Results
| Model | AUC | AUCPR |
|-------|-----|-------|
| Logistic Regression | ~0.79 | ~0.47 |
| H2O GLM | ~0.80 | ~0.48 |
| **H2O GBM (Final)** | **0.82** | **0.53** |

## Key Features Engineered
- `loan_to_sba_ratio` — SBA guarantee as proportion of total loan
- `disbursement_to_approval` — how much of approved amount was disbursed
- `jobs_created_ratio` — jobs created relative to existing employees
- `log_approval` — log-transformed approval amount
- `is_franchise`, `is_urban` — binary business context flags

## Files
- `Project-1-HimaVarsha-Komanduri-dal850892.ipynb` — full analysis notebook
- `scoring_template.py` — reproducible scoring function for unseen data
- `artifacts/` — saved models and encoders
