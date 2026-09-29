# Employee Attrition Prediction (Classification)

Predicts which employees are likely to leave the company using a logistic
regression classifier, and identifies which factors drive that risk most —
so HR can act early instead of reacting after someone resigns.

## Problem
Given employee data (income, tenure, satisfaction, overtime, commute, etc.),
can we predict who is likely to leave, and which factors matter most?

## Dataset
`data/employee_attrition.csv` — 1,210 employee records with Age, Department,
Monthly Income, Years at Company, Job Satisfaction, Work-Life Balance,
OverTime, Commute Distance, Promotions, and Attrition (Yes/No). Modeled on
the structure of the well-known IBM HR Analytics Attrition dataset.
(Synthetic data, generated with `generate_data.py`, with a real underlying
relationship between features and attrition risk — 15.6% overall attrition rate.)

## Approach
1. **Clean** — removed duplicates, imputed missing Work-Life Balance scores with the median.
2. **EDA** — attrition rate by overtime status and job satisfaction.
3. **Preprocessing** — one-hot encoded Department, scaled features, 75/25 train/test split (stratified).
4. **Model** — Logistic Regression with class balancing (attrition is a minority class).
5. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve.
6. **Feature importance** — model coefficients ranked by impact on attrition.

## Key Findings
- **OverTime is the strongest positive driver of attrition.**
- **Job satisfaction and work-life balance are the strongest protective factors.**
- Longer commutes slightly raise attrition risk; more promotions and tenure lower it.
- The model reaches a ROC-AUC of ~0.80 on held-out data — a useful early-warning
  signal for prioritizing retention conversations, not a certainty for any one employee.

## Tools
Python · Pandas · Scikit-learn (Logistic Regression) · Matplotlib · Seaborn

## Files
- `employee_attrition_prediction.ipynb` — full analysis notebook
- `generate_data.py` — synthetic dataset generator
- `data/employee_attrition.csv` — the dataset
- `charts/` — exported chart images

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook employee_attrition_prediction.ipynb
```
