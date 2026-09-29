"""
Generates a synthetic HR employee attrition dataset with a real (built-in)
relationship between features and attrition, similar in spirit to the
well-known IBM HR Analytics Attrition dataset.
"""
import numpy as np
import pandas as pd

np.random.seed(33)
N = 1200

age = np.random.randint(21, 60, N)
monthly_income = np.round(np.random.normal(55000, 18000, N), -2)
monthly_income = np.clip(monthly_income, 15000, 150000)
years_at_company = np.clip((age - 21) * np.random.uniform(0.1, 0.9, N), 0, 35).round(1)
job_satisfaction = np.random.randint(1, 5, N)  # 1-4 scale
work_life_balance = np.random.randint(1, 5, N)  # 1-4 scale
overtime = np.random.choice(["Yes", "No"], N, p=[0.3, 0.7])
distance_from_home = np.random.randint(1, 30, N)
department = np.random.choice(["Sales", "R&D", "HR", "IT Support"], N, p=[0.3, 0.35, 0.1, 0.25])
num_promotions = np.random.poisson(0.7, N)

# Build attrition probability with real signal
logit = (
    -2.0
    - 0.6 * (job_satisfaction - 2.5)
    - 0.5 * (work_life_balance - 2.5)
    + 0.9 * (overtime == "Yes").astype(int)
    + 0.03 * (distance_from_home - 10)
    - 0.05 * years_at_company
    - 0.3 * num_promotions
    - 0.00002 * (monthly_income - 55000)
)
prob_attrition = 1 / (1 + np.exp(-logit))
attrition = (np.random.rand(N) < prob_attrition).astype(int)

df = pd.DataFrame({
    "Employee_ID": [f"EMP-{2000+i}" for i in range(N)],
    "Age": age,
    "Department": department,
    "Monthly_Income": monthly_income,
    "Years_At_Company": years_at_company,
    "Job_Satisfaction": job_satisfaction,
    "Work_Life_Balance": work_life_balance,
    "OverTime": overtime,
    "Distance_From_Home_KM": distance_from_home,
    "Num_Promotions_Last_5Yrs": num_promotions,
    "Attrition": np.where(attrition == 1, "Yes", "No"),
})

# messiness
dupe_rows = df.sample(10, random_state=4)
df = pd.concat([df, dupe_rows], ignore_index=True)
null_idx = df.sample(18, random_state=5).index
df.loc[null_idx, "Work_Life_Balance"] = np.nan

df.to_csv("data/employee_attrition.csv", index=False)
print(f"Generated {len(df)} rows -> data/employee_attrition.csv")
print("Attrition rate:", round((df['Attrition'] == 'Yes').mean() * 100, 1), "%")
