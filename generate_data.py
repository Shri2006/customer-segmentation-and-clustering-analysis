"""
generate_data.py
Generates a realistic synthetic e-commerce customer dataset with five latent
behavioral/demographic segments (High-Value, Growth Potential, Loyal, At-Risk, New).
The segments are NOT hard-labeled in the output CSV -- clustering is expected to
rediscover them from the raw features, exactly like a real unsupervised task.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 1000

# Target segment proportions (mirrors a realistic customer base)
segment_props = {
    "High-Value": 0.15,
    "Growth Potential": 0.25,
    "Loyal": 0.30,
    "At-Risk": 0.20,
    "New": 0.10,
}

# Per-segment generative parameters: (mean, std) for each feature
# Features: Age, Annual_Income (k$), Spending_Score (1-100),
#           Purchase_Frequency (orders/year), Tenure_Months, Customer_Lifetime_Value ($)
params = {
    "High-Value":       dict(age=(42, 8),  income=(105, 15), spend=(85, 8),  freq=(28, 5),  tenure=(48, 12), clv=(5200, 700)),
    "Growth Potential": dict(age=(31, 6),  income=(68, 12),  spend=(72, 10), freq=(18, 4),  tenure=(20, 8),  clv=(3000, 500)),
    "Loyal":            dict(age=(46, 10), income=(55, 10),  spend=(55, 9),  freq=(22, 5),  tenure=(60, 15), clv=(2200, 400)),
    "At-Risk":          dict(age=(38, 9),  income=(60, 14),  spend=(28, 9),  freq=(6, 2),   tenure=(30, 14), clv=(1300, 350)),
    "New":              dict(age=(27, 5),  income=(50, 10),  spend=(40, 12), freq=(3, 1.5), tenure=(2, 1.2), clv=(700, 250)),
}

rows = []
for seg, prop in segment_props.items():
    n = int(round(N * prop))
    p = params[seg]
    for _ in range(n):
        rows.append({
            "Age": max(18, np.random.normal(*p["age"])),
            "Annual_Income_k": max(15, np.random.normal(*p["income"])),
            "Spending_Score": np.clip(np.random.normal(*p["spend"]), 1, 100),
            "Purchase_Frequency": max(0, np.random.normal(*p["freq"])),
            "Tenure_Months": max(0, np.random.normal(*p["tenure"])),
            "Customer_Lifetime_Value": max(50, np.random.normal(*p["clv"])),
            "_true_segment": seg,  # kept only for validation, dropped before modeling
        })

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.insert(0, "CustomerID", [f"CUST{i:05d}" for i in range(1, len(df) + 1)])

for col in ["Age", "Annual_Income_k", "Spending_Score", "Purchase_Frequency", "Tenure_Months", "Customer_Lifetime_Value"]:
    df[col] = df[col].round(2)

df.to_csv("/home/claude/customer_segmentation/data/customers_raw.csv", index=False)
print(f"Generated {len(df)} customer records")
print(df["_true_segment"].value_counts(normalize=True).round(3))
