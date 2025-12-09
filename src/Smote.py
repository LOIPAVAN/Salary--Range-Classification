"""
smote.py
Generates all graphs related to class balancing:
- Before SMOTE imbalance graph
- After SMOTE balanced graph
- Comparison graph (Before vs After)
"""

import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import os

# ---------------------------------------------------
# 1. LOAD PROCESSED DATA
# ---------------------------------------------------

# Path when this file is inside: project/src/Smote.py
train = pd.read_csv("../data/processed/train_processed.csv")

# Folder where graphs will be saved
SAVE_PATH = "../reports/"

# Create folder if missing
os.makedirs(SAVE_PATH, exist_ok=True)

# ---------------------------------------------------
# 2. ENCODE CATEGORICAL COLUMNS FOR SMOTE
# ---------------------------------------------------

df = train.copy()
le_dict = {}   # store encoders

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

# Split features and target
y = df['salary']
X = df.drop(columns=['salary'])

# ---------------------------------------------------
# 3. BEFORE SMOTE GRAPH
# ---------------------------------------------------
plt.figure(figsize=(6, 4))
train['salary'].value_counts().plot(kind='bar')
plt.title("Salary Distribution (Before SMOTE)")
plt.xlabel("Salary (0 = Low, 1 = High)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "salary_before_smote.png"))
plt.close()


# ---------------------------------------------------
# 4. APPLY SMOTE
# ---------------------------------------------------
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)


# ---------------------------------------------------
# 5. AFTER SMOTE GRAPH
# ---------------------------------------------------
plt.figure(figsize=(6, 4))
y_res.value_counts().plot(kind='bar')
plt.title("Salary Distribution (After SMOTE)")
plt.xlabel("Salary (0 = Low, 1 = High)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "salary_after_smote.png"))
plt.close()


# ---------------------------------------------------
# 6. COMPARISON GRAPH
# ---------------------------------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
train['salary'].value_counts().plot(kind='bar')
plt.title("Before SMOTE")

plt.subplot(1, 2, 2)
y_res.value_counts().plot(kind='bar')
plt.title("After SMOTE")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "salary_comparison_smote.png"))
plt.close()


# ---------------------------------------------------
# 7. CONFIRMATION MESSAGE
# ---------------------------------------------------
print("✔ All graphs saved in:", SAVE_PATH)
print(" - salary_before_smote.png")
print(" - salary_after_smote.png")
print(" - salary_comparison_smote.png")
