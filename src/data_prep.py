"""
Load the UCI Adult dataset, perform EDA summary prints and save processed CSVs.
Run: python src/data_prep.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",++++++++++++++++++++++++++++++
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "salary"
]

def download_and_load():
    train_path = os.path.join(DATA_DIR, 'adult.data')
    test_path = os.path.join(DATA_DIR, 'adult.test')

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(train_path):
        print("Downloading train dataset...")
        pd.read_csv(TRAIN_URL, header=None).to_csv(train_path, index=False, header=False)
    if not os.path.exists(test_path):
        print("Downloading test dataset...")
        pd.read_csv(TEST_URL, header=None,skiprows=1).to_csv(test_path, index=False, header=False)

    df_train = pd.read_csv(train_path, names=COLUMN_NAMES, skipinitialspace=True)
    df_test = pd.read_csv(test_path, names=COLUMN_NAMES, skipinitialspace=True, comment='|')
    # test file has a header row and trailing '.' in salary, remove trailing dot
    df_test['salary'] = df_test['salary'].astype(str).str.replace('.', '', regex=False).str.strip()
    return df_train, df_test

def basic_eda(df):
    print("=== BASIC INFO ===")
    print(df.info())
    print("\n=== HEAD ===")
    print(df.head())
    print("\n=== Describe (numerical) ===")
    print(df.describe().T)
    print("\n=== Missing values count ===")
    print(df.isin(['?']).sum())
    print("\n=== Value counts for salary ===")
    print(df['salary'].value_counts())

def preprocess(df):
    # Replace '?' with NaN
    df = df.replace('?', np.nan)
    # Drop rows with missing target
    df = df.dropna(subset=['salary'])
    # For simplicity, drop rows with any missing values
    df = df.dropna().reset_index(drop=True)
    # Binarize salary: <=50K -> 0, >50K -> 1
    df['salary'] = df['salary'].apply(lambda x: 1 if '>50' in str(x) else 0)
    # Drop fnlwgt (not useful usually)
    df = df.drop(columns=['fnlwgt'])
    # Keep education_num (redundant with education)
    # We'll one-hot encode categorical variables later in training pipeline
    return df

def save_processed(df, filename):
    path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved processed data to {path}")

def plot_some(df):
    # simple plots saved to data/processed/plots
    pdir = os.path.join(PROCESSED_DIR, 'plots')
    os.makedirs(pdir, exist_ok=True)
    plt.figure(figsize=(6,4))
    sns.countplot(x='salary', data=df)
    plt.title('Salary distribution (0 <=50K, 1 >50K)')
    plt.savefig(os.path.join(pdir, 'salary_dist.png'))
    plt.close()

    plt.figure(figsize=(8,4))
    sns.boxplot(x='salary', y='age', data=df)
    plt.title('Age by Salary')
    plt.savefig(os.path.join(pdir, 'age_by_salary.png'))
    plt.close()

def main():
    df_train, df_test = download_and_load()
    print("=== TRAIN EDA ===")
    basic_eda(df_train)
    df_train_p = preprocess(df_train)
    df_test_p = preprocess(df_test)
    save_processed(df_train_p, 'train_processed.csv')
    save_processed(df_test_p, 'test_processed.csv')
    plot_some(df_train_p)
    print("Preprocessing complete. Processed files in data/processed/")

if __name__ == "__main__":
    main()
