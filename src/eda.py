import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# =========================================================
# Create folder
# =========================================================
def ensure_dirs():
    os.makedirs("./reports/eda", exist_ok=True)

# =========================================================
# Load Data
# =========================================================
def load_data():
    df = pd.read_csv('../data/adult.data')
    
    df.columns = [
        "age","workclass","fnlwgt","education","education_num","marital_status",
        "occupation","relationship","race","sex","capital_gain","capital_loss",
        "hours_per_week","native_country","income"
    ]
    return df

# =========================================================
# Basic Info
# =========================================================
def basic_info(df):
    print("\n===== BASIC INFO =====")
    print(df.info())
    print("\n===== NULL VALUES =====")
    print(df.isnull().sum())
    print("\n===== DUPLICATES =====")
    print(df.duplicated().sum())
    print("\n===== SAMPLE ROWS =====")
    print(df.head())

# =========================================================
# Clean Data
# =========================================================
def clean_data(df):
    df.replace(" ?", pd.NA, inplace=True)
    print("Missing before cleaning:")
    print(df.isnull().sum())
    df.dropna(inplace=True)
    print("Missing after cleaning:")
    print(df.isnull().sum())
    return df

# =========================================================
# Encode data for SMOTE
# =========================================================
def encode_for_smote(df):
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object":
            df_encoded[col] = le.fit_transform(df_encoded[col])
    return df_encoded

# =========================================================
# Apply SMOTE
# =========================================================
def apply_smote(df):
    df_encoded = encode_for_smote(df)

    X = df_encoded.drop("income", axis=1)
    y = df_encoded["income"]

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    df_balanced = pd.DataFrame(X_res, columns=X.columns)
    df_balanced["income"] = y_res

    print("SMOTE Applied: Balanced dataset created")
    return df_balanced

# =========================================================
# Plot Wrapper
# =========================================================
def save_plot(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(f"Plot error in {func.__name__}: {e}")
    return wrapper

# =========================================================
# Income distribution
# =========================================================
@save_plot
def income_distribution(df, balanced=False):
    label = "" if not balanced else "_balanced"
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x="income")
    plt.title(f"Income Distribution{label}")
    plt.savefig(f"./reports/eda/income_distribution{label}.png")
    plt.close()

# =========================================================
# Numerical Distributions
# =========================================================
@save_plot
def numerical_distributions(df, balanced=False):
    label = "" if not balanced else "_balanced"
    numeric_cols = ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]

    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f"{col} Distribution{label}")
        plt.savefig(f"./reports/eda/{col}_distribution{label}.png")
        plt.close()

# =========================================================
# Categorical Counts
# =========================================================
@save_plot
def categorical_counts(df, balanced=False):
    label = "" if not balanced else "_balanced"
    cat_cols = ["workclass","education","marital_status","occupation","relationship","race","sex","native_country"]

    for col in cat_cols:
        plt.figure(figsize=(10,4))
        df[col].value_counts().head(10).plot(kind="bar")
        plt.title(f"{col} Count{label}")
        plt.savefig(f"./reports/eda/{col}_count{label}.png")
        plt.close()

# =========================================================
# Correlation Heatmap
# =========================================================
@save_plot
def correlation_heatmap(df, balanced=False):
    label = "" if not balanced else "_balanced"
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
    plt.title(f"Correlation Heatmap{label}")
    plt.savefig(f"./reports/eda/correlation_heatmap{label}.png")
    plt.close()

# =========================================================
# Confusion Matrix Plot 
# =========================================================
@save_plot
def plot_confusion_matrix(df):
    print("Generating confusion matrix...")
    
    df_encoded = encode_for_smote(df)
    X = df_encoded.drop("income", axis=1)
    y = df_encoded["income"]

    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)
    y_pred = model.predict(X)

    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")

    plt.title("Confusion Matrix")
    plt.savefig("./reports/eda/confusion_matrix.png")
    plt.close()

# =========================================================
# Main
# =========================================================
def main():
    ensure_dirs()

    print("Loading dataset...")
    df = load_data()
    basic_info(df)

    df = clean_data(df)

    print("\nGenerating ORIGINAL (unbalanced) graphs...")
    income_distribution(df)
    numerical_distributions(df)
    categorical_counts(df)
    correlation_heatmap(df)
    plot_confusion_matrix(df)

    print("\nApplying SMOTE to balance...")
    df_balanced = apply_smote(df)

    print("\nGenerating BALANCED EDA graphs...")
    income_distribution(df_balanced, balanced=True)
    numerical_distributions(df_balanced, balanced=True)
    categorical_counts(df_balanced, balanced=True)
    correlation_heatmap(df_balanced, balanced=True)

    print("\n=== All graphs generated (balanced + unbalanced) ===")
    print("Check folder: reports/eda/")

if __name__ == "__main__":
    main()
