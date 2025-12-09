# Salary Range Classification (Adult dataset) - From Scratch

## Project goal
Predict whether an individual earns > $50k/year using the 1994 U.S. Census (Adult) dataset.

## Repo structure
(see the earlier folder structure in this README)

## Quick start
1. Create virtual environment and install requirements:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. Download and preprocess:


3. Train models:


4. Run prediction UI:


streamlit run src/predict_streamlit.py

## What I implemented
- EDA: basic dataset overview, missing value counts, simple plots in `data/processed/plots`.
- Preprocessing: missing value removal, label conversion for salary, one-hot-like encoding of top categories.
- Models implemented from scratch:
- Logistic Regression (gradient descent)
- Gaussian Naive Bayes
- K-Nearest Neighbors
- Decision Tree (CART-style)
- Evaluation metrics implemented from scratch: accuracy, precision, recall, f1, confusion matrix, approximate ROC-AUC.
- Streamlit app for interactive predictions.

## Notes & improvements
- Current preprocessing drops rows with missing values for simplicity; you could implement smarter imputations.
- Categorical encoding uses top-K categories; you can expand or use target encoding.
- Decision Tree implementation is simple and may be slower for large feature spaces; a pruned or optimized version is recommended.
- For production use, consider cross-validation, feature selection, and hyperparameter tuning.

## How to push to GitHub
git init
git add .
git commit -m "Initial commit - salary classification from scratch"
gh repo create your-repo-name --public --source=. --remote=origin
git push -u origin main