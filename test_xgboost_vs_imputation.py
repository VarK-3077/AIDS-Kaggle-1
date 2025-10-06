import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

# --- 1. Load and Prepare Data ---
print("Loading data...")
try:
    df_train = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Error: train.csv not found. Please make sure it's in the same directory.")
    exit()

# Separate features (X) from the target variable (y)
X = df_train.drop(['id', 'song_popularity'], axis=1)
y = df_train['song_popularity']

print("Data loaded successfully.")
print("-" * 30)


# --- 2. Define Cross-Validation Strategy ---
# We use StratifiedKFold to ensure the proportion of classes is the same in each fold.
# This is important for classification tasks.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# --- 3. Strategy A: Plain XGBoost (Baseline) ---
print("Evaluating Strategy A: Plain XGBoost...")

# Initialize the XGBoost Classifier
xgb_plain = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Perform cross-validation
# We use 'roc_auc' as our scoring metric, which is a robust measure for binary classification.
scores_plain = cross_val_score(xgb_plain, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

print(f"Strategy A - Plain XGBoost CV AUC Score: {np.mean(scores_plain):.4f} (+/- {np.std(scores_plain):.4f})")
print("-" * 30)


# --- 4. Strategy B: KNN Imputation + XGBoost ---
print("Evaluating Strategy B: KNN Imputer + XGBoost...")

# Create a pipeline that first imputes the data, then trains the model
# A pipeline bundles preprocessing and modeling steps into a single object.
pipeline = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('classifier', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

# Perform cross-validation on the entire pipeline
scores_pipeline = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

print(f"Strategy B - Imputer Pipeline CV AUC Score: {np.mean(scores_pipeline):.4f} (+/- {np.std(scores_pipeline):.4f})")
print("-" * 30)


# --- 5. Compare Results and Conclude ---
print("Comparing the two strategies...")

if np.mean(scores_pipeline) > np.mean(scores_plain):
    print("\n🏆 The winner is Strategy B (Imputation + XGBoost)!")
    print("It's better to impute the missing values before training the model.")
else:
    print("\n🏆 The winner is Strategy A (Plain XGBoost)!")
    print("XGBoost's built-in handling of missing values works best for this data.")

print("\nNext step: Train the winning model on all the training data and predict on the test set.")