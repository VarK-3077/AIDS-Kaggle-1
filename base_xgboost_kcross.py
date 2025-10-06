import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

# --- 1. Load Training Data ---
print("Loading training data...")
try:
    df_train = pd.read_csv('train.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure train.csv is in the directory.")
    exit()

# --- 2. Prepare Data for XGBoost ---
# Ensure 'song_popularity' and 'id' are handled correctly
if 'song_popularity' not in df_train.columns or 'id' not in df_train.columns:
    print("Error: 'train.csv' must contain 'id' and 'song_popularity' columns.")
    exit()
    
X_train = df_train.drop(['id', 'song_popularity'], axis=1)
y_train = df_train['song_popularity']

print("Data prepared.")
print("-" * 30)

# --- 3. Evaluate the Base Model with K-Fold Cross-Validation ---
print("Evaluating the base XGBoost model using 10-fold cross-validation...")

# Instantiate a base XGBoost model with default parameters for a baseline
# We set a random_state for reproducibility
base_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='auc')

# Define the cross-validation strategy
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Calculate cross-validated scores for ROC AUC and Accuracy
roc_auc_scores = cross_val_score(base_model, X_train, y_train, cv=kfold, scoring='roc_auc', n_jobs=-1)
accuracy_scores = cross_val_score(base_model, X_train, y_train, cv=kfold, scoring='accuracy', n_jobs=-1)

print("-" * 30)
print("Model Evaluation Complete:")
print(f"Mean ROC AUC Score: {np.mean(roc_auc_scores):.4f} (Std: {np.std(roc_auc_scores):.4f})")
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f} (Std: {np.std(accuracy_scores):.4f})")
print("✅ Success! Base model has been evaluated.")

