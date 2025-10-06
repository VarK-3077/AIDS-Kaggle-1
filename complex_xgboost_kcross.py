import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score
import optuna

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

# --- 3. Hyperparameter Tuning with Optuna ---
print("Starting hyperparameter tuning with Optuna...")

def objective(trial):
    """Define the objective function for Optuna to optimize."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'random_state': 42
    }

    model = xgb.XGBClassifier(**params)
    
    # Use cross-validation to get a robust score for hyperparameter tuning
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Using roc_auc as the optimization metric
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    
    return score

# Create a study object and optimize the objective function.
# We aim to maximize the AUC score.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50) # Increase n_trials for a more thorough search

print(f"Best trial ROC AUC score (during tuning): {study.best_value:.4f}")
print("Best hyperparameters found:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print("-" * 30)

# --- 4. Evaluate the Final Model with K-Fold Cross-Validation ---
print("Evaluating the final model with the best hyperparameters using 10-fold cross-validation...")

# Instantiate the model with the best parameters found by Optuna
best_params = study.best_params
final_model = xgb.XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='auc')

# Define the cross-validation strategy
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Calculate cross-validated scores for ROC AUC and Accuracy
roc_auc_scores = cross_val_score(final_model, X_train, y_train, cv=kfold, scoring='roc_auc', n_jobs=-1)
accuracy_scores = cross_val_score(final_model, X_train, y_train, cv=kfold, scoring='accuracy', n_jobs=-1)

print("-" * 30)
print("Model Evaluation Complete:")
print(f"Mean ROC AUC Score: {np.mean(roc_auc_scores):.4f} (Std: {np.std(roc_auc_scores):.4f})")
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f} (Std: {np.std(accuracy_scores):.4f})")
print("✅ Success! Model has been tuned and evaluated.")
