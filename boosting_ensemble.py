import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold

# --- 1. Load and Prepare Data ---
print("Loading and preparing data...")
try:
    df_train = pd.read_csv('train.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure train.csv is in the directory.")
    exit()

if 'song_popularity' not in df_train.columns or 'id' not in df_train.columns:
    print("Error: 'train.csv' must contain 'id' and 'song_popularity' columns.")
    exit()

X_train = df_train.drop(['id', 'song_popularity'], axis=1)
y_train = df_train['song_popularity']

# Fill missing values - a safe preprocessing step for all models
X_train.fillna(0, inplace=True)
print("Data prepared.")
print("=" * 60)

# Suppress Optuna's logging to keep the output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- 2. Tune and Evaluate XGBoost ---
print("--- 2. Tuning and Evaluating XGBoost ---")

def objective_xgb(trial):
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'use_label_encoder': False,
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42
    }
    model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    return score

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=50)
final_model_xgb = xgb.XGBClassifier(**study_xgb.best_params, random_state=42)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
roc_auc_scores_xgb = cross_val_score(final_model_xgb, X_train, y_train, cv=kfold, scoring='roc_auc')
print(f"XGBoost Best Trial AUC: {study_xgb.best_value:.4f}")
print(f"XGBoost Final 10-Fold ROC AUC: {np.mean(roc_auc_scores_xgb):.4f} (Std: {np.std(roc_auc_scores_xgb):.4f})")
print("=" * 60)

# --- 3. Tune and Evaluate LightGBM ---
print("--- 3. Tuning and Evaluating LightGBM ---")

def objective_lgb(trial):
    params = {
        'objective': 'binary', 'metric': 'auc', 'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    model = lgb.LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc').mean()
    return score

study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=50)
final_model_lgb = lgb.LGBMClassifier(**study_lgb.best_params, random_state=42)
roc_auc_scores_lgb = cross_val_score(final_model_lgb, X_train, y_train, cv=kfold, scoring='roc_auc')
print(f"LightGBM Best Trial AUC: {study_lgb.best_value:.4f}")
print(f"LightGBM Final 10-Fold ROC AUC: {np.mean(roc_auc_scores_lgb):.4f} (Std: {np.std(roc_auc_scores_lgb):.4f})")
print("=" * 60)

# --- 4. Tune and Evaluate CatBoost ---
print("--- 4. Tuning and Evaluating CatBoost ---")

def objective_cat(trial):
    params = {
        'objective': 'Logloss', 'eval_metric': 'AUC', 'random_seed': 42, 'verbose': 0,
        'iterations': trial.suggest_int('iterations', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 100.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
    }
    model = cat.CatBoostClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc').mean()
    return score

study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=50)
final_model_cat = cat.CatBoostClassifier(**study_cat.best_params, random_seed=42, verbose=0)
roc_auc_scores_cat = cross_val_score(final_model_cat, X_train, y_train, cv=kfold, scoring='roc_auc')
print(f"CatBoost Best Trial AUC: {study_cat.best_value:.4f}")
print(f"CatBoost Final 10-Fold ROC AUC: {np.mean(roc_auc_scores_cat):.4f} (Std: {np.std(roc_auc_scores_cat):.4f})")
print("=" * 60)

print("✅ Benchmark complete. You can now compare the final scores to select models for ensembling.")
