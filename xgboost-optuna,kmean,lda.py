import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# --- 1. Load Training Data ---
print("Loading training data...")
try:
    df_train = pd.read_csv('train.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure train.csv is in the directory.")
    exit()

# --- 2. Prepare Data for the Model ---
if 'song_popularity' not in df_train.columns or 'id' not in df_train.columns:
    print("Error: 'train.csv' must contain 'id' and 'song_popularity' columns.")
    exit()

X_train = df_train.drop(['id', 'song_popularity'], axis=1)
y_train = df_train['song_popularity']

# --- 3. Handle Missing Values ---
# This is necessary for StandardScaler, LDA, and KMeans to work correctly.
print("Handling missing values by filling with 0...")
X_train.fillna(0, inplace=True)
print("Data prepared.")
print("-" * 30)

# --- 4. Feature Engineering with LDA and K-Means ---
print("Applying StandardScaler, LDA, and K-Means clustering...")
# Preprocessing is sensitive to feature scales, so we standardize the data first
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Apply LDA for dimensionality reduction. For binary classification, n_components can be at most 1.
lda = LinearDiscriminantAnalysis(n_components=1)
# LDA is a supervised method, so it requires both X and y for fitting
X_lda = lda.fit_transform(X_scaled, y_train)

# Define and fit the KMeans model on the LDA-transformed data.
# The number of clusters (k) is a hyperparameter you can tune.
# n_clusters = 5
# kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
# cluster_labels = kmeans.fit_predict(X_lda)

# Add the cluster labels as a new feature to the original (unscaled) training data
# X_train['cluster'] = cluster_labels
# print(f"Added {n_clusters} cluster labels from LDA->KMeans as a new feature ('cluster').")
# print("-" * 30)


# --- 5. Tune XGBoost Hyperparameters with Optuna ---
print("Starting hyperparameter tuning for XGBoost with the new features...")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    """Define the objective function for Optuna to optimize."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False,
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42
    }
    model = xgb.XGBClassifier(**params)
    # Use 5 splits for faster tuning
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # The X_train passed here now includes the 'cluster' feature
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Hyperparameter tuning complete.")
print(f"Best trial score (AUC): {study.best_value:.4f}")
print("Best hyperparameters found:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print("-" * 30)

# --- 6. Build and Evaluate the Final Model ---
print("Building the final XGBoost model with the best hyperparameters...")
best_params = study.best_params
final_model = xgb.XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='auc')

print("Evaluating the final model using 10-fold cross-validation...")
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Calculate final scores on the data with the cluster feature
roc_auc_scores = cross_val_score(final_model, X_train, y_train, cv=kfold, scoring='roc_auc')
accuracy_scores = cross_val_score(final_model, X_train, y_train, cv=kfold, scoring='accuracy')

print("-" * 30)
print("Final Model Evaluation Complete:")
print(f"Mean ROC AUC Score: {np.mean(roc_auc_scores):.4f} (Std: {np.std(roc_auc_scores):.4f})")
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f} (Std: {np.std(accuracy_scores):.4f})")
print("✅ Success! The model has been evaluated with LDA + K-Means features.")

