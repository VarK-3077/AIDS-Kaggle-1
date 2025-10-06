import pandas as pd
# Use the GPU-accelerated RandomForestClassifier from cuML
from cuml.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
# Import roc_auc_score for the manual loop
from sklearn.metrics import roc_auc_score
import optuna

# --- 1. Load Training Data ---
print("Loading training data...")
try:
    df_train = pd.read_csv('train.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure train.csv is in the directory.")
    exit()

# --- Handle Missing Values ---
df_train.fillna(0, inplace=True)
print("Missing values handled by filling with 0.")

# --- 2. Prepare Data for the Model ---
if 'song_popularity' not in df_train.columns or 'id' not in df_train.columns:
    print("Error: 'train.csv' must contain 'id' and 'song_popularity' columns.")
    exit()
    
X_train = df_train.drop(['id', 'song_popularity'], axis=1)
y_train = df_train['song_popularity']

# --- Diagnostic: Check Class Distribution ---
print("Class distribution in the target variable:")
print(y_train.value_counts())
print("-" * 30)


# Ensure all feature columns are numeric for GPU processing
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.int32)


print("Data prepared.")
print("-" * 30)

# --- 3. Hyperparameter Tuning with Optuna ---
print("Starting hyperparameter tuning with Optuna for GPU Random Forest...")

def objective(trial):
    """Define the objective function for Optuna to optimize."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'random_state': 42      
    }
    
    model = RandomForestClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    # Manual cross-validation loop for robustness
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Check if the validation fold has more than one class
        if len(np.unique(y_val_fold)) < 2:
            # If not, skip this fold as ROC AUC is not defined
            continue
            
        model.fit(X_train_fold, y_train_fold)
        # cuML predict_proba returns a DataFrame, get the probability of the positive class
        y_pred_proba = model.predict_proba(X_val_fold)[1]
        scores.append(roc_auc_score(y_val_fold, y_pred_proba))

    # If all folds were skipped, the trial is invalid
    if not scores:
        raise optuna.TrialPruned()

    return np.mean(scores)


# Create a study object and optimize the objective function, aiming to maximize ROC AUC
study = optuna.create_study(direction='maximize')
try:
    study.optimize(objective, n_trials=50) # Increase n_trials for a more thorough search
    print(f"Best trial ROC AUC score (during tuning): {study.best_value:.4f}")
    print("Best hyperparameters found:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
except optuna.exceptions.TrialPruned as e:
    print(f"All trials were pruned. This might indicate a severe issue with the data or model configuration. Error: {e}")
    # Exit if no successful trials were completed
    exit()

print("-" * 30)

# --- 4. Evaluate the Final Model with Best Hyperparameters ---
# Check if a best_params attribute exists. It might not if all trials failed.
if not hasattr(study, 'best_params'):
     print("Could not find best parameters because all Optuna trials failed. Exiting.")
     exit()

print("Evaluating the tuned GPU Random Forest model using 10-fold cross-validation...")

# Instantiate the model with the best parameters found by Optuna
tuned_model = RandomForestClassifier(**study.best_params, random_state=42)

# Define the 10-fold cross-validation strategy for final evaluation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Calculate cross-validated scores for ROC AUC and Accuracy
roc_auc_scores = cross_val_score(tuned_model, X_train, y_train, cv=kfold, scoring='roc_auc')
accuracy_scores = cross_val_score(tuned_model, X_train, y_train, cv=kfold, scoring='accuracy')

print("-" * 30)
print("Tuned Model Evaluation Complete:")
print(f"Mean ROC AUC Score: {np.mean(roc_auc_scores):.4f} (Std: {np.std(roc_auc_scores):.4f})")
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f} (Std: {np.std(accuracy_scores):.4f})")
print("✅ Success! Tuned GPU Random Forest model has been evaluated.")

