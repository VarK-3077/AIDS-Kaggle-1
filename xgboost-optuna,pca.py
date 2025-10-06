import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import optuna

# We can suppress Optuna's logging to keep the output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_model(X_train, y_train):
    """
    Builds and returns an XGBoost model tuned using Optuna.
    """
    
    # --- 1. Define the Objective Function for Optuna ---
    # This function takes a 'trial' and returns a score for Optuna to maximize.
    def objective(trial):
        # Define the search space for the hyperparameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'random_state': 42
        }
        
        # Instantiate the XGBoost model with the suggested params
        model = xgb.XGBClassifier(**params)
        
        # Use cross-validation to get a robust score for this set of params
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # Use 3 splits for faster tuning
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
        
        return score

    # --- 2. Run the Hyperparameter Search ---
    print("Starting hyperparameter tuning for the XGBoost model...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50) # Run 50 trials to find good parameters
    
    print("Hyperparameter tuning complete.")
    best_params = study.best_params
    
    # --- 3. Build the Final Tuned XGBoost Model ---
    tuned_xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42,
        **best_params # Add the tuned parameters here
    )
    
    return tuned_xgb_model

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
# This is required because StandardScaler and PCA cannot handle NaNs.
print("Handling missing values by filling with 0...")
X_train.fillna(0, inplace=True)

print("Data prepared.")
print("-" * 30)

# --- 4. Scale Data and Apply PCA ---
print("Applying StandardScaler and PCA...")
# First, scale the data as PCA is sensitive to feature scales
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Initialize PCA to retain 95% of the variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA selected {pca.n_components_} components from the original {X_train.shape[1]} features.")
print("-" * 30)

# --- 5. Build the XGBoost Model ---
print("Building the custom XGBoost model (with Optuna tuning) on PCA features...")
# The get_model function now runs the tuning process on the PCA-transformed data
tuned_model = get_model(X_pca, y_train)
print("Model built successfully.")
print("-" * 30)

# --- 6. Evaluate the Model with 10-Fold Cross-Validation ---
print("Evaluating the final tuned XGBoost model using 10-fold cross-validation...")

# Define the 10-fold cross-validation strategy for final evaluation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Calculate cross-validated scores for ROC AUC and Accuracy
roc_auc_scores = cross_val_score(tuned_model, X_pca, y_train, cv=kfold, scoring='roc_auc')
accuracy_scores = cross_val_score(tuned_model, X_pca, y_train, cv=kfold, scoring='accuracy')

print("-" * 30)
print("Tuned XGBoost Model Evaluation Complete:")
print(f"Mean ROC AUC Score: {np.mean(roc_auc_scores):.4f} (Std: {np.std(roc_auc_scores):.4f})")
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f} (Std: {np.std(accuracy_scores):.4f})")
print("✅ Success! The model has been evaluated.")

