import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
import optuna

# --- 1. Load Training and Test Data ---
print("Loading training and test data...")
try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure both train.csv and test.csv are in the directory.")
    exit()

# Store test IDs for the submission file
test_ids = df_test['id']

# --- 2. Prepare Data for XGBoost ---
X_train = df_train.drop(['id', 'song_popularity'], axis=1)
y_train = df_train['song_popularity']
X_test = df_test.drop('id', axis=1)
X_test = X_test[X_train.columns]

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
    
    # Use cross-validation to get a robust score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    
    return score

# Create a study object and optimize the objective function.
# We aim to maximize the AUC score.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50) # You can increase n_trials for a more thorough search

print(f"Best trial score: {study.best_value:.4f}")
print("Best hyperparameters found:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print("-" * 30)


# --- 4. Train the Final Model with Best Hyperparameters ---
print("Training the final XGBoost model with the best hyperparameters...")
best_params = study.best_params
final_model = xgb.XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='auc')
final_model.fit(X_train, y_train)
print("Model training complete.")
print("-" * 30)

# --- 5. Find Optimal Threshold (Youden's J) with the Tuned Model ---
print("Finding optimal threshold with the tuned model...")
y_train_pred_proba = cross_val_predict(final_model, X_train, y_train, cv=5, method='predict_proba')[:, 1]
fpr, tpr, thresholds = roc_curve(y_train, y_train_pred_proba)
j_scores = tpr - fpr
best_threshold = thresholds[np.argmax(j_scores)]
print(f"Found best threshold: {best_threshold:.4f}")
print("-" * 30)

# --- 6. Make Predictions on the Test Data ---
print("Making final label predictions...")
test_probabilities = final_model.predict_proba(X_test)[:, 1]
predictions = (test_probabilities >= best_threshold).astype(int)
print("Final label predictions generated.")
print("-" * 30)

# --- 7. Create the Submission File ---
print("Creating the submission file...")
submission_df = pd.DataFrame({'id': test_ids, 'song_popularity': predictions})
submission_df.to_csv('submission4.csv', index=False)
print("✅ Success! The submission.csv file has been created.")

