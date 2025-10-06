import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def get_model(X_train):
    """
    Builds and returns the custom 'FrankenBoost' model.
    Requires the training data to calculate the feature ratio.
    """
    # k: Number of random features to consider at each tree level
    k_features_ratio = np.sqrt(X_train.shape[1]) / X_train.shape[1]
    
    # Define the custom XGBoost core
    xgb_core = xgb.XGBClassifier(
        gamma=0,
        colsample_bylevel=k_features_ratio,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42
    )
    
    # Create the "Random Forest Shell" using BaggingClassifier
    frankenboost_model = BaggingClassifier(
        estimator=xgb_core,
        n_estimators=50,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    
    return frankenboost_model

# --- 1. Load Training Data ---
print("Loading training data...")
try:
    df_train = pd.read_csv('train.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure train.csv is in the directory.")
    exit()

# --- 2. Handle Missing Values ---
# We will fill any NaN values with 0, consistent with previous models.
df_train.fillna(0, inplace=True)
print("Missing values handled by filling with 0.")

# --- 3. Prepare Data for the Model ---
if 'song_popularity' not in df_train.columns or 'id' not in df_train.columns:
    print("Error: 'train.csv' must contain 'id' and 'song_popularity' columns.")
    exit()
    
X_train = df_train.drop(['id', 'song_popularity'], axis=1)
y_train = df_train['song_popularity']

print("Data prepared.")
print("-" * 30)

# --- 4. Build the FrankenBoost Model ---
print("Building the custom FrankenBoost model...")
frankenboost_model = get_model(X_train)
print("Model built successfully.")
print("-" * 30)

# --- 5. Evaluate the Model with 10-Fold Cross-Validation ---
print("Evaluating the FrankenBoost model using 10-fold cross-validation...")

# Define the 10-fold cross-validation strategy for final evaluation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Calculate cross-validated scores for ROC AUC and Accuracy
roc_auc_scores = cross_val_score(frankenboost_model, X_train, y_train, cv=kfold, scoring='roc_auc')
accuracy_scores = cross_val_score(frankenboost_model, X_train, y_train, cv=kfold, scoring='accuracy')

print("-" * 30)
print("FrankenBoost Model Evaluation Complete:")
print(f"Mean ROC AUC Score: {np.mean(roc_auc_scores):.4f} (Std: {np.std(roc_auc_scores):.4f})")
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f} (Std: {np.std(accuracy_scores):.4f})")
print("✅ Success! The model has been evaluated.")
