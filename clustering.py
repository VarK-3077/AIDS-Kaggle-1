import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import optuna

# --- 1. Load Training and Test Data ---
print("Loading training and test data...")
try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure both train.csv and test.csv are in the directory.")
    exit()

test_ids = df_test['id']

# --- 2. Prepare Data ---
X_train = df_train.drop(['id', 'song_popularity'], axis=1)
y_train = df_train['song_popularity']
X_test = df_test.drop('id', axis=1)
X_test = X_test[X_train.columns]

print("Data prepared.")
print("-" * 30)

# --- 3. Feature Engineering via Unsupervised Clustering ---
print("Starting feature engineering with K-Means clustering...")

# Step 3.1: Preprocess data for clustering (Impute and Scale)
# We need a complete and scaled dataset for K-Means to work effectively
imputer = KNNImputer(n_neighbors=5)
scaler = StandardScaler()

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Step 3.2: Find the optimal number of clusters using the Elbow Method
print("Finding optimal number of clusters with the Elbow Method...")
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.xticks(K_range)
plt.grid(True)
plt.savefig('elbow_plot.png')
print("Elbow plot saved as 'elbow_plot.png'. Check the plot to confirm the best 'k'.")
# NOTE: From the plot, we manually select the 'elbow' point. Let's assume it's 4 for this run.
optimal_k = 4 # <--- YOU CAN ADJUST THIS VALUE AFTER INSPECTING elbow_plot.png
print(f"Selected optimal k = {optimal_k}")

# Step 3.3: Create the new cluster feature
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
X_train['cluster'] = kmeans.fit_predict(X_train_scaled)
X_test['cluster'] = kmeans.predict(X_test_scaled)

print("New 'cluster' feature created and added to the datasets.")
print("-" * 30)


# --- 4. Hyperparameter Tuning with Optuna on Enriched Data ---
print("Starting hyperparameter tuning with Optuna on the new data...")

def objective(trial):
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'use_label_encoder': False,
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
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best trial score: {study.best_value:.4f}")
print("Best hyperparameters found:", study.best_params)
print("-" * 30)

# --- 5. Train Final Model with Best Hyperparameters ---
print("Training the final XGBoost model with best hyperparameters...")
best_params = study.best_params
final_model = xgb.XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='auc')
final_model.fit(X_train, y_train)
print("Model training complete.")
print("-" * 30)

# --- 6. Find Optimal Threshold (Youden's J) ---
print("Finding optimal threshold with the tuned model...")
y_train_pred_proba = cross_val_predict(final_model, X_train, y_train, cv=5, method='predict_proba')[:, 1]
fpr, tpr, thresholds = roc_curve(y_train, y_train_pred_proba)
best_threshold = thresholds[np.argmax(tpr - fpr)]
print(f"Found best threshold: {best_threshold:.4f}")
print("-" * 30)

# --- 7. Make Predictions ---
print("Making final label predictions...")
test_probabilities = final_model.predict_proba(X_test)[:, 1]
predictions = (test_probabilities >= best_threshold).astype(int)
print("Final label predictions generated.")
print("-" * 30)

# --- 8. Create Submission File ---
print("Creating the submission file...")
submission_df = pd.DataFrame({'id': test_ids, 'song_popularity': predictions})
submission_df.to_csv('submission.csv', index=False)
print("✅ Success! The submission.csv file has been created.")

