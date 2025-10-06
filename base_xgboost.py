import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve

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
# Features (X) are all columns except 'id' and the target 'song_popularity'
# Target (y) is the 'song_popularity' column
X_train = df_train.drop(['id', 'song_popularity'], axis=1)
y_train = df_train['song_popularity']

# The test set has the same feature columns as the training set (minus the target)
X_test = df_test.drop('id', axis=1)

# Align columns just in case they are in a different order
# This is a good practice to prevent errors
X_test = X_test[X_train.columns]

print("Data prepared for training.")
print("-" * 30)


# --- 3. Initialize and Train the Final Model ---
print("Training the final XGBoost model on the entire training dataset...")

# Initialize the XGBoost Classifier. We set eval_metric to 'auc' because that is the
# official competition metric. This tells the model to optimize its internal parameters
# to produce the best possible ranking of probabilities.
final_model = xgb.XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc'
)

# Train the model on ALL available training data
final_model.fit(X_train, y_train)

print("Model training complete.")
print("-" * 30)

# --- 4. Find Optimal Threshold using Youden's J Statistic ---
print("Finding optimal threshold based on AUC-related metric (Youden's J)...")

# Get cross-validated probability predictions for the training set.
y_train_pred_proba = cross_val_predict(final_model, X_train, y_train, cv=5, method='predict_proba')[:, 1]

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_train, y_train_pred_proba)

# Calculate Youden's J statistic for each threshold
j_scores = tpr - fpr
# Find the index of the best threshold
best_j_idx = np.argmax(j_scores)
best_threshold = thresholds[best_j_idx]

print(f"Found best threshold: {best_threshold:.4f} which provides the best balance between TPR and FPR.")
print("-" * 30)


# --- 5. Make Predictions on the Test Data ---
print("Making final label predictions (0 or 1) for submission...")

# Predict probabilities on the test set
test_probabilities = final_model.predict_proba(X_test)[:, 1]

# Apply the new optimal threshold to get the final labels
predictions = (test_probabilities >= best_threshold).astype(int)

print("Final label predictions generated using the optimal threshold.")
print("-" * 30)


# --- 6. Create the Submission File ---
print("Creating the submission file...")

# Create a new DataFrame for the submission
submission_df = pd.DataFrame({
    'id': test_ids,
    'song_popularity': predictions
})

# Save the DataFrame to a CSV file
submission_df.to_csv('submission3.csv', index=False)

print("✅ Success! The submission.csv file has been created.")
print("This file contains the final 0/1 labels required for submission.")

