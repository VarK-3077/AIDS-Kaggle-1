import os
# Forcefully hide the GPU from PyTorch and TabPFN
# This MUST be done before importing torch or tabpfen
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
import torch # Now this will correctly report no CUDA
from tabpfn import TabPFNClassifier

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
# TabPFN can handle NaNs, but we'll fill them for consistency with other experiments.
print("Handling missing values by filling with 0...")
X_train.fillna(0, inplace=True)

print("Data prepared.")
print("-" * 30)

# --- 4. Check Feature Count for TabPFN ---
print("Checking feature count...")
num_features = X_train.shape[1]
print(f"Dataset has {num_features} features.")

if num_features > 100:
    print("\nError: TabPFN requires the number of features to be 100 or less.")
    print("The current dataset has too many features to run without PCA or feature selection.")
    exit() # Exit the script if the condition is not met

print("Feature count is within TabPFN's limit.")
print("-" * 30)


# --- 5. Build the TabPFN Model ---
# We now explicitly check for CUDA availability before creating the model
# to prevent errors during the cross-validation step.
print("Building the TabPFN model...")

if torch.cuda.is_available():
    device = 'cuda'
    print("TabPFN model will run on GPU.")
else:
    device = 'cpu'
    print("CUDA not available or PyTorch not compiled with CUDA. TabPFN model will run on CPU (this may be slow).")

# Create the model with the determined device setting and the required flag
# for large datasets.
model = TabPFNClassifier(device=device, ignore_pretraining_limits=True)

print("Model built successfully.")
print("-" * 30)

# --- 6. Evaluate the Model with 10-Fold Cross-Validation ---
print("Evaluating the TabPFN model using 10-fold cross-validation...")

# Define the 10-fold cross-validation strategy
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Calculate cross-validated scores for ROC AUC and Accuracy
# We use a try-except block here as TabPFN can sometimes be memory intensive
try:
    # Use the original features (X_train) instead of PCA-transformed features
    roc_auc_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc')
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    print("-" * 30)
    print("TabPFN Model Evaluation Complete:")
    print(f"Mean ROC AUC Score: {np.mean(roc_auc_scores):.4f} (Std: {np.std(roc_auc_scores):.4f})")
    print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f} (Std: {np.std(accuracy_scores):.4f})")
    print("✅ Success! The model has been evaluated.")

except Exception as e:
    print(f"\nAn error occurred during cross-validation: {e}")
    print("This can sometimes happen with TabPFN due to memory constraints.")
    print("Consider using a machine with more VRAM/RAM.")

