import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import time
import importlib
import os

# --- 1. Define Models to Validate ---
# The names of the python files containing the models.
# The script will automatically find and import them.
model_files = [
    # 'simple_xgboost_model',
    # 'random_forest_model',
    'frankenboost'
]
print(f"Found {len(model_files)} model files to validate.")
print("-" * 30)


# --- 2. Load and Prepare Data ---
print("Loading data for validation...")
try:
    df_train = pd.read_csv('train.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure train.csv is in the directory.")
    exit()

# Handle non-numeric columns if any (like 'key') by label encoding
for col in df_train.columns:
    if df_train[col].dtype == 'object':
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col])

X_train = df_train.drop(['id', 'song_popularity'], axis=1)
y_train = df_train['song_popularity']
print("Data loaded and prepared.")
print("-" * 30)


# --- 3. Run Cross-Validation for Each Model ---
print("Starting 5-fold cross-validation for each model...")
k_folds = 5
cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
results_to_write = []

for model_file in model_files:
    start_time = time.time()
    print(f"Validating '{model_file}'...")
    
    try:
        # Dynamically import the get_model function from the current file
        module = importlib.import_module(model_file)
        get_model_func = getattr(module, 'get_model')
        
        # Get the model instance
        # Special handling for frankenboost which needs X_train
        if model_file == 'frankenboost_model':
            model = get_model_func(X_train)
        else:
            model = get_model_func()
            
        # Perform cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        mean_auc = np.mean(scores)
        
        # Format the result string
        result_line = f"{model_file} : {mean_auc:.5f}"
        results_to_write.append(result_line)
        
        end_time = time.time()
        print(f"'{model_file}' | Mean AUC: {mean_auc:.5f} | Time: {end_time - start_time:.2f}s")

    except Exception as e:
        print(f"Could not validate '{model_file}'. Error: {e}")

print("-" * 30)

# --- 4. Write Results to File ---
output_filename = 'results.txt'
print(f"Writing results to {output_filename}...")
try:
    with open(output_filename, 'w') as f:
        for line in results_to_write:
            f.write(line + '\n')
    print(f"✅ Validation complete. Results saved to {output_filename}.")
except IOError as e:
    print(f"Error writing to file: {e}")
