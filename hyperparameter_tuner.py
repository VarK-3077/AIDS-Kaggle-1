import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
import optuna

# Suppress Optuna's logging to keep the output clean during validation
optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_hyperparameters(model_class, search_space, X_train, y_train, method='standard_auc', n_trials=50):
    """
    A modular function to tune hyperparameters for any given model.

    Args:
        model_class: The class of the model to be tuned (e.g., xgb.XGBClassifier).
        search_space (dict): The Optuna search space definition.
        X_train, y_train: The training data.
        method (str): 'standard_auc' or 'kaggle_simulated_auc'.
        n_trials (int): The number of Optuna trials to run.

    Returns:
        dict: The best hyperparameters found.
    """

    def objective(trial):
        # Generate parameters from the provided search space
        params = {name: trial.suggest_int(name, low, high) if isinstance(high, int) 
                  else trial.suggest_float(name, low, high) 
                  for name, (low, high) in search_space.items()}
        
        # Add fixed parameters required by the model
        params.update({
            'use_label_encoder': False,
            'eval_metric': 'auc',
            'random_state': 42
        })

        model = model_class(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        if method == 'standard_auc':
            # Method 1: Standard AUC on probabilities
            score = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
        
        elif method == 'kaggle_simulated_auc':
            # Method 2: Simulate the Kaggle evaluation process
            y_probs = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_train, y_probs)
            j_scores = tpr - fpr
            best_threshold = thresholds[np.argmax(j_scores)]
            y_pred_binary = (y_probs >= best_threshold).astype(int)
            score = roc_auc_score(y_train, y_pred_binary)
        else:
            raise ValueError("Method must be 'standard_auc' or 'kaggle_simulated_auc'")
            
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params
