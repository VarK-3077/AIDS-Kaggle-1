import xgboost as xgb
from hyperparameter_tuner import tune_hyperparameters

def get_model(X_train, y_train, method='standard_auc'):
    """
    Builds a tuned XGBoost model by calling the central tuner.
    """
    print(f"Tuning XGBoost using the '{method}' method...")

    # Define the specific search space for a standard XGBoost model
    search_space = {
        'n_estimators': (100, 1000),
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 12),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0),
        'gamma': (0, 5.0)
    }

    # Call the modular tuner to get the best parameters
    best_params = tune_hyperparameters(
        model_class=xgb.XGBClassifier,
        search_space=search_space,
        X_train=X_train,
        y_train=y_train,
        method=method,
        n_trials=50 
    )
    print(f"Best XGBoost params found: {best_params}")

    # Build the final model with the best parameters
    tuned_xgb_model = xgb.XBClassifier(
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42,
        **best_params
    )
    
    return tuned_xgb_model

