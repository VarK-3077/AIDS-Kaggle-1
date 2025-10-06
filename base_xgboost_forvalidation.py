import xgboost as xgb

def get_model():
    """Returns a standard XGBoost classifier."""
    return xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42
    )