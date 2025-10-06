from sklearn.ensemble import RandomForestClassifier

def get_model():
    """Returns a standard Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
