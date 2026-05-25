from sklearn.ensemble import IsolationForest

def detect_anomalies(data):

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    preds = model.fit_predict(data)

    return preds