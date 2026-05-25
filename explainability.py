import shap
import joblib
import pandas as pd

model = joblib.load("models/xgboost.pkl")

X = pd.read_csv("data/processed/X_sample.csv")

explainer = shap.Explainer(model)
shap_values = explainer(X)

shap.summary_plot(shap_values, X)