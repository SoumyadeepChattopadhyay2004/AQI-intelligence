import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

df = pd.read_csv("data/processed/final_data.csv")

X = df.drop("AQI", axis=1)
y = df["AQI"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6
)

xgb.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)

ensemble_pred = (rf_pred + xgb_pred) / 2

rmse = mean_squared_error(
    y_test,
    ensemble_pred,
    squared=False
)

print("RMSE:", rmse)

joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(xgb, "models/xgboost.pkl")