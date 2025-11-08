import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


patients = pd.read_csv(r"patients.csv")
patients["arrival_date"] = pd.to_datetime(patients["arrival_date"])
patients["Week"] = patients["arrival_date"].dt.isocalendar().week
patients["LengthOfStay"] = (
    pd.to_datetime(patients["departure_date"])
    - pd.to_datetime(patients["arrival_date"])
).dt.days


services = pd.read_csv(r"services_weekly.csv")


merged = pd.merge(
    patients,
    services,
    how="inner",
    left_on=["Week", "service"],
    right_on=["week", "service"],
)


merged.drop(
    columns=["week", "month", "event", "arrival_date", "departure_date"], inplace=True
)


X = merged[
    [
        "LengthOfStay",
        "service",
        "available_beds",
        "patients_request",
        "patients_admitted",
        "patients_refused",
        "staff_morale",
    ]
]
y = merged["satisfaction"]

# Preprocessing pipeline
categorical = ["service"]
numerical = [
    "LengthOfStay",
    "available_beds",
    "patients_request",
    "patients_admitted",
    "patients_refused",
    "staff_morale",
]

preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(drop="first"), categorical),
        ("num", StandardScaler(), numerical),
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipelines for models
lr_pipeline = Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])

rf_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("model", RandomForestRegressor(random_state=42))]
)

# Train models
lr_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

# Evaluate models
y_pred_lr = lr_pipeline.predict(X_test)
y_pred_rf = rf_pipeline.predict(X_test)

print("Linear Regression:")
print("  MAE:", mean_absolute_error(y_test, y_pred_lr))
print("  R²:", r2_score(y_test, y_pred_lr))

print("Random Forest:")
print("  MAE:", mean_absolute_error(y_test, y_pred_rf))
print("  R²:", r2_score(y_test, y_pred_rf))

# Save best model and preprocessor
joblib.dump(rf_pipeline, "model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
