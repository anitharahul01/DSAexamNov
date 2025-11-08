import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Load patient data
patients = pd.read_csv(r"DSA Nov Retest Data\patients.csv")
patients["arrival_date"] = pd.to_datetime(patients["arrival_date"])
patients["Week"] = patients["arrival_date"].dt.isocalendar().week
patients["LengthOfStay"] = (
    pd.to_datetime(patients["departure_date"]) - patients["arrival_date"]
).dt.days

# Load service-level data
services = pd.read_csv(r"DSA Nov Retest Data\services_weekly.csv")

# Merge patient and service data
merged = pd.merge(
    patients,
    services,
    how="left",
    left_on=["Week", "service"],
    right_on=["week", "service"],
)

# BONUS TASK: ActualStaffPresent
schedule = pd.read_csv(r"DSA Nov Retest Data\staff_schedule_filtered.csv")
schedule_present = schedule[schedule["present"] == "Yes"]

# Group by week and service to count staff present
actual_staff = (
    schedule_present.groupby(["week", "service"])
    .size()
    .reset_index(name="ActualStaffPresent")
)

# Merge into main dataset
merged = pd.merge(
    merged,
    actual_staff,
    how="left",
    left_on=["Week", "service"],
    right_on=["week", "service"],
)
merged["ActualStaffPresent"] = merged["ActualStaffPresent"].fillna(0)

# Define features and target
features = [
    "LengthOfStay",
    "service",
    "available_beds",
    "patients_request",
    "patients_admitted",
    "patients_refused",
    "ActualStaffPresent",
]
target = "satisfaction"
X = merged[features]
y = merged[target]

# Preprocessing pipeline
categorical = ["service"]
numerical = [col for col in features if col not in categorical]

preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numerical),
    ]
)

pipeline = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

# Save model and preprocessor
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/best_model.pkl")
joblib.dump(preprocessor, "model/preprocessor1.pkl")
