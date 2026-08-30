import re
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.tracking import MlflowClient

FEATURE_COLS = [
    "num_functional_requirements", "num_integrations", "requires_web_component",
    "requires_mobile_component", "requires_ai_ml_component", "requires_cloud_infra",
    "requires_data_pipeline", "num_technologies_mentioned", "has_deadline",
    "deadline_days", "complexity_keyword_count", "requirement_text_length"
]
TARGET = "estimated_effort_person_days"

df = pd.read_csv("synthetic_project_requests.csv")
X = df[FEATURE_COLS]
y = df[TARGET]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("effort_regression")

def log_run(name, model, params):
    with mlflow.start_run(run_name=name) as run:
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        mae = mean_absolute_error(yte, pred)
        rmse = mean_squared_error(yte, pred) ** 0.5
        r2 = r2_score(yte, pred)
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        print(f"[{name}] MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.4f}")
        return run, model, r2

with mlflow.start_run(run_name="baseline_linear") as base_run:
    baseline_model = LinearRegression()
    baseline_model.fit(Xtr, ytr)
    pred = baseline_model.predict(Xte)
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mae", mean_absolute_error(yte, pred))
    mlflow.log_metric("rmse", mean_squared_error(yte, pred) ** 0.5)
    mlflow.log_metric("r2", r2_score(yte, pred))
    mlflow.sklearn.log_model(baseline_model, "model", registered_model_name="effort_regressor")
    print(f"[baseline_linear] MAE={mean_absolute_error(yte, pred):.2f} RMSE={mean_squared_error(yte, pred) ** 0.5:.2f} R2={r2_score(yte, pred):.4f}")

    client = MlflowClient()
    latest = client.get_latest_versions("effort_regressor")[0]
    client.set_registered_model_alias("effort_regressor", "champion", latest.version)
    print(f"Aliased effort_regressor v{latest.version} as @champion")
run, model, r2 = log_run(
    "random_forest",
    RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42),
    {"model_type": "RandomForestRegressor", "n_estimators": 200, "max_depth": 12},
)

print(f"\nRun ID: {run.info.run_id}")
print("Done. Run 'mlflow ui' and open http://127.0.0.1:5000 to inspect both runs.")
