import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

FEATURE_COLS = [
    "num_functional_requirements", "num_integrations", "requires_web_component",
    "requires_mobile_component", "requires_ai_ml_component", "requires_cloud_infra",
    "requires_data_pipeline", "num_technologies_mentioned", "has_deadline",
    "deadline_days", "complexity_keyword_count", "requirement_text_length"
]
TARGET = "project_category"

df = pd.read_csv("synthetic_project_requests.csv")
X = df[FEATURE_COLS]
y = df[TARGET]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

mlflow.set_experiment("project_category_classification")

# --- Baseline: Logistic Regression ---
with mlflow.start_run(run_name="baseline_logreg") as base_run:
    # scaler baked INTO the pipeline so serving is self-contained -- no separate
    # scaler artifact to keep in sync with the model at inference time.
    baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    baseline.fit(Xtr, ytr)
    pred = baseline.predict(Xte)
    acc, f1 = accuracy_score(yte, pred), f1_score(yte, pred, average="macro")
    mlflow.log_param("model_type", "LogisticRegression (Pipeline: scaler+clf)")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("macro_f1", f1)
    mlflow.sklearn.log_model(baseline, "model", registered_model_name="project_category_classifier")
    print(f"[baseline] accuracy={acc:.4f} macro_f1={f1:.4f}")

    client = MlflowClient()
    latest = client.get_latest_versions("project_category_classifier")[0]
    client.set_registered_model_alias("project_category_classifier", "champion", latest.version)
    print(f"Aliased project_category_classifier v{latest.version} as @champion")
# --- Selected model: Random Forest ---
with mlflow.start_run(run_name="random_forest") as run:
    n_estimators, max_depth = 200, 12
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("macro_f1", f1)

    import re
    report = classification_report(yte, pred, output_dict=True)
    for cls, metrics in report.items():
        if isinstance(metrics, dict):
            safe_name = re.sub(r"[^A-Za-z0-9_.\- ]", "_", cls).replace(" ", "_")
            mlflow.log_metric(f"f1_{safe_name}", metrics["f1-score"])

    cm = confusion_matrix(yte, pred, labels=sorted(y.unique()))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix — project_category")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    mlflow.sklearn.log_model(clf, "model")
    print(f"[random_forest] accuracy={acc:.4f} macro_f1={f1:.4f}")
    print(f"Run ID: {run.info.run_id}")

print("\nDone. Run 'mlflow ui' and open http://127.0.0.1:5000 to inspect both runs.")

