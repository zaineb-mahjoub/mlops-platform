import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load data — iris dataset, built into scikit-learn
X, y = load_iris(return_X_y=True)

# 2. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

max_iter = 200

with mlflow.start_run():
    # 3. Train
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)

    # 4. Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # 5. Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", max_iter)

    # 6. Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # 7. Log the model itself
    mlflow.sklearn.log_model(model, "model")

    print(f"Run complete. Accuracy: {accuracy:.4f}")
    
    