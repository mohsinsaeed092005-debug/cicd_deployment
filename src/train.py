import argparse
import os
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import load_data

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))


def build_model(model_name, task):
    if model_name == "random_forest":
        if task == "classification":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        return RandomForestRegressor(n_estimators=100, random_state=42)
    if model_name == "logistic_regression":
        if task != "classification":
            raise ValueError("Logistic Regression works only with the disease dataset")
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="disease", choices=["disease", "boston"])
    parser.add_argument("--model", default="random_forest",
                        choices=["random_forest", "logistic_regression"])
    parser.add_argument("--alias", default="Challenger")
    args = parser.parse_args()

    X, y, task = load_data(args.dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model(args.model, task)
    registered_name = f"{args.dataset}_{args.model}"

    mlflow.set_experiment(f"{args.dataset}_experiment")
    with mlflow.start_run(run_name=registered_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mlflow.log_param("dataset", args.dataset)
        mlflow.log_param("model", args.model)

        if task == "classification":
            mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
            mlflow.log_metric("f1", f1_score(y_test, preds))
        else:
            mlflow.log_metric("r2", r2_score(y_test, preds))
            mlflow.log_metric("rmse", float(np.sqrt(mean_squared_error(y_test, preds))))

        mlflow.sklearn.log_model(model, name="model", registered_model_name=registered_name, skops_trusted_types=["sklearn.tree._tree.Tree"])

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{registered_name}'")
    latest = max(int(v.version) for v in versions)
    client.set_registered_model_alias(registered_name, args.alias, latest)
    print(f"Registered {registered_name} v{latest} with alias '{args.alias}'")


if __name__ == "__main__":
    main()
