import argparse
import os
import sys
import mlflow
import mlflow.pyfunc
import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from data import load_data

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

THRESHOLDS = {"classification": 0.75, "regression": 0.60}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["disease", "boston"])
    parser.add_argument("--model", required=True,
                        choices=["random_forest", "logistic_regression"])
    parser.add_argument("--alias", default="Challenger")
    args = parser.parse_args()

    X, y, task = load_data(args.dataset)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    name = f"{args.dataset}_{args.model}"
    model = mlflow.pyfunc.load_model(f"models:/{name}@{args.alias}")
    preds = np.asarray(model.predict(X_test))

    if task == "classification":
        score = accuracy_score(y_test, preds)
        metric = "accuracy"
    else:
        score = r2_score(y_test, preds)
        metric = "r2"

    threshold = THRESHOLDS[task]
    print(f"{name}@{args.alias}: {metric}={score:.4f} (threshold {threshold})")
    if score < threshold:
        print("MODEL TEST FAILED")
        sys.exit(1)
    print("MODEL TEST PASSED")


if __name__ == "__main__":
    main()
