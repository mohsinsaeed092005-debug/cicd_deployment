import argparse
import os
import shutil
import mlflow
import mlflow.pyfunc
from mlflow.artifacts import download_artifacts

from data import load_data

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["disease", "boston"])
    parser.add_argument("--model", required=True,
                        choices=["random_forest", "logistic_regression"])
    parser.add_argument("--alias", required=True)
    parser.add_argument("--env", required=True, choices=["dev", "preprod", "prod"])
    args = parser.parse_args()

    name = f"{args.dataset}_{args.model}"
    uri = f"models:/{name}@{args.alias}"
    target = os.path.join(os.path.expanduser("~"), "deployments", args.env, name)

    if os.path.exists(target):
        shutil.rmtree(target)
    os.makedirs(target)
    download_artifacts(artifact_uri=uri, dst_path=target)

    # smoke test: deployed model must be able to predict
    model = mlflow.pyfunc.load_model(uri)
    X, _, _ = load_data(args.dataset)
    preds = model.predict(X.head(5))
    print(f"DEPLOYED {name}@{args.alias} to {target}, sample predictions: {list(preds)}")


if __name__ == "__main__":
    main()
