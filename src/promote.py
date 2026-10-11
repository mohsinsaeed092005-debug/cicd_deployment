import argparse
import os
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="registered model name")
    parser.add_argument("--from-alias", required=True)
    parser.add_argument("--to-alias", required=True)
    args = parser.parse_args()

    client = MlflowClient()
    mv = client.get_model_version_by_alias(args.name, args.from_alias)
    client.set_registered_model_alias(args.name, args.to_alias, mv.version)
    print(f"{args.name}: version {mv.version} now has alias '{args.to_alias}'")


if __name__ == "__main__":
    main()
