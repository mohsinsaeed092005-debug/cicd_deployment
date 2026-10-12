import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

BOSTON_COLS = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV",
]


def load_data(dataset: str):
    """Returns X, y and task type ('classification' or 'regression')."""
    if dataset == "disease":
        df = pd.read_csv(os.path.join(DATA_DIR, "disease.csv"), encoding="utf-8-sig")
        target, task = "target", "classification"
    elif dataset == "boston":
        df = pd.read_csv(
            os.path.join(DATA_DIR, "boston.csv"),
            sep=r"\s+",
            header=None,
            names=BOSTON_COLS,
        )
        target, task = "MEDV", "regression"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    df = df.fillna(df.median(numeric_only=True))
    X = df.drop(columns=[target])
    y = df[target]
    return X, y, task
