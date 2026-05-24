import argparse
import json
import os
import tarfile

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifact", type=str, default="/opt/ml/processing/model/model.tar.gz")
    parser.add_argument("--test-data", type=str, default="/opt/ml/processing/test")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/evaluation")
    parser.add_argument("--target-column", type=str, default="sales")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_model(model_artifact_path):
    extract_dir = "/tmp/model"
    os.makedirs(extract_dir, exist_ok=True)

    if os.path.isdir(model_artifact_path):
        model_artifact_path = os.path.join(model_artifact_path, "model.tar.gz")

    with tarfile.open(model_artifact_path) as tar:
        tar.extractall(extract_dir)

    return joblib.load(os.path.join(extract_dir, "model.joblib"))


def load_evaluation_frame(test_data_path):
    if os.path.isdir(test_data_path):
        csv_files = [name for name in os.listdir(test_data_path) if name.lower().endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in evaluation directory: {test_data_path}")
        test_data_path = os.path.join(test_data_path, csv_files[0])

    return pd.read_csv(test_data_path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.model_artifact)
    data = load_evaluation_frame(args.test_data)
    if args.target_column not in data.columns:
        raise ValueError(f"Target column '{args.target_column}' was not found in evaluation data.")

    x = data.drop(columns=[args.target_column])
    y = data[args.target_column]
    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=args.random_state,
    )

    predictions = model.predict(x_test)
    report = {
        "regression_metrics": {
            "mae": {"value": mean_absolute_error(y_test, predictions)},
            "mse": {"value": mean_squared_error(y_test, predictions)},
            "r2": {"value": r2_score(y_test, predictions)},
        }
    }

    with open(os.path.join(args.output_dir, "evaluation.json"), "w", encoding="utf-8") as output_file:
        json.dump(report, output_file)


if __name__ == "__main__":
    main()
