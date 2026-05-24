import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--target-column", type=str, default="sales")
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "data/raw"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    return parser.parse_args()


def load_training_frame(train_path):
    if os.path.isdir(train_path):
        csv_files = [name for name in os.listdir(train_path) if name.lower().endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in training directory: {train_path}")
        train_path = os.path.join(train_path, csv_files[0])

    return pd.read_csv(train_path)


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)

    data = load_training_frame(args.train)
    if args.target_column not in data.columns:
        raise ValueError(f"Target column '{args.target_column}' was not found in training data.")

    x = data.drop(columns=[args.target_column])
    y = data[args.target_column]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=args.random_state,
    )

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

    with open(os.path.join(args.output_data_dir, "metrics.txt"), "w", encoding="utf-8") as metrics_file:
        metrics_file.write(f"mae={mae:.6f}\n")
        metrics_file.write(f"r2={r2:.6f}\n")


if __name__ == "__main__":
    main()
