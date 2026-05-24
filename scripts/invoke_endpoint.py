import argparse
import json
from pathlib import Path

import boto3
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--environment", choices=["staging", "production"], default="staging")
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--gender", type=float, required=True)
    parser.add_argument("--miles", type=float, required=True)
    parser.add_argument("--debt", type=float, required=True)
    parser.add_argument("--income", type=float, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    endpoint_name = config["endpoints"][args.environment]["name"]
    values = {
        "age": args.age,
        "gender": args.gender,
        "miles": args.miles,
        "debt": args.debt,
        "income": args.income,
    }
    payload = {
        "instances": [[values[column] for column in config["model"]["feature_columns"]]]
    }

    runtime_client = boto3.client("sagemaker-runtime", region_name=config["aws_region"])
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    print(response["Body"].read().decode("utf-8"))


if __name__ == "__main__":
    main()
