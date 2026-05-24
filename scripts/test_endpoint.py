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
    return parser.parse_args()


def sample_payload(config):
    feature_columns = config["model"]["feature_columns"]
    example = {
        "age": 28,
        "gender": 0,
        "miles": 23,
        "debt": 0,
        "income": 4099,
    }
    return {"instances": [[example[column] for column in feature_columns]]}


def main():
    args = parse_args()
    config = load_config(args.config)
    endpoint_name = config["endpoints"][args.environment]["name"]

    runtime_client = boto3.client("sagemaker-runtime", region_name=config["aws_region"])
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(sample_payload(config)),
    )
    body = json.loads(response["Body"].read().decode("utf-8"))
    predictions = body.get("predictions")

    if not predictions or not isinstance(predictions[0], (int, float)):
        raise RuntimeError(f"Endpoint returned an invalid prediction response: {body}")

    print(f"Endpoint smoke test passed for {endpoint_name}: {body}")


if __name__ == "__main__":
    main()
