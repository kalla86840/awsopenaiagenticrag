import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.prompt_ops import build_payload, explain_prediction, extract_features_from_text


def load_config(path):
    import yaml

    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--environment", choices=["staging", "production"], default="staging")
    parser.add_argument("--description", help="Natural-language scenario to parse with OpenAI.")
    parser.add_argument("--explain", action="store_true", help="Ask OpenAI to explain the prediction.")
    parser.add_argument("--age", type=float)
    parser.add_argument("--gender", type=float)
    parser.add_argument("--miles", type=float)
    parser.add_argument("--debt", type=float)
    parser.add_argument("--income", type=float)
    return parser.parse_args()


def values_from_args(args):
    values = {
        "age": args.age,
        "gender": args.gender,
        "miles": args.miles,
        "debt": args.debt,
        "income": args.income,
    }
    return {key: value for key, value in values.items() if value is not None}


def invoke_endpoint(config, environment, payload):
    import boto3

    endpoint_name = config["endpoints"][environment]["name"]
    runtime_client = boto3.client("sagemaker-runtime", region_name=config["aws_region"])
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    body = json.loads(response["Body"].read().decode("utf-8"))
    return endpoint_name, body


def main():
    args = parse_args()
    config = load_config(args.config)
    feature_columns = config["model"]["feature_columns"]
    openai_config = config.get("openai", {})
    openai_model = openai_config.get("model", "gpt-5.2")
    max_output_tokens = openai_config.get("max_output_tokens", 700)

    if args.description:
        values = extract_features_from_text(
            args.description,
            feature_columns,
            openai_model,
            max_output_tokens,
        )
    else:
        values = values_from_args(args)

    payload = build_payload(feature_columns, values)
    endpoint_name, response_body = invoke_endpoint(config, args.environment, payload)
    predictions = response_body.get("predictions", [])
    prediction = predictions[0] if predictions else None

    result = {
        "endpoint_name": endpoint_name,
        "features": values,
        "payload": payload,
        "response": response_body,
    }

    if args.explain:
        result["explanation"] = explain_prediction(
            args.description or "Numeric feature invocation.",
            values,
            prediction,
            openai_model,
            max_output_tokens,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
