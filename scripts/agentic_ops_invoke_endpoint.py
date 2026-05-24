import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.agentic_ops import (
    build_agentic_payload,
    build_agentic_rag_payload,
    explain_agentic_prediction,
)


def load_config(path):
    import yaml

    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--environment", choices=["staging", "production"], default="staging")
    parser.add_argument("--scenario", required=True, help="Hospital scenario for the three-agent inference workflow.")
    parser.add_argument("--explain", action="store_true", help="Ask OpenAI to explain the endpoint result.")
    parser.add_argument("--use-rag", action="store_true", help="Retrieve local context before running the three-agent workflow.")
    parser.add_argument("--top-k", type=int, help="Number of local knowledge chunks to retrieve when --use-rag is set.")
    return parser.parse_args()


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
    rag_config = config.get("rag", {})
    openai_model = openai_config.get("model", "gpt-5.2")
    max_output_tokens = openai_config.get("max_output_tokens", 700)
    knowledge_base_path = rag_config.get("knowledge_base_path", "data/rag/car_sales_knowledge.jsonl")
    top_k = args.top_k or rag_config.get("top_k", 3)

    if args.use_rag:
        agentic_result = build_agentic_rag_payload(
            args.scenario,
            knowledge_base_path,
            feature_columns,
            openai_model,
            max_output_tokens,
            top_k,
        )
    else:
        agentic_result = build_agentic_payload(
            args.scenario,
            feature_columns,
            openai_model,
            max_output_tokens,
        )
    endpoint_name, response_body = invoke_endpoint(config, args.environment, agentic_result["payload"])
    predictions = response_body.get("predictions", [])
    prediction = predictions[0] if predictions else None

    result = {
        "endpoint_name": endpoint_name,
        "scenario": args.scenario,
        "retrieved_context": agentic_result.get("context_documents", []),
        "agents": agentic_result["agents"],
        "features": agentic_result["features"],
        "payload": agentic_result["payload"],
        "response": response_body,
    }

    if args.explain:
        result["explanation"] = explain_agentic_prediction(
            args.scenario,
            agentic_result["agents"],
            agentic_result["features"],
            prediction,
            openai_model,
            max_output_tokens,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
