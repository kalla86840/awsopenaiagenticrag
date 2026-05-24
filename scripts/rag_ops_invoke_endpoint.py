import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.rag_ops import build_rag_payload, explain_rag_prediction


def load_config(path):
    import yaml

    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--environment", choices=["staging", "production"], default="staging")
    parser.add_argument("--query", help="User question or scenario for RAG inference.")
    parser.add_argument("--request-file", type=Path, help="JSON file containing a RAG inference request.")
    parser.add_argument("--explain", action="store_true", help="Ask OpenAI to explain the endpoint result with retrieved context.")
    parser.add_argument("--top-k", type=int, help="Number of local knowledge chunks to retrieve.")
    return parser.parse_args()


def load_request(args):
    request = {}
    if args.request_file:
        with open(args.request_file, "r", encoding="utf-8") as request_file:
            request = json.load(request_file)

    query = args.query or request.get("query")
    if not query:
        raise ValueError("Provide a RAG inference query with --query or request_file.query.")

    return {
        "query": query,
        "environment": request.get("environment", args.environment),
        "top_k": args.top_k or request.get("top_k"),
        "explain": args.explain or bool(request.get("explain", False)),
    }


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
    request = load_request(args)
    config = load_config(args.config)
    feature_columns = config["model"]["feature_columns"]
    openai_config = config.get("openai", {})
    rag_config = config.get("rag", {})
    openai_model = openai_config.get("model", "gpt-5.2")
    max_output_tokens = openai_config.get("max_output_tokens", 700)
    knowledge_base_path = rag_config.get("knowledge_base_path", "data/rag/car_sales_knowledge.txt")
    top_k = request["top_k"] or rag_config.get("top_k", 3)

    rag_result = build_rag_payload(
        request["query"],
        knowledge_base_path,
        feature_columns,
        openai_model,
        max_output_tokens,
        top_k,
    )
    endpoint_name, response_body = invoke_endpoint(config, request["environment"], rag_result["payload"])
    predictions = response_body.get("predictions", [])
    prediction = predictions[0] if predictions else None

    result = {
        "endpoint_name": endpoint_name,
        "query": request["query"],
        "retrieved_context": rag_result["context_documents"],
        "features": rag_result["features"],
        "internal_endpoint_payload": rag_result["payload"],
        "endpoint_response": response_body,
    }

    if request["explain"]:
        result["explanation"] = explain_rag_prediction(
            request["query"],
            rag_result["context_documents"],
            rag_result["features"],
            prediction,
            openai_model,
            max_output_tokens,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
