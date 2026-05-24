import argparse
import time
from pathlib import Path

import boto3

from pipeline import create_pipeline, load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def upload_dataset(config):
    local_path = Path(config["data"]["local_path"])
    if not local_path.exists():
        raise FileNotFoundError(f"Dataset not found: {local_path}")

    s3_client = boto3.client("s3", region_name=config["aws_region"])
    key = f"{config['data']['s3_prefix'].rstrip('/')}/{local_path.name}"
    s3_client.upload_file(str(local_path), config["default_bucket"], key)
    return f"s3://{config['default_bucket']}/{key}"


def wait_for_execution(execution, poll_seconds):
    while True:
        description = execution.describe()
        status = description["PipelineExecutionStatus"]
        print(f"Pipeline execution status: {status}")

        if status == "Succeeded":
            return
        if status in {"Failed", "Stopped"}:
            raise RuntimeError(f"Pipeline execution ended with status: {status}")

        time.sleep(poll_seconds)


def main():
    args = parse_args()
    config = load_config(args.config)
    pipeline = create_pipeline(str(args.config))

    definition = pipeline.definition()
    if args.dry_run:
        print(definition)
        return

    uploaded_uri = upload_dataset(config)
    print(f"Uploaded dataset to {uploaded_uri}")

    pipeline.upsert(role_arn=config["sagemaker_execution_role_arn"])
    execution = pipeline.start()
    print(f"Started pipeline execution: {execution.arn}")

    if args.wait:
        wait_for_execution(execution, args.poll_seconds)


if __name__ == "__main__":
    main()
