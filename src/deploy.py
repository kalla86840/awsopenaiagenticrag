import argparse
from pathlib import Path

import boto3
import sagemaker
import yaml
from sagemaker import ModelPackage


def load_config(path):
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def latest_approved_model_package(sm_client, package_group_name):
    paginator = sm_client.get_paginator("list_model_packages")
    pages = paginator.paginate(
        ModelPackageGroupName=package_group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
    )

    for page in pages:
        summaries = page.get("ModelPackageSummaryList", [])
        if summaries:
            return summaries[0]["ModelPackageArn"]

    raise RuntimeError(f"No approved model packages found in group: {package_group_name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--environment", choices=["staging", "production"], default="staging")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    endpoint_config = config["endpoints"][args.environment]

    boto_session = boto3.Session(region_name=config["aws_region"])
    sm_client = boto_session.client("sagemaker")
    session = sagemaker.Session(
        boto_session=boto_session,
        default_bucket=config["default_bucket"],
    )

    package_arn = latest_approved_model_package(
        sm_client,
        config["model"]["model_package_group_name"],
    )

    model = ModelPackage(
        role=config["sagemaker_execution_role_arn"],
        model_package_arn=package_arn,
        sagemaker_session=session,
    )
    model.deploy(
        endpoint_name=endpoint_config["name"],
        initial_instance_count=endpoint_config["initial_instance_count"],
        instance_type=endpoint_config["instance_type"],
    )

    print(f"Deployed {package_arn} to endpoint {endpoint_config['name']}")


if __name__ == "__main__":
    main()
