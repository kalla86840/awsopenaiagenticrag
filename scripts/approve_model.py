import argparse
from pathlib import Path

import boto3
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def latest_pending_model_package(sm_client, package_group_name):
    paginator = sm_client.get_paginator("list_model_packages")
    pages = paginator.paginate(
        ModelPackageGroupName=package_group_name,
        ModelApprovalStatus="PendingManualApproval",
        SortBy="CreationTime",
        SortOrder="Descending",
    )

    for page in pages:
        summaries = page.get("ModelPackageSummaryList", [])
        if summaries:
            return summaries[0]["ModelPackageArn"]

    raise RuntimeError(f"No pending model packages found in group: {package_group_name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    sm_client = boto3.client("sagemaker", region_name=config["aws_region"])
    package_arn = latest_pending_model_package(
        sm_client,
        config["model"]["model_package_group_name"],
    )

    sm_client.update_model_package(
        ModelPackageArn=package_arn,
        ModelApprovalStatus="Approved",
        ApprovalDescription="Approved by CI/CD release workflow.",
    )
    print(f"Approved model package: {package_arn}")


if __name__ == "__main__":
    main()
