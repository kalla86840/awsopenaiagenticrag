from pathlib import Path

import boto3
import sagemaker
import yaml
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.functions import Join


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def create_pipeline(config_path="config/default.yaml"):
    config = load_config(config_path)
    boto_session = boto3.Session(region_name=config["aws_region"])
    pipeline_session = PipelineSession(
        boto_session=boto_session,
        default_bucket=config["default_bucket"],
    )

    base_job_prefix = config["pipeline"]["base_job_prefix"]
    sklearn_image_uri = sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=config["aws_region"],
        version="1.2-1",
        py_version="py3",
        instance_type=config["model"]["training_instance_type"],
    )

    estimator = SKLearn(
        entry_point="train.py",
        source_dir=str(ROOT_DIR / "src"),
        framework_version="1.2-1",
        py_version="py3",
        role=config["sagemaker_execution_role_arn"],
        instance_type=config["model"]["training_instance_type"],
        instance_count=1,
        base_job_name=f"{base_job_prefix}-train",
        sagemaker_session=pipeline_session,
        hyperparameters={
            "n-estimators": 100,
            "random-state": 42,
            "target-column": config["model"]["target_column"],
        },
    )

    train_step = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=f"s3://{config['default_bucket']}/{config['data']['s3_prefix']}/",
                content_type="text/csv",
            )
        },
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    evaluation_processor = ScriptProcessor(
        image_uri=sklearn_image_uri,
        command=["python3"],
        role=config["sagemaker_execution_role_arn"],
        instance_count=1,
        instance_type=config["model"]["training_instance_type"],
        base_job_name=f"{base_job_prefix}-evaluate",
        sagemaker_session=pipeline_session,
    )

    evaluation_step = ProcessingStep(
        name="EvaluateModel",
        processor=evaluation_processor,
        inputs=[
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=f"s3://{config['default_bucket']}/{config['data']['s3_prefix']}/",
                destination="/opt/ml/processing/test",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
            )
        ],
        code=str(ROOT_DIR / "src" / "evaluate.py"),
        job_arguments=[
            "--target-column",
            config["model"]["target_column"],
        ],
        property_files=[evaluation_report],
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                evaluation_step.properties.ProcessingOutputConfig.Outputs[
                    "evaluation"
                ].S3Output.S3Uri,
                    "evaluation.json",
                ],
            ),
            content_type="application/json",
        )
    )

    model = SKLearnModel(
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=config["sagemaker_execution_role_arn"],
        entry_point="inference.py",
        source_dir=str(ROOT_DIR / "src"),
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=pipeline_session,
    )
    register_args = model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=[config["model"]["inference_instance_type"]],
        transform_instances=[config["model"]["inference_instance_type"]],
        model_package_group_name=config["model"]["model_package_group_name"],
        approval_status=config["model"]["approval_status"],
        model_metrics=model_metrics,
    )
    register_step = ModelStep(
        name="RegisterModel",
        step_args=register_args,
        depends_on=[evaluation_step],
    )

    return Pipeline(
        name=config["pipeline"]["name"],
        steps=[train_step, evaluation_step, register_step],
        sagemaker_session=pipeline_session,
    )
