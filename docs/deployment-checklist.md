# Deployment Checklist

Use this checklist when moving the project into AWS.

## Prerequisites

- AWS CLI is authenticated for account `659613508664` or your target account.
- The target region is `us-west-1` unless you update `config/default.yaml`.
- S3 artifact/model bucket exists.
- SageMaker execution role exists and trusts SageMaker.
- CodeStar connection is created and authorized for your source repository.
- OpenAI API key is stored locally for development or in AWS Secrets Manager for CI.

## Configure Project

Update `config/default.yaml`:

- `aws_region`
- `default_bucket`
- `sagemaker_execution_role_arn`
- `model.model_package_group_name`
- `endpoints.staging.name`
- `endpoints.production.name`
- `openai.model`

Update `infrastructure/codepipeline-parameters.example.env` or `infrastructure/codepipeline-parameters.example.json`:

- `ArtifactBucketName`
- `CodeStarConnectionArn`
- `RepositoryId`
- `BranchName`
- `OpenAIApiKeySecretArn`

## Create OpenAI Secret In AWS

```bash
aws secretsmanager create-secret \
  --name openai/api-key \
  --secret-string "$OPENAI_API_KEY" \
  --region us-west-1
```

Use the returned ARN as `OpenAIApiKeySecretArn`.

## Validate Locally

```bash
python -m compileall src pipelines scripts
python pipelines/run_pipeline.py --config config/default.yaml --dry-run
python scripts/prompt_ops_invoke_endpoint.py --help
```

## Deploy Pipeline

```bash
aws cloudformation deploy \
  --template-file infrastructure/codepipeline.yaml \
  --stack-name agentic-rag-open-ai-cicd \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    ProjectName=agentic-rag-open-ai \
    ArtifactBucketName=mlopswithsagemaker111 \
    CodeStarConnectionArn=arn:aws:codestar-connections:us-west-1:659613508664:connection/REPLACE_WITH_CONNECTION_ID \
    RepositoryId=REPLACE_WITH_OWNER/REPLACE_WITH_REPOSITORY \
    BranchName=main \
    DeployEnvironment=staging \
    OpenAIApiKeySecretArn=arn:aws:secretsmanager:us-west-1:659613508664:secret:openai/api-key-REPLACE
```

## Post-Deploy Checks

- Confirm CodePipeline starts from the source stage.
- Review SageMaker evaluation metrics before manual approval.
- Confirm the endpoint exists after the deploy stage.
- Confirm `TestEndpoint` passes.
- Confirm `PromptOpsRealtimeTest` passes.
- Confirm `RagOpsRealtimeTest` and `AgenticOpsRealtimeTest` pass.

## Manual Prompt Ops Invocation

```bash
python scripts/prompt_ops_invoke_endpoint.py \
  --config config/default.yaml \
  --environment staging \
  --description "A 28 year old customer encoded as gender 0 has a vehicle with 23 miles, no debt, and income of 4099." \
  --explain
```
