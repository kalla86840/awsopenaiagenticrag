# OpenAI Prompt Ops Architecture

This project combines OpenAI prompt operations, RAG operations, and a three-agent hospital workflow with an AWS SageMaker real-time inference endpoint.

## Runtime Flow

```text
User or CI prompt
  -> OpenAI Responses API
  -> strict JSON feature extraction
  -> SageMaker Runtime invoke_endpoint
  -> trained regression model
  -> prediction JSON
  -> optional OpenAI explanation
```

## Agentic Runtime Flow

```text
Hospital workflow scenario
  -> Agent 1: hospital operations intake
  -> Agent 2: doctor validation
  -> Agent 3: nurse readiness and handoff
  -> OpenAI Responses API structured coordinator
  -> strict JSON endpoint payload
  -> SageMaker Runtime invoke_endpoint
  -> real-time endpoint response
  -> optional OpenAI explanation
```

## CI/CD Flow

```text
Source repo
  -> CodeBuild train/register
  -> SageMaker Pipeline train/evaluate/register package
  -> Manual approval
  -> CodeBuild approve model package
  -> CodeBuild deploy SageMaker endpoint
  -> CodeBuild smoke-test numeric endpoint invocation
  -> CodeBuild Prompt Ops endpoint invocation
  -> CodeBuild RAG Ops endpoint invocation
  -> CodeBuild Agentic Ops endpoint invocation
```

## Key Files

- `config/default.yaml`: AWS region, SageMaker bucket, model package group, endpoint names, OpenAI model.
- `src/prompt_ops.py`: OpenAI structured-output helpers and SageMaker payload builder.
- `src/agentic_ops.py`: Hospital, doctor, and nurse agent workflow helpers.
- `scripts/prompt_ops_invoke_endpoint.py`: CLI for natural-language or numeric endpoint invocation.
- `scripts/agentic_ops_invoke_endpoint.py`: CLI for the hospital/doctor/nurse endpoint invocation flow.
- `buildspec-prompt-ops.yml`: CodeBuild stage for Prompt Ops endpoint validation.
- `buildspec-agentic-ops.yml`: CodeBuild stage for Agentic Ops endpoint validation.
- `infrastructure/codepipeline.yaml`: CodePipeline and CodeBuild infrastructure.
- `infrastructure/codepipeline-parameters.example.json`: Parameter file template for stack deployment.
- `infrastructure/codepipeline-parameters.example.env`: Deploy-ready parameter override values.
- `.env.example`: Local environment placeholder values.

## Secret Handling

Local development reads `OPENAI_API_KEY` from the environment.

CI/CD reads `OPENAI_API_KEY` from AWS Secrets Manager when `OpenAIApiKeySecretArn` is supplied to the CloudFormation stack. Leave that parameter empty to run the Prompt Ops stage with numeric features only. The RAG Ops and Agentic Ops stages skip OpenAI-backed invocation until the secret is configured.

## Inference Contract

The trained SageMaker endpoint expects:

```json
{
  "instances": [[28, 0, 23, 0, 4099]]
}
```

Feature order:

```text
age, gender, miles, debt, income
```

The endpoint returns:

```json
{
  "predictions": [734.42]
}
```
