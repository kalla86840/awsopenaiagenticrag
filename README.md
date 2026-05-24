# Build Agentic RAG Pipeline

This repository contains AWS CI/CD pipelines for OpenAI-backed real-time RAG endpoints.

It now includes two deployable endpoint paths:

1. `rag_endpoint/`: a multi-agent OpenAI RAG Lambda Function URL that retrieves from a plain text knowledge file and applies multiple agents before returning an answer.
2. `agentic_endpoint/`: an agentic hospital RAG Lambda Function URL that retrieves hospital context, runs three OpenAI agents, and returns a structured care-coordination inference.

## Architecture

```text
GitHub repository
  -> AWS CodePipeline
  -> AWS CodeBuild packages endpoint code, dependencies, and RAG text files
  -> CodeBuild uploads Lambda artifacts to S3
  -> CloudFormation deploys Lambda Function URL endpoints
  -> Lambda retrieves relevant RAG context
  -> Lambda calls OpenAI
  -> Endpoint returns grounded real-time inference
```

## Standalone OpenAI RAG Endpoint

This endpoint answers questions from a bundled `.txt` user manual using multiple OpenAI agents. The Lambda loads `rag_endpoint/knowledge/open_ai_rag_knowledge.txt`, splits it into retrievable sections, retrieves the most relevant sections for the user question, runs three agents over that retrieved context, and returns a final synthesized answer.

Runtime files:

```text
rag_endpoint/app.py
rag_endpoint/requirements.txt
rag_endpoint/knowledge/open_ai_rag_knowledge.txt
```

AWS pipeline files:

```text
buildspec-open-ai-rag-endpoint.yml
infrastructure/open-ai-rag-endpoint.yaml
infrastructure/open-ai-rag-endpoint-cicd.yaml
```

Sample files:

```text
samples/open_ai_rag_endpoint_request.json
samples/open_ai_rag_endpoint_response.example.json
```

Request shape:

```json
{
  "question": "How do I connect a washer and dryer?",
  "top_k": 8,
  "agents": [
    "manual_retrieval_agent",
    "procedure_agent",
    "safety_agent"
  ]
}
```

Agent sequence:

- `manual_retrieval_agent`: identifies the most relevant manual sections and citations.
- `procedure_agent`: turns retrieved context into ordered steps.
- `safety_agent`: extracts warnings, stop conditions, and professional escalation guidance.

The endpoint response includes `agents`, `answer`, `steps`, `safety_notes`, `citations`, `agent_consensus`, and `retrieved_context`.

After deployment, get the URL from the `open-ai-rag-endpoint` CloudFormation stack output named `EndpointUrl`.

## Agentic Hospital RAG Endpoint

This endpoint uses OpenAI to run three agents:

- Agent 1: `hospital`, for operational intake and coordination.
- Agent 2: `doctor`, for clinical review and escalation considerations.
- Agent 3: `nurse`, for bedside handoff and monitoring priorities.

Runtime files:

```text
agentic_endpoint/app.py
agentic_endpoint/agent_profiles.yaml
agentic_endpoint/hospital_agentic_rag_knowledge.txt
agentic_endpoint/requirements.txt
```

AWS pipeline files:

```text
buildspec-agentic.yml
infrastructure/agentic-endpoint.yaml
infrastructure/agentic-cicd.yaml
```

Sample files:

```text
data/hospital_aiops/patient_events.jsonl
samples/agentic_hospital_request.json
samples/agentic_hospital_sample_inference.json
```

After deployment, get the URL from the `agentic-open-ai-agentic-endpoint` CloudFormation stack output named `AgenticFunctionUrl`.

## AWS And OpenAI Configuration

The OpenAI API key comes from OpenAI. Store it in AWS Secrets Manager and pass the secret ARN to the pipeline.

The AWS defaults currently filled in are:

- Region: `us-west-1`
- Artifact bucket: `mlopswithsagemaker111`
- CodeStar connection: `arn:aws:codeconnections:us-west-1:659613508664:connection/4ea8863c-728d-450a-8752-251946939b36`
- GitHub repository: `kalla86840/awspromptops`
- OpenAI secret ARN: `arn:aws:secretsmanager:us-west-1:659613508664:secret:openai/api-key-6BGXhJ`

Update an existing secret:

```bash
aws secretsmanager put-secret-value \
  --region us-west-1 \
  --secret-id openai/api-key \
  --secret-string "sk-your-openai-api-key"
```

## Deploy Standalone RAG Endpoint Pipeline

```bash
aws cloudformation deploy \
  --region us-west-1 \
  --template-file infrastructure/open-ai-rag-endpoint-cicd.yaml \
  --stack-name open-ai-rag-endpoint-cicd \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    ProjectName=open-ai-rag \
    ArtifactBucketName=mlopswithsagemaker111 \
    CodeStarConnectionArn=arn:aws:codeconnections:us-west-1:659613508664:connection/4ea8863c-728d-450a-8752-251946939b36 \
    RepositoryId=kalla86840/awspromptops \
    BranchName=main \
    OpenAIApiKeySecretArn=arn:aws:secretsmanager:us-west-1:659613508664:secret:openai/api-key-6BGXhJ
```

Invoke after deployment:

```bash
curl -X POST "$ENDPOINT_URL" \
  -H "content-type: application/json" \
  -d @samples/open_ai_rag_endpoint_request.json
```

## Deploy Agentic Hospital RAG Pipeline

```bash
aws cloudformation deploy \
  --region us-west-1 \
  --template-file infrastructure/agentic-cicd.yaml \
  --stack-name agentic-open-ai-cicd \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    ProjectName=agentic-open-ai \
    ArtifactBucketName=mlopswithsagemaker111 \
    CodeStarConnectionArn=arn:aws:codeconnections:us-west-1:659613508664:connection/4ea8863c-728d-450a-8752-251946939b36 \
    RepositoryId=kalla86840/awspromptops \
    BranchName=main \
    OpenAIApiKeySecretArn=arn:aws:secretsmanager:us-west-1:659613508664:secret:openai/api-key-6BGXhJ \
    OpenAIModel=gpt-5.2
```

Because the agentic endpoint uses `AWS_IAM` auth, invoke it with a SigV4-capable client:

```bash
curl \
  --request POST "$AGENTIC_FUNCTION_URL" \
  --aws-sigv4 "aws:amz:us-west-1:lambda" \
  --user "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY" \
  --header "x-amz-security-token: $AWS_SESSION_TOKEN" \
  --header "content-type: application/json" \
  --data-binary @samples/agentic_hospital_request.json
```

## Local Validation

```bash
python -m compileall rag_endpoint agentic_endpoint src scripts pipelines
python -c "from rag_endpoint.app import load_documents, retrieve; docs=load_documents('rag_endpoint/knowledge/open_ai_rag_knowledge.txt'); print([d['title'] for d in retrieve('How do I connect a washer and dryer?', docs, 8)])"
python -c "import sys, types, json; sys.modules['boto3']=types.SimpleNamespace(client=lambda *a, **k: None); sys.modules['openai']=types.SimpleNamespace(OpenAI=object); sys.modules['yaml']=types.SimpleNamespace(safe_load=lambda f: {}); import agentic_endpoint.app as app; payload=json.load(open('samples/agentic_hospital_request.json', encoding='utf-8')); print([d['title'] for d in app.retrieve_context(payload)])"
```
