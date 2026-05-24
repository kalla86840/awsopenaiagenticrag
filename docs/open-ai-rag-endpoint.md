# OpenAI RAG Endpoint

This endpoint is a real-time AWS Lambda Function URL for OpenAI-backed multi-agent RAG inference. It retrieves context from a bundled `.txt` file, calls multiple OpenAI agents with the retrieved sections, and returns a final answer with agent outputs, steps, safety notes, citations, and retrieved context.

## Files

- `rag_endpoint/app.py`: Lambda handler and retrieval logic.
- `rag_endpoint/knowledge/open_ai_rag_knowledge.txt`: Plain text RAG source file.
- `rag_endpoint/requirements.txt`: Lambda package dependencies.
- `infrastructure/open-ai-rag-endpoint.yaml`: Lambda Function URL CloudFormation template.
- `infrastructure/open-ai-rag-endpoint-cicd.yaml`: CodePipeline/CodeBuild template for the endpoint.
- `buildspec-open-ai-rag-endpoint.yml`: Packages, deploys, and smoke-tests the endpoint.
- `samples/open_ai_rag_endpoint_request.json`: Example request.
- `samples/open_ai_rag_endpoint_response.example.json`: Example response.

## Request

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

The default agents are:

- `manual_retrieval_agent`: document retrieval analyst.
- `procedure_agent`: step-by-step procedure specialist.
- `safety_agent`: safety and escalation reviewer.

## Response

The endpoint returns:

```json
{
  "question": "How do I connect a washer and dryer?",
  "agents": [],
  "answer": "OpenAI-generated answer grounded in retrieved context.",
  "steps": [],
  "safety_notes": [],
  "citations": [
    {
      "id": "doc-3",
      "title": "Washer Water Connections"
    }
  ],
  "agent_consensus": "Summary of how the agents agree.",
  "retrieved_context": []
}
```

## Deploy

Create the CI/CD pipeline:

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

The pipeline deploys the `open-ai-rag-endpoint` stack. Use the `EndpointUrl` output for real-time inference.
