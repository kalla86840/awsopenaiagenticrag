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
