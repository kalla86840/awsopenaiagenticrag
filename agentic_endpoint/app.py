import json
import os
import re
from collections import Counter
from pathlib import Path

import boto3
import yaml
from openai import OpenAI


PROFILE_PATH = Path(__file__).with_name("agent_profiles.yaml")
DEFAULT_RAG_PATH = Path(__file__).with_name("hospital_agentic_rag_knowledge.txt")

AGENT_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "string"}},
        "next_actions": {"type": "array", "items": {"type": "string"}},
        "risk_level": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
    },
    "required": ["summary", "findings", "next_actions", "risk_level"],
}

INFERENCE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "case_summary": {"type": "string"},
        "care_team_consensus": {"type": "string"},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
        "signals_to_monitor": {"type": "array", "items": {"type": "string"}},
        "escalation_level": {"type": "string", "enum": ["routine", "urgent", "emergent"]},
        "handoff": {"type": "string"},
    },
    "required": [
        "case_summary",
        "care_team_consensus",
        "recommended_actions",
        "signals_to_monitor",
        "escalation_level",
        "handoff",
    ],
}


def response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "content-type": "application/json",
            "access-control-allow-origin": "*",
        },
        "body": json.dumps(body),
    }


def load_profiles():
    with open(PROFILE_PATH, "r", encoding="utf-8") as profile_file:
        return yaml.safe_load(profile_file)


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def load_text_knowledge_base(path):
    content = Path(path).read_text(encoding="utf-8").strip()
    sections = [
        section.strip()
        for section in re.split(r"\n\s*\n", content)
        if section.strip()
    ]
    if not sections:
        raise ValueError(f"No RAG sections were found in {path}.")

    documents = []
    for index, section in enumerate(sections, start=1):
        lines = section.splitlines()
        title = lines[0].strip("# ").strip() if lines else f"Section {index}"
        documents.append(
            {
                "id": f"hospital-rag-{index}",
                "title": title or f"Section {index}",
                "source": str(path),
                "content": section,
            }
        )
    return documents


def retrieve_context(payload):
    knowledge_path = Path(payload.get("rag_knowledge_path") or os.getenv("RAG_KNOWLEDGE_PATH", DEFAULT_RAG_PATH))
    top_k = int(payload.get("rag_top_k") or os.getenv("RAG_TOP_K", "4"))
    documents = load_text_knowledge_base(knowledge_path)
    query = json.dumps(
        {
            "task": payload.get("task", ""),
            "chief_concern": payload.get("chief_concern", ""),
            "vitals": payload.get("vitals", {}),
            "signals": payload.get("signals", {}),
            "notes": payload.get("notes", []),
            "requested_inference": payload.get("requested_inference", ""),
        }
    )
    query_terms = Counter(_tokenize(query))

    scored_documents = []
    for document in documents:
        document_terms = Counter(_tokenize(document["title"] + "\n" + document["content"]))
        overlap_score = sum(query_terms[term] * document_terms.get(term, 0) for term in query_terms)
        scored_documents.append((overlap_score, document))

    scored_documents.sort(key=lambda item: item[0], reverse=True)
    selected = [document for score, document in scored_documents if score > 0]
    if not selected:
        selected = [document for _, document in scored_documents]
    return selected[:top_k]


def parse_body(event):
    body = event.get("body", event)
    if isinstance(body, str):
        return json.loads(body or "{}")
    return body or {}


def get_openai_api_key():
    direct_key = os.getenv("OPENAI_API_KEY")
    if direct_key:
        return direct_key

    secret_arn = os.getenv("OPENAI_API_KEY_SECRET_ARN")
    if not secret_arn:
        raise RuntimeError("OPENAI_API_KEY_SECRET_ARN or OPENAI_API_KEY must be set.")

    client = boto3.client("secretsmanager")
    secret = client.get_secret_value(SecretId=secret_arn)
    return secret["SecretString"]


def build_agent_input(payload, prior_outputs, retrieved_context):
    return json.dumps(
        {
            "task": payload.get("task", "Create an agentic hospital care-coordination inference."),
            "patient_context": payload.get("patient_context", {}),
            "chief_concern": payload.get("chief_concern", ""),
            "vitals": payload.get("vitals", {}),
            "signals": payload.get("signals", {}),
            "notes": payload.get("notes", []),
            "requested_inference": payload.get("requested_inference", ""),
            "retrieved_context": retrieved_context,
            "prior_agent_outputs": prior_outputs,
        },
        indent=2,
    )


def call_agent(client, agent_name, profile, payload, prior_outputs, retrieved_context, max_output_tokens):
    result = client.responses.create(
        model=payload.get("model") or profile.get("model") or os.getenv("OPENAI_MODEL", "gpt-5.2"),
        instructions=profile["instructions"],
        input=build_agent_input(payload, prior_outputs, retrieved_context),
        text={
            "format": {
                "type": "json_schema",
                "name": f"{agent_name}_agent_result",
                "schema": AGENT_OUTPUT_SCHEMA,
                "strict": True,
            }
        },
        max_output_tokens=max_output_tokens,
    )
    return json.loads(result.output_text)


def run_final_inference(client, payload, agent_outputs, retrieved_context, max_output_tokens):
    instructions = payload.get("coordinator_instructions") or (
        "You are the agentic care-coordination endpoint. Synthesize the hospital, "
        "doctor, and nurse agent outputs into a real-time inference result for "
        "hospital operations. Use the retrieved RAG context as grounding. This is "
        "decision support only; avoid claiming a diagnosis or replacing clinician "
        "judgment."
    )
    result = client.responses.create(
        model=payload.get("model") or os.getenv("OPENAI_MODEL", "gpt-5.2"),
        instructions=instructions,
        input=json.dumps(
            {
                "request": payload,
                "retrieved_context": retrieved_context,
                "agent_outputs": agent_outputs,
            },
            indent=2,
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "agentic_hospital_inference",
                "schema": payload.get("response_schema") or INFERENCE_SCHEMA,
                "strict": True,
            }
        },
        max_output_tokens=max_output_tokens,
    )
    return json.loads(result.output_text)


def lambda_handler(event, context):
    try:
        payload = parse_body(event)
        profiles = load_profiles()
        max_output_tokens = int(payload.get("max_output_tokens") or os.getenv("MAX_OUTPUT_TOKENS", "1400"))
        requested_agents = payload.get("agents") or ["hospital", "doctor", "nurse"]
        retrieved_context = retrieve_context(payload)

        client = OpenAI(api_key=get_openai_api_key())
        agent_outputs = []
        for agent_name in requested_agents:
            profile = profiles.get(agent_name)
            if profile is None:
                return response(
                    400,
                    {
                        "error": f"Unknown agent '{agent_name}'.",
                        "available_agents": sorted(profiles.keys()),
                    },
                )

            agent_result = call_agent(
                client,
                agent_name,
                profile,
                payload,
                agent_outputs,
                retrieved_context,
                max_output_tokens,
            )
            agent_outputs.append(
                {
                    "agent": agent_name,
                    "result": agent_result,
                }
            )

        inference = run_final_inference(client, payload, agent_outputs, retrieved_context, max_output_tokens)
        return response(
            200,
            {
                "task": payload.get("task", "Create an agentic hospital care-coordination inference."),
                "retrieved_context": retrieved_context,
                "agents": agent_outputs,
                "inference": inference,
            },
        )
    except Exception as exc:
        return response(500, {"error": str(exc)})
