import json
import os
import re
from collections import Counter
from pathlib import Path

KNOWLEDGE_PATH = Path(os.environ.get("RAG_KNOWLEDGE_PATH", "knowledge/open_ai_rag_knowledge.txt"))
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2")
OPENAI_API_KEY_SECRET_ARN = os.environ.get("OPENAI_API_KEY_SECRET_ARN")
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "700"))
TOP_K = int(os.environ.get("TOP_K", "3"))
_OPENAI_API_KEY = None

AGENTS = [
    {
        "name": "manual_retrieval_agent",
        "role": "Document retrieval analyst",
        "instructions": (
            "You are Agent 1, the document retrieval analyst. Use only the retrieved "
            "manual sections to identify the sections that answer the question. "
            "Return practical findings and cite section titles. If the retrieved "
            "context is insufficient, say what is missing."
        ),
    },
    {
        "name": "procedure_agent",
        "role": "Step-by-step procedure specialist",
        "instructions": (
            "You are Agent 2, the procedure specialist. Use only the retrieved manual "
            "sections to extract an ordered, user-safe procedure. Preserve important "
            "sequence, prerequisites, and checks from the document."
        ),
    },
    {
        "name": "safety_agent",
        "role": "Safety and escalation reviewer",
        "instructions": (
            "You are Agent 3, the safety reviewer. Use only the retrieved manual "
            "sections to identify warnings, stop conditions, and when a qualified "
            "professional should be contacted."
        ),
    },
]

AGENT_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "string"}},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["id", "title"],
            },
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["summary", "findings", "citations", "confidence"],
}

FINAL_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
        "steps": {"type": "array", "items": {"type": "string"}},
        "safety_notes": {"type": "array", "items": {"type": "string"}},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["id", "title"],
            },
        },
        "agent_consensus": {"type": "string"},
    },
    "required": ["answer", "steps", "safety_notes", "citations", "agent_consensus"],
}


def _tokenize(text):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    normalized = []
    for token in tokens:
        if token.startswith("connect"):
            normalized.append("connect")
        elif token.startswith("install"):
            normalized.append("install")
        elif token.startswith("leak"):
            normalized.append("leak")
        elif token.endswith("s") and len(token) > 3:
            normalized.append(token[:-1])
        else:
            normalized.append(token)
    return normalized


def load_documents(path=KNOWLEDGE_PATH):
    content = Path(path).read_text(encoding="utf-8").strip()
    sections = [
        section.strip()
        for section in re.split(r"\n\s*\n", content)
        if section.strip()
    ]

    documents = []
    for index, section in enumerate(sections, start=1):
        lines = section.splitlines()
        title = lines[0].strip("# ").strip() if lines else f"Section {index}"
        documents.append(
            {
                "id": f"doc-{index}",
                "title": title,
                "content": section,
            }
        )
    return documents


def retrieve(question, documents, top_k=TOP_K):
    question_terms = Counter(_tokenize(question))
    scored = []
    for document in documents:
        document_terms = Counter(_tokenize(f"{document['title']} {document['content']}"))
        score = sum(
            question_terms[term] * document_terms.get(term, 0)
            for term in question_terms
        )
        scored.append((score, document))

    scored.sort(key=lambda item: item[0], reverse=True)
    matches = [document for score, document in scored if score > 0]
    if not matches:
        matches = [document for _, document in scored]
    return matches[:top_k]


def call_agent(client, agent, question, context_documents):
    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=agent["instructions"],
        input=json.dumps(
            {
                "question": question,
                "retrieved_context": context_documents,
            }
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": f"{agent['name']}_result",
                "schema": AGENT_OUTPUT_SCHEMA,
                "strict": True,
            }
        },
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return json.loads(response.output_text)


def synthesize_answer(client, question, context_documents, agent_outputs):
    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=(
            "You are the final coordinator for an AWS-hosted multi-agent RAG endpoint. "
            "Use only the retrieved document context and the three agent outputs. "
            "Answer the user's question clearly, cite section titles, and include "
            "safety notes. If the manual does not provide enough information, say so."
        ),
        input=json.dumps(
            {
                "question": question,
                "retrieved_context": context_documents,
                "agent_outputs": agent_outputs,
            }
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "multi_agent_rag_answer",
                "schema": FINAL_OUTPUT_SCHEMA,
                "strict": True,
            }
        },
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return json.loads(response.output_text)


def answer_question(question, context_documents, requested_agents=None):
    from openai import OpenAI

    client = OpenAI(api_key=get_openai_api_key())
    requested_agents = requested_agents or [agent["name"] for agent in AGENTS]
    selected_agents = [agent for agent in AGENTS if agent["name"] in requested_agents]
    if not selected_agents:
        raise ValueError("No valid agents were requested.")

    agent_outputs = []
    for agent in selected_agents:
        result = call_agent(client, agent, question, context_documents)
        agent_outputs.append(
            {
                "agent": agent["name"],
                "role": agent["role"],
                "result": result,
            }
        )

    final = synthesize_answer(client, question, context_documents, agent_outputs)
    return agent_outputs, final


def get_openai_api_key():
    global _OPENAI_API_KEY
    if _OPENAI_API_KEY:
        return _OPENAI_API_KEY

    api_key = os.environ.get
