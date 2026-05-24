import json
import re
from collections import Counter
from pathlib import Path

from src.prompt_ops import build_payload


FEATURE_DEFINITIONS = {
    "age": "Person or patient age in years.",
    "gender": "Numeric gender encoding expected by the trained endpoint.",
    "miles": "Operational numeric measure expected by the trained endpoint.",
    "debt": "Numeric risk or liability value expected by the trained endpoint.",
    "income": "Numeric financial or resource value expected by the trained endpoint.",
}


def _feature_schema(feature_columns):
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            column: {
                "type": "number",
                "description": FEATURE_DEFINITIONS.get(column, f"Feature value for {column}."),
            }
            for column in feature_columns
        },
        "required": list(feature_columns),
    }


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def load_knowledge_base(path):
    knowledge_path = Path(path)
    if knowledge_path.suffix.lower() == ".txt":
        return load_text_knowledge_base(knowledge_path)

    documents = []
    with knowledge_path.open("r", encoding="utf-8") as knowledge_file:
        for line_number, line in enumerate(knowledge_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            content = record.get("content", "")
            if not content:
                raise ValueError(f"Knowledge record on line {line_number} is missing content.")
            documents.append(
                {
                    "id": record.get("id", f"line-{line_number}"),
                    "title": record.get("title", record.get("id", f"line-{line_number}")),
                    "source": record.get("source", str(knowledge_path)),
                    "tags": record.get("tags", []),
                    "content": content,
                }
            )
    return documents


def load_text_knowledge_base(path):
    content = path.read_text(encoding="utf-8").strip()
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
                "id": f"text-section-{index}",
                "title": title or f"Section {index}",
                "source": str(path),
                "tags": [],
                "content": section,
            }
        )

    if not documents:
        raise ValueError(f"No RAG text sections were found in {path}.")
    return documents


def load_documents(path):
    return load_knowledge_base(path)


def retrieve_context(query, knowledge_base_path, top_k):
    documents = load_knowledge_base(knowledge_base_path)
    query_terms = Counter(_tokenize(query))

    scored_documents = []
    for document in documents:
        searchable_text = " ".join(
            [
                document["title"],
                document["content"],
                " ".join(document.get("tags", [])),
            ]
        )
        document_terms = Counter(_tokenize(searchable_text))
        overlap_score = sum(
            query_terms[term] * document_terms.get(term, 0)
            for term in query_terms
        )
        tag_bonus = sum(1 for tag in document.get("tags", []) if tag.lower() in query.lower())
        scored_documents.append((overlap_score + tag_bonus, document))

    scored_documents.sort(key=lambda item: item[0], reverse=True)
    selected = [document for score, document in scored_documents if score > 0]
    if not selected:
        selected = [document for _, document in scored_documents]
    return selected[:top_k]


def extract_features_with_context(
    query,
    context_documents,
    feature_columns,
    model,
    max_output_tokens,
):
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=(
            "You are a retrieval-augmented endpoint inference coordinator. "
            "Use the retrieved context and user scenario to extract the numeric "
            "features for the real-time endpoint. Return only values that match "
            "the provided JSON schema. Use conservative numeric defaults only "
            "when the retrieved context or scenario clearly supports them."
        ),
        input=json.dumps(
            {
                "query": query,
                "feature_columns": feature_columns,
                "retrieved_context": context_documents,
            }
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "rag_endpoint_features",
                "schema": _feature_schema(feature_columns),
                "strict": True,
            }
        },
        max_output_tokens=max_output_tokens,
    )
    return json.loads(response.output_text)


def build_rag_payload(
    query,
    knowledge_base_path,
    feature_columns,
    model,
    max_output_tokens,
    top_k,
):
    context_documents = retrieve_context(query, knowledge_base_path, top_k)
    features = extract_features_with_context(
        query,
        context_documents,
        feature_columns,
        model,
        max_output_tokens,
    )
    return {
        "context_documents": context_documents,
        "features": features,
        "payload": build_payload(feature_columns, features),
    }


def explain_rag_prediction(query, context_documents, features, prediction, model, max_output_tokens):
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=(
            "Explain the real-time endpoint result for an operations user. "
            "Ground the explanation in the retrieved context, cite context titles "
            "when useful, and avoid unsupported certainty."
        ),
        input=json.dumps(
            {
                "query": query,
                "retrieved_context": context_documents,
                "features": features,
                "prediction": prediction,
            }
        ),
        max_output_tokens=max_output_tokens,
    )
    return response.output_text
