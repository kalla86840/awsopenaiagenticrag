import json

from src.prompt_ops import build_payload
from src.rag_ops import retrieve_context


AGENT_DEFINITIONS = [
    {
        "name": "hospital",
        "role": "Hospital operations agent",
        "instructions": (
            "You are the hospital operations agent. Convert the scenario into a "
            "structured operational intake for endpoint inference. Focus on the "
            "patient or case attributes that map to the model feature schema."
        ),
    },
    {
        "name": "doctor",
        "role": "Doctor agent",
        "instructions": (
            "You are the doctor agent. Review the hospital intake for clinical "
            "plausibility and identify the final numeric features that should be "
            "sent to the real-time endpoint."
        ),
    },
    {
        "name": "nurse",
        "role": "Nurse agent",
        "instructions": (
            "You are the nurse agent. Perform a final readiness check, make sure "
            "all required endpoint fields are present, and prepare a concise "
            "handoff summary for the inference call."
        ),
    },
]


FEATURE_DEFINITIONS = {
    "age": "Patient age in years.",
    "gender": "Numeric gender encoding expected by the trained endpoint.",
    "miles": "Operational numeric measure expected by the trained endpoint. Use a directly stated value when available.",
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


def _call_agent(client, model, agent, input_payload, max_output_tokens):
    response = client.responses.create(
        model=model,
        instructions=agent["instructions"],
        input=json.dumps(input_payload),
        text={
            "format": {
                "type": "json_schema",
                "name": f"{agent['name']}_agent_output",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "agent": {"type": "string"},
                        "summary": {"type": "string"},
                        "observations": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["agent", "summary", "observations"],
                },
                "strict": True,
            }
        },
        max_output_tokens=max_output_tokens,
    )
    return json.loads(response.output_text)


def _extract_agentic_features(
    client,
    model,
    scenario,
    feature_columns,
    agent_outputs,
    context_documents,
    max_output_tokens,
):
    response = client.responses.create(
        model=model,
        instructions=(
            "You are the endpoint inference coordinator for an agentic hospital "
            "workflow. Use the hospital, doctor, and nurse agent outputs to return "
            "the final numeric endpoint features. Return only values that match "
            "the provided JSON schema. Do not invent values unless the scenario or "
            "agent handoff clearly implies a conservative numeric default."
        ),
        input=json.dumps(
            {
                "scenario": scenario,
                "feature_columns": feature_columns,
                "retrieved_context": context_documents,
                "agent_outputs": agent_outputs,
            }
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "agentic_endpoint_features",
                "schema": _feature_schema(feature_columns),
                "strict": True,
            }
        },
        max_output_tokens=max_output_tokens,
    )
    return json.loads(response.output_text)


def build_agentic_payload(
    scenario,
    feature_columns,
    model,
    max_output_tokens,
    context_documents=None,
):
    from openai import OpenAI

    client = OpenAI()
    context_documents = context_documents or []
    agent_outputs = []
    shared_context = {
        "scenario": scenario,
        "feature_columns": feature_columns,
        "retrieved_context": context_documents,
        "prior_agent_outputs": [],
    }

    for agent in AGENT_DEFINITIONS:
        output = _call_agent(client, model, agent, shared_context, max_output_tokens)
        output["agent"] = agent["name"]
        output["role"] = agent["role"]
        agent_outputs.append(output)
        shared_context["prior_agent_outputs"] = agent_outputs

    features = _extract_agentic_features(
        client,
        model,
        scenario,
        feature_columns,
        agent_outputs,
        context_documents,
        max_output_tokens,
    )

    return {
        "agents": agent_outputs,
        "context_documents": context_documents,
        "features": features,
        "payload": build_payload(feature_columns, features),
    }


def build_agentic_rag_payload(
    scenario,
    knowledge_base_path,
    feature_columns,
    model,
    max_output_tokens,
    top_k,
):
    context_documents = retrieve_context(scenario, knowledge_base_path, top_k)
    return build_agentic_payload(
        scenario,
        feature_columns,
        model,
        max_output_tokens,
        context_documents=context_documents,
    )


def explain_agentic_prediction(scenario, agent_outputs, features, prediction, model, max_output_tokens):
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=(
            "Explain the real-time endpoint result for a hospital operations user. "
            "Reference the hospital, doctor, and nurse agent handoff only at a high "
            "level. Be concise and avoid clinical certainty."
        ),
        input=json.dumps(
            {
                "scenario": scenario,
                "agent_outputs": agent_outputs,
                "features": features,
                "prediction": prediction,
            }
        ),
        max_output_tokens=max_output_tokens,
    )
    return response.output_text
