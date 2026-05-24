import json


FEATURE_DEFINITIONS = {
    "age": "Customer age in years.",
    "gender": "Numeric gender encoding expected by the trained model.",
    "miles": "Vehicle mileage value expected by the trained model.",
    "debt": "Customer debt value expected by the trained model.",
    "income": "Customer income value expected by the trained model.",
}


def build_payload(feature_columns, values):
    missing = [column for column in feature_columns if column not in values]
    if missing:
        raise ValueError(f"Missing feature values for: {', '.join(missing)}")

    return {
        "instances": [
            [float(values[column]) for column in feature_columns]
        ]
    }


def extract_features_from_text(description, feature_columns, model, max_output_tokens):
    from openai import OpenAI

    schema = {
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

    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=(
            "Extract the numeric model features from the user's scenario. "
            "Return only values that match the provided JSON schema. "
            "Use conservative numeric defaults only when the scenario implies a value."
        ),
        input=description,
        text={
            "format": {
                "type": "json_schema",
                "name": "car_sales_features",
                "schema": schema,
                "strict": True,
            }
        },
        max_output_tokens=max_output_tokens,
    )
    return json.loads(response.output_text)


def explain_prediction(description, features, prediction, model, max_output_tokens):
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=(
            "Explain this realtime model prediction for an operations user. "
            "Be concise, avoid unsupported certainty, and mention the input features "
            "that likely drove the result."
        ),
        input=json.dumps(
            {
                "scenario": description,
                "features": features,
                "prediction": prediction,
            }
        ),
        max_output_tokens=max_output_tokens,
    )
    return response.output_text
