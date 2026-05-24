import json
import os

import joblib
import numpy as np


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, request_content_type):
    content_type = request_content_type.split(";")[0].strip().lower()
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    payload = json.loads(request_body)
    instances = payload.get("instances")
    if instances is None:
        raise ValueError("Request JSON must include an 'instances' field.")

    return np.array(instances)


def predict_fn(input_data, model):
    return model.predict(input_data)


def output_fn(prediction, response_content_type):
    body = json.dumps({"predictions": prediction.tolist()})
    return body, "application/json"
