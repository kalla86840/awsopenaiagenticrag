# Local Commands

Common commands for this project.

## Install

```bash
pip install -r requirements.txt
```

## Syntax Check

```bash
python -m compileall src pipelines scripts
```

## Dry-Run SageMaker Pipeline Definition

```bash
python pipelines/run_pipeline.py --config config/default.yaml --dry-run
```

## Run SageMaker Pipeline

```bash
python pipelines/run_pipeline.py --config config/default.yaml --wait
```

## Approve Latest Pending Model Package

```bash
python scripts/approve_model.py --config config/default.yaml
```

## Deploy Endpoint

```bash
python src/deploy.py --config config/default.yaml --environment staging
```

## Smoke-Test Endpoint

```bash
python scripts/test_endpoint.py --config config/default.yaml --environment staging
```

## Invoke Endpoint With Numeric Features

```bash
python scripts/invoke_endpoint.py \
  --config config/default.yaml \
  --environment staging \
  --age 28 \
  --gender 0 \
  --miles 23 \
  --debt 0 \
  --income 4099
```

## Invoke Endpoint With Prompt Ops

```bash
python scripts/prompt_ops_invoke_endpoint.py \
  --config config/default.yaml \
  --environment staging \
  --description "A 28 year old customer encoded as gender 0 has a vehicle with 23 miles, no debt, and income of 4099."
```
