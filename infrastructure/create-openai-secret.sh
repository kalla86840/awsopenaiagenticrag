#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
  echo "Usage: ./infrastructure/create-openai-secret.sh sk-your-openai-api-key"
  exit 1
fi

aws secretsmanager create-secret \
  --region "${AWS_REGION:-us-west-1}" \
  --name "${OPENAI_SECRET_NAME:-openai/api-key}" \
  --secret-string "$1"
