#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
daemon_path=${VLLM_APPLE_DAEMON_PATH:-/Users/Shared/Program/python310/.venv/bin/vllm-appled}
backend_path=${VLLM_APPLE_BACKEND_PATH:-/Users/Shared/Program/python310/.venv/bin/vllm-apple-mlx-server}
model_path=${VLLM_APPLE_E2E_MODEL:-${repository_root}/models/gemma-2-2b-it-4bit}

for executable in "${daemon_path}" "${backend_path}"; do
    if [ ! -f "${executable}" ] || [ ! -x "${executable}" ]; then
        echo "required executable is unavailable: ${executable}" >&2
        exit 2
    fi
done
if [ ! -d "${model_path}" ]; then
    echo "E2E model directory is unavailable: ${model_path}" >&2
    exit 2
fi

exec swift run --package-path "${repository_root}/sdk/swift" VLLMAppleModelE2E \
    --daemon "${daemon_path}" \
    --backend "${backend_path}" \
    --backend-kind mlx_lm \
    --model "${model_path}" \
    --request-model default_model \
    --timeout "${VLLM_APPLE_E2E_TIMEOUT:-600}"
