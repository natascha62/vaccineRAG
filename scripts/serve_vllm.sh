#!/usr/bin/env bash
# Serve an open-source model with vLLM's OpenAI-compatible API (Linux + NVIDIA GPU).
#
# Usage (in the `vaxrag-vllm` conda environment):
#   bash scripts/serve_vllm.sh google/gemma-3-27b-it
#   bash scripts/serve_vllm.sh google/medgemma-27b-text-it
#
# The research plan requires unquantized bf16 weights and identical settings for
# Gemma and MedGemma. A 27B model in bf16 needs ~55 GB of GPU memory, so serve one
# model at a time on an 80 GB GPU. Record the printed vLLM version in the methods.
set -euo pipefail

MODEL="${1:?usage: serve_vllm.sh <hf-model-id> [port]}"
PORT="${2:-8000}"
MAX_LEN="${MAX_MODEL_LEN:-32768}"

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

python -c "import vllm; print('vLLM version:', vllm.__version__)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

exec vllm serve "${MODEL}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_LEN}" \
  --port "${PORT}" \
  --seed 0 \
  --gpu-memory-utilization 0.90 \
  ${VLLM_API_KEY:+--api-key "${VLLM_API_KEY}"}
