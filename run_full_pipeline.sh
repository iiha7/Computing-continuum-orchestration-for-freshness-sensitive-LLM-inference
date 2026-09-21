#!/usr/bin/env bash
set -euo pipefail

# Final-paper profiling path:
#   Edge  = compact local LLM through Ollama (default qwen2.5:0.5b)
#   Cloud = larger local LLM through Ollama (default qwen2.5:7b)
#
# Override any of these from the shell if needed, e.g.:
#   EDGE_MODEL=llama3.2:1b CLOUD_MODEL=qwen2.5:14b bash run_real_profile.sh

export EDGE_PROVIDER="${EDGE_PROVIDER:-ollama}"
export EDGE_MODEL="${EDGE_MODEL:-qwen2.5:0.5b}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export CLOUD_PROVIDER="${CLOUD_PROVIDER:-ollama}"
export CLOUD_MODEL="${CLOUD_MODEL:-qwen2.5:7b}"

python -m src.data_prepare --raw-csv data/raw/algerian_forest_fires.csv
python -m src.prompt_builder --window-lengths 1,3,6 --few-shot 3

# Pull the local models manually before running, if needed:
#   ollama pull qwen2.5:0.5b
#   ollama pull qwen2.5:7b
python -m src.profile_llms_real \
  --edge-provider "$EDGE_PROVIDER" \
  --edge-model "$EDGE_MODEL" \
  --edge-max-tokens 128 \
  --cloud-provider "$CLOUD_PROVIDER" \
  --cloud-model "$CLOUD_MODEL" \
  --cloud-max-tokens 256

python -m src.prepare_trace_datasets \
  --access-raw data/raw/loed_full.csv \
  --cloud-rtt-raw data/raw/cloud_edge_rtt_full.csv \
  --cloud-label Milan-MIL01.csv \
  --cloud-mode fixed \
  --gateway-strategy largest \
  --max-attempts 3 \
  --retransmission-backoff-s 1.0
python -m src.calibrate_freshness
python -m src.fit_models
python -m src.calibrate_network_regimes
python -m src.calibrate_normalization

python -m src.train_drl --seeds 1 --steps 5000 --risk-mask
python -m src.evaluate --policies all --seeds 1 --episodes 50 --steps-per-episode 50
python -m src.analyze_results
