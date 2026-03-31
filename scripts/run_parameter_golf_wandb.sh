#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export RUN_ID="${RUN_ID:-baseline_sp1024_wandb}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export WANDB_PROJECT="${WANDB_PROJECT:-parameter-golf}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PG_INIT_METHOD="${PG_INIT_METHOD:-default}"

python -m modelCraft.parameter_golf_wandb "$@"
