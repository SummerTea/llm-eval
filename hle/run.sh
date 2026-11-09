#!/bin/bash
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${MODEL:-${LLM_EVAL_MODEL:-}}"
if [ -z "${MODEL}" ]; then
    echo "set MODEL or LLM_EVAL_MODEL" >&2
    exit 1
fi

N_SAMPLE="${N_SAMPLE:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
RATIO="${RATIO:-}"
SUFFIX="${RATIO:+_r${RATIO}}"
TAG="${TAG:-${MODEL//\//_}-n${N_SAMPLE}-r${RATIO:-full}-$(date -u +%Y%m%dT%H%M%SZ)}"
DATA_FILE="${DATA_FILE:-${EVAL_DIR}/data/hle_test${SUFFIX}.jsonl}"
OUT="${EVAL_DIR}/outputs/hle-${TAG}"
TEXT_ONLY="${TEXT_ONLY:-0}"
CHECK="${CHECK:-}"

ARGS=(--bench hle --data-file "${DATA_FILE}" --model "${MODEL}" --n-sample "${N_SAMPLE}" --temperature "${TEMPERATURE}" --tag "${TAG}")
[ -n "${REASONING_EFFORT}" ] && ARGS+=(--reasoning-effort "${REASONING_EFFORT}")
[ "${TEXT_ONLY}" = "1" ] && ARGS+=(--text-only)

echo "[hle] model=${MODEL} n_sample=${N_SAMPLE} ratio=${RATIO:-full} check=${CHECK:-auto} text_only=${TEXT_ONLY} -> ${OUT}"

# 1) inference (needs LLM_EVAL_BASE_URL / LLM_EVAL_API_KEY / LLM_EVAL_MODEL, or pass --base-url/--api-key/--model)
python "${EVAL_DIR}/run_eval.py" "${ARGS[@]}"

# 2) evaluate (judge env only needed when the effective check is llm: HLE always; others only with CHECK=llm)
# needs EVAL_JUDGE_BASE_URL / EVAL_JUDGE_API_KEY / EVAL_JUDGE_MODEL
SCORE_ARGS=(--out-dir "${OUT}" --judge-concurrency 16)
[ -n "${CHECK}" ] && SCORE_ARGS+=(--check "${CHECK}")
python "${EVAL_DIR}/score.py" "${SCORE_ARGS[@]}"

echo "done: ${OUT}/summary.json"
