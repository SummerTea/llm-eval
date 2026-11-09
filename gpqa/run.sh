#!/bin/bash
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${MODEL:-${LLM_EVAL_MODEL:-}}"
if [ -z "${MODEL}" ]; then
    echo "set MODEL or LLM_EVAL_MODEL" >&2
    exit 1
fi

N_SAMPLE="${N_SAMPLE:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
RATIO="${RATIO:-}"
SUFFIX="${RATIO:+_r${RATIO}}"
DATA_FILE="${DATA_FILE:-${EVAL_DIR}/data/gpqa_diamond${SUFFIX}.jsonl}"
TASK="${TASK:-${MODEL//\//_}-n${N_SAMPLE}-r${RATIO:-full}-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${EVAL_DIR}/outputs/${TASK}"
TEXT_ONLY="${TEXT_ONLY:-0}"
CHECK="${CHECK:-}"
MAX_TOKENS="${MAX_TOKENS:-}"
LIMIT="${LIMIT:-}"
CONCURRENCY="${CONCURRENCY:-16}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-16}"

ARGS=(--task "${TASK}" --test-set "gpqa:${DATA_FILE}" --model "${MODEL}" \
    --n-sample "${N_SAMPLE}" --temperature "${TEMPERATURE}" --concurrency "${CONCURRENCY}")
[ -n "${REASONING_EFFORT}" ] && ARGS+=(--reasoning-effort "${REASONING_EFFORT}")
[ -n "${MAX_TOKENS}" ] && ARGS+=(--max-tokens "${MAX_TOKENS}")
[ -n "${LIMIT}" ] && ARGS+=(--limit "${LIMIT}")
[ "${TEXT_ONLY}" = "1" ] && ARGS+=(--text-only)

echo "[gpqa] task=${TASK} model=${MODEL} n_sample=${N_SAMPLE} ratio=${RATIO:-full} check=${CHECK:-auto} -> ${OUT}"

python "${EVAL_DIR}/run_eval.py" "${ARGS[@]}"

SCORE_ARGS=(--out-dir "${OUT}" --judge-concurrency "${JUDGE_CONCURRENCY}")
[ -n "${CHECK}" ] && SCORE_ARGS+=(--check "${CHECK}")
python "${EVAL_DIR}/score.py" "${SCORE_ARGS[@]}"

echo "done: ${OUT}/summary.json"
