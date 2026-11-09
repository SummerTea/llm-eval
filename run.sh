#!/bin/bash
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-${LLM_EVAL_MODEL:-}}"
if [ -z "${MODEL}" ]; then
    echo "set MODEL or LLM_EVAL_MODEL" >&2
    exit 1
fi

if [ -z "${TEST_SETS:-}" ]; then
    echo 'set TEST_SETS="aime:data/aime_2026.jsonl gpqa:data/gpqa_diamond_r0.1.jsonl"' >&2
    exit 1
fi

TASK="${TASK:-${MODEL//\//_}-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${EVAL_DIR}/outputs/${TASK}"

N_SAMPLE="${N_SAMPLE:-}"
TEMPERATURE="${TEMPERATURE:-}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
MAX_TOKENS="${MAX_TOKENS:-}"
LIMIT="${LIMIT:-}"
CONCURRENCY="${CONCURRENCY:-16}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-16}"
TEXT_ONLY="${TEXT_ONLY:-0}"
CHECK="${CHECK:-}"

TS_ARGS=()
for spec in ${TEST_SETS}; do
    TS_ARGS+=(--test-set "${spec}")
done

ARGS=(--task "${TASK}" --model "${MODEL}" --concurrency "${CONCURRENCY}" "${TS_ARGS[@]}")
[ -n "${N_SAMPLE}" ] && ARGS+=(--n-sample "${N_SAMPLE}")
[ -n "${TEMPERATURE}" ] && ARGS+=(--temperature "${TEMPERATURE}")
[ -n "${REASONING_EFFORT}" ] && ARGS+=(--reasoning-effort "${REASONING_EFFORT}")
[ -n "${MAX_TOKENS}" ] && ARGS+=(--max-tokens "${MAX_TOKENS}")
[ -n "${LIMIT}" ] && ARGS+=(--limit "${LIMIT}")
[ "${TEXT_ONLY}" = "1" ] && ARGS+=(--text-only)

echo "[task] task=${TASK} model=${MODEL} test_sets=${TEST_SETS} check=${CHECK:-auto} -> ${OUT}"

# 1) inference: all test sets in one task
python "${EVAL_DIR}/run_eval.py" "${ARGS[@]}"

# 2) evaluate: one summary + one report for the whole task
# judge env is only needed when the effective check is llm (HLE always; others only with CHECK=llm)
SCORE_ARGS=(--out-dir "${OUT}" --judge-concurrency "${JUDGE_CONCURRENCY}")
[ -n "${CHECK}" ] && SCORE_ARGS+=(--check "${CHECK}")
python "${EVAL_DIR}/score.py" "${SCORE_ARGS[@]}"

echo "done: ${OUT}/summary.json"
