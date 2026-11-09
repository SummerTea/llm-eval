from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from common.io import read_jsonl, write_jsonl  # noqa: E402
from common.loader import load_bench  # noqa: E402
from common.passk import default_k_list, print_report, summarize  # noqa: E402
from common.report import write_report  # noqa: E402

# Default judge per benchmark: rule judging is the default, except HLE which requires an LLM judge.
BENCH_DEFAULT_JUDGE = {"aime": "rule", "hle": "llm", "gpqa": "rule"}


def load_raw(out_dir: str) -> list[dict]:
    """Merge every rank's raw output; for a given (test_set, task_id, sample_idx) the last write wins."""
    merged: dict[tuple, dict] = {}
    order: list[tuple] = []
    paths = sorted(glob.glob(os.path.join(out_dir, "raw_rank*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"no raw_rank*.jsonl under {out_dir} (did generation finish?)")
    for path in paths:
        for row in read_jsonl(path):
            key = (row.get("test_set", ""), row["task_id"], row["sample_idx"])
            if key not in merged:
                order.append(key)
            merged[key] = row
    return [merged[key] for key in order]


def _normalize_test_sets(meta: dict) -> list[dict[str, Any]]:
    test_sets = meta.get("test_sets")
    if isinstance(test_sets, list) and test_sets:
        return test_sets
    if meta.get("bench"):
        # Legacy single-benchmark meta.
        return [
            {
                "name": meta["bench"],
                "bench": meta["bench"],
                "data_file": meta.get("data_file"),
                "sample_ratio": meta.get("sample_ratio", 1.0),
            }
        ]
    raise SystemExit("meta.json has neither test_sets nor bench; cannot score")


def _judge(results: list[dict], by_id: dict, bench_name: str, judge: str, concurrency: int) -> str | None:
    """Score `results` in place. Returns the judge model name for the LLM judge, else None."""
    if judge == "rule" and bench_name == "hle":
        raise SystemExit(
            "HLE does not support rule judging: the official protocol requires an LLM judge.\n"
            "Most HLE answers are free text, so normalized exact match marks equivalent-but-"
            "differently-written answers wrong and systematically understates the score.\n"
            "Set EVAL_JUDGE_BASE_URL / EVAL_JUDGE_API_KEY / EVAL_JUDGE_MODEL and use --judge llm."
        )
    if judge != "llm":
        for row in results:
            row["rule_correct"] = row["correct"]
        return None

    from common.judge import judge_answers, resolve_model, truncate_output

    judge_kind = "hle" if bench_name == "hle" else "equivalence"

    def judge_input(text: str) -> str:
        # HLE official protocol: drop the thinking section, send the rest in full, no truncation.
        if judge_kind == "hle":
            return text.split("</think>")[-1].strip()
        return truncate_output(text)

    items = [
        (
            by_id[row["task_id"]].get("judge_question") or by_id[row["task_id"]]["question"],
            by_id[row["task_id"]]["answer"],
            judge_input(row["raw_text"]),
        )
        for row in results
    ]
    for row, verdict in zip(results, judge_answers(items, kind=judge_kind, concurrency=concurrency), strict=True):
        row["rule_correct"] = row["correct"]
        row["correct"] = bool(verdict["judge_correct"])
        row["judge_raw"] = verdict.get("judge_raw", "")
        row["judge_parsed"] = bool(verdict.get("judge_parsed", True))
        if "judge_verdict_unparsed" in verdict:
            row["judge_verdict_unparsed"] = bool(verdict["judge_verdict_unparsed"])
        if verdict.get("judge_error"):
            row["judge_error"] = verdict["judge_error"]
    return resolve_model()


def _score_test_set(
    test_set: dict[str, Any],
    meta: dict,
    rollouts: list[dict],
    *,
    judge: str,
    judge_concurrency: int,
    data_file_override: str | None,
    legacy_selected: list[str] | None,
) -> tuple[dict[str, Any], list[dict], str, str | None]:
    name = test_set["name"]
    bench_name = test_set["bench"]
    bench = load_bench(bench_name)
    data_path = data_file_override or test_set.get("data_file")
    if not data_path:
        raise SystemExit(f"test set {name!r} has no data_file; pass --data-file explicitly for single-set runs")

    problems = bench.load(data_path)
    records = bench.build_records(problems, limit=meta.get("limit"))
    if meta.get("text_only"):
        records = [record for record in records if not record.get("image_paths")]
    if legacy_selected is not None:
        keep = set(legacy_selected)
        records = [record for record in records if record["task_id"] in keep]
    by_id = {record["task_id"]: record for record in records}

    set_rollouts = [row for row in rollouts if str(row.get("test_set", "")) == name]
    unknown = [row["task_id"] for row in set_rollouts if row["task_id"] not in by_id]
    assert not unknown, f"[{name}] raw output has task_ids that are not in the dataset: {sorted(set(unknown))[:5]}"

    results = [bench.score_rollout(row, by_id[row["task_id"]]) for row in set_rollouts]
    for row in results:
        row["test_set"] = name

    effective_judge = judge if judge != "auto" else BENCH_DEFAULT_JUDGE.get(bench_name, "rule")
    judge_model = _judge(results, by_id, bench_name, effective_judge, judge_concurrency)

    if not results:
        raise SystemExit(f"no rollout for test set {name!r} in raw_rank*.jsonl; refusing to emit an empty score")

    seen = {row["task_id"] for row in results}
    n_sample = int(meta.get("n_sample", max((row["sample_idx"] for row in results), default=0) + 1))
    missing = sorted(set(by_id) - seen)
    expected_rollouts = len(records) * n_sample
    coverage_complete = not missing and len(results) == expected_rollouts
    if not coverage_complete:
        print(
            f"[warn] {name}: incomplete coverage: {len(seen)}/{len(records)} tasks, "
            f"{len(results)}/{expected_rollouts} rollouts; this is not a complete result",
            flush=True,
        )

    finish_reasons = dict(Counter(str(row["finish_reason"]) for row in results if row.get("finish_reason")))
    extract_sources = dict(Counter(str(row["extract_source"]) for row in results if row.get("extract_source")))
    truncated = sum(finish_reasons.get(reason, 0) for reason in ("length", "prompt_length_exceeded"))
    invalid = sum(finish_reasons.get(reason, 0) for reason in ("ignored", "aborted", "error", "preempted"))

    summary = summarize(
        results,
        n_sample=n_sample,
        k_list=default_k_list(n_sample),
        extra={
            "test_set": name,
            "bench": bench_name,
            "model_name": meta.get("model_name"),
            "judge": effective_judge,
            "judge_model": judge_model,
            "temperature": meta.get("temperature"),
            "top_p": meta.get("top_p"),
            "reasoning_effort": meta.get("reasoning_effort"),
            "max_tokens": meta.get("max_tokens"),
            "extract_failed": sum(1 for row in results if row.get("extract_failed", row.get("pred") is None)) / len(results),
            "expected_tasks": len(records),
            "expected_rollouts": expected_rollouts,
            "coverage_complete": coverage_complete,
            "missing_tasks": len(missing),
            "ragged_samples": len({sum(1 for row in results if row["task_id"] == task_id) for task_id in seen}) > 1,
            "finish_reasons": finish_reasons,
            "extract_sources": extract_sources,
            "truncated_rollouts": truncated,
            "invalid_rollouts": invalid,
            "sample_ratio": test_set.get("sample_ratio", meta.get("sample_ratio")),
            "selected_task_count": len(records),
        },
    )
    groups: dict[str, list[dict]] = {}
    for row in results:
        if key := str(by_id[row["task_id"]].get("modality") or ""):
            groups.setdefault(key, []).append(row)
    if len(groups) > 1:
        summary["by_modality"] = {
            modality: {
                "num_tasks": len({row["task_id"] for row in rows}),
                "num_rollouts": len(rows),
                "Pass@1": sum(1 for row in rows if row["correct"]) / len(rows),
            }
            for modality, rows in sorted(groups.items())
        }

    if effective_judge == "llm":
        agree = sum(1 for row in results if bool(row["rule_correct"]) == bool(row["correct"]))
        summary["rule_vs_llm_agreement"] = agree / len(results)
        unparsed = sum(1 for row in results if not row.get("judge_parsed", True))
        summary["judge_unparsed"] = unparsed / len(results)
        verdict_unparsed = sum(1 for row in results if row.get("judge_verdict_unparsed"))
        summary["judge_verdict_unparsed"] = verdict_unparsed / len(results)
        if verdict_unparsed:
            print(
                f"[warn] {name}: {verdict_unparsed}/{len(results)} judge replies had no extractable verdict; "
                "all three fallbacks missed, they count as wrong",
                flush=True,
            )
        if unparsed:
            print(
                f"[warn] {name}: {unparsed}/{len(results)} judge calls returned no verdict and count as wrong; "
                "this depresses the score -- raise the judge max_tokens and re-run",
                flush=True,
            )
    return summary, results, effective_judge, judge_model


def _overall(per_set: dict[str, dict], results: list[dict]) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for row in results if row.get("correct"))
    finish_reasons = dict(Counter(str(row["finish_reason"]) for row in results if row.get("finish_reason")))
    extract_sources = dict(Counter(str(row["extract_source"]) for row in results if row.get("extract_source")))

    def _sum(key: str) -> int:
        return sum(int(summary.get(key, 0)) for summary in per_set.values())

    return {
        "num_test_sets": len(per_set),
        "num_tasks": sum(summary.get("num_tasks", 0) for summary in per_set.values()),
        "num_rollouts": total,
        "Pass@1": correct / total if total else 0.0,
        "score": correct / total if total else 0.0,
        "coverage_complete": all(summary.get("coverage_complete") for summary in per_set.values()),
        "expected_tasks": _sum("expected_tasks"),
        "expected_rollouts": _sum("expected_rollouts"),
        "missing_tasks": _sum("missing_tasks"),
        "truncated_rollouts": _sum("truncated_rollouts"),
        "invalid_rollouts": _sum("invalid_rollouts"),
        "extract_failed": sum(1 for row in results if row.get("extract_failed", row.get("pred") is None)) / total if total else 0.0,
        "finish_reasons": finish_reasons,
        "extract_sources": extract_sources,
    }


def score_run(
    out_dir: str,
    *,
    judge: str = "auto",
    judge_concurrency: int = 8,
    data_file: str | None = None,
    verbose: bool = True,
) -> int:
    meta_path = os.path.join(out_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise SystemExit(f"missing {meta_path}; run run_eval.py first (score.py scores a generation run)")
    with open(meta_path) as handle:
        meta = json.load(handle)

    test_sets = _normalize_test_sets(meta)
    if data_file and len(test_sets) != 1:
        raise SystemExit("--data-file can only override a single-test-set task")

    rollouts = load_raw(out_dir)
    if rollouts and not any("test_set" in row for row in rollouts) and len(test_sets) == 1:
        # Legacy output: all rows belong to the single test set.
        for row in rollouts:
            row["test_set"] = test_sets[0]["name"]

    legacy_selected = meta.get("selected_task_ids")
    per_set: dict[str, dict] = {}
    all_results: list[dict] = []
    for test_set in test_sets:
        summary, results, effective_judge, _ = _score_test_set(
            test_set,
            meta,
            rollouts,
            judge=judge,
            judge_concurrency=judge_concurrency,
            data_file_override=data_file if len(test_sets) == 1 else None,
            legacy_selected=legacy_selected if len(test_sets) == 1 else None,
        )
        per_set[test_set["name"]] = summary
        all_results.extend(results)

    if not all_results:
        raise SystemExit(f"no rollout at all in {out_dir}/raw_rank*.jsonl; refusing to emit an empty score")

    summary = {
        "task": meta.get("task") or meta.get("tag") or os.path.basename(os.path.normpath(out_dir)),
        "model_name": meta.get("model_name"),
        "judge": judge,
        "num_test_sets": len(per_set),
        "n_sample": meta.get("n_sample"),
        "temperature": meta.get("temperature"),
        "top_p": meta.get("top_p"),
        "max_tokens": meta.get("max_tokens"),
        "reasoning_effort": meta.get("reasoning_effort"),
        "test_sets": per_set,
        "overall": _overall(per_set, all_results),
    }

    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, "scored.jsonl"), all_results)
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    try:
        report_path = write_report(out_dir, summary, all_results)
        if verbose:
            print(f"wrote {report_path}")
    except Exception as exc:  # report must never break scoring
        print(f"[warn] failed to write report.html: {type(exc).__name__}: {exc}", flush=True)

    if verbose:
        for name, set_summary in per_set.items():
            print_report(set_summary, title=f"{name}  {set_summary.get('model_name')}  judge={set_summary.get('judge')}")
            print(
                f"  coverage: expected {set_summary['expected_rollouts']} rollouts / "
                f"{set_summary['expected_tasks']} tasks, got {set_summary['num_rollouts']} / "
                f"{set_summary['num_tasks']}, complete={set_summary['coverage_complete']}"
            )
            print(f"  finish_reason={set_summary['finish_reasons']}")
            print(f"  extract_failed={set_summary['extract_failed'] * 100:.2f}%")
        overall = summary["overall"]
        print(
            f"overall: test_sets={overall['num_test_sets']} tasks={overall['num_tasks']} "
            f"rollouts={overall['num_rollouts']} Pass@1={overall['Pass@1'] * 100:.2f}% "
            f"complete={overall['coverage_complete']}"
        )
        print(f"wrote {os.path.join(out_dir, 'scored.jsonl')} / summary.json")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="score one task (one or more test sets) and summarize Pass@k")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--judge", choices=["auto", "rule", "llm"], default="auto", help="auto: rule by default, except HLE which uses llm"
    )
    parser.add_argument("--judge-concurrency", type=int, default=8)
    parser.add_argument(
        "--check", choices=["rule", "llm"], default=None, help="override judge mode; default: rule, except HLE which uses llm"
    )
    parser.add_argument("--data-file", default=None, help="override meta.json's data_file (single test set only)")
    args = parser.parse_args()
    judge = args.check or args.judge
    return score_run(
        args.out_dir,
        judge=judge,
        judge_concurrency=args.judge_concurrency,
        data_file=args.data_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())
