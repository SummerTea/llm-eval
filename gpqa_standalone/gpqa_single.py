"""Standalone GPQA rule-scoring runner. Keep prompt/extraction in sync with ../gpqa/bench.py."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NoReturn

SAMPLE_SEED = 20260101
DEFAULT_DATA_FILE = Path(__file__).resolve().parent / "gpqa_diamond.jsonl"

GPQA_PROMPT = (
    "Answer the following multiple choice question. The last line of your response should be "
    "of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
    "Think step by step before answering.\n\n{question}"
)

_ANSWER_LINE_RE = re.compile(r"(?im)^\s*ANSWER\s*[:：]\s*\(?([A-D])\b")
_BOXED_RE = re.compile(r"\\boxed\{\s*\(?([A-D])\)?\s*\}")
_STANDALONE_RE = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class RetryableError(Exception):
    """Transient failure: timeout, connection error, or a retryable HTTP status."""


# --------------------------------------------------------------------------- #
# Data loading and deterministic selection
# --------------------------------------------------------------------------- #


def load_questions(path: str | os.PathLike | None = None) -> list[dict[str, Any]]:
    data_file = Path(path) if path is not None else DEFAULT_DATA_FILE
    if not data_file.is_file():
        print(f"[错误] 数据文件不存在: {data_file}", file=sys.stderr)
        print("请用 --data-file PATH 指定 gpqa_diamond.jsonl 的路径。", file=sys.stderr)
        raise SystemExit(1)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with data_file.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[错误] 第 {line_no} 行不是合法 JSON: {exc}", file=sys.stderr)
                raise SystemExit(1) from None
            if not isinstance(raw, dict):
                print(f"[错误] 第 {line_no} 行不是 JSON 对象", file=sys.stderr)
                raise SystemExit(1)

            task_id = str(raw.get("task_id") or "").strip()
            question = str(raw.get("question") or "").strip()
            answer = str(raw.get("answer") or "").strip().upper()
            if not task_id:
                print(f"[错误] 第 {line_no} 行 task_id 为空", file=sys.stderr)
                raise SystemExit(1)
            if not question:
                print(f"[错误] 第 {line_no} 行 question 为空 (task_id={task_id})", file=sys.stderr)
                raise SystemExit(1)
            if answer not in {"A", "B", "C", "D"}:
                print(
                    f"[错误] 第 {line_no} 行 answer 不在 A-D 范围内 (task_id={task_id}, answer={answer!r})",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            if task_id in seen:
                print(f"[错误] 第 {line_no} 行 task_id 重复: {task_id}", file=sys.stderr)
                raise SystemExit(1)

            seen.add(task_id)
            rows.append(
                {
                    "task_id": task_id,
                    "question": question,
                    "answer": answer,
                    "category": str(raw.get("category") or "unknown"),
                }
            )

    if not rows:
        print(f"[错误] 数据文件没有可用题目: {data_file}", file=sys.stderr)
        raise SystemExit(1)
    return rows


def _sample_key(task_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).digest()


def select_deterministic(
    rows: list[dict[str, Any]], n: int, seed: int = SAMPLE_SEED
) -> list[dict[str, Any]]:
    if not 1 <= n <= len(rows):
        raise ValueError(f"sample size {n} out of range 1..{len(rows)}")
    ranked = sorted(rows, key=lambda row: _sample_key(row["task_id"], seed))
    keep = {row["task_id"] for row in ranked[:n]}
    return [row for row in rows if row["task_id"] in keep]


def ids_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(sorted(row["task_id"] for row in rows)).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Prompt / extraction / scoring
# --------------------------------------------------------------------------- #


def build_prompt(question: str) -> str:
    return GPQA_PROMPT.format(question=str(question).strip())


def extract_answer_with_source(text: str) -> tuple[str | None, str]:
    if not text:
        return None, "none"
    matches = _ANSWER_LINE_RE.findall(text)
    if matches:
        return matches[-1].upper(), "answer_line"
    matches = _BOXED_RE.findall(text)
    if matches:
        return matches[-1].upper(), "boxed"
    tail = text.rsplit("</think>", 1)[-1] if "</think>" in text else text[-400:]
    matches = _STANDALONE_RE.findall(tail)
    if matches:
        return matches[-1].upper(), "tail_letter"
    return None, "none"


def is_correct(pred: str | None, gold: str) -> bool:
    return pred is not None and pred.upper() == str(gold).strip().upper()


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def _post_once(
    url: str,
    headers: dict[str, str],
    body: bytes,
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:  # body is best-effort diagnostics only
            detail = ""
        message = f"HTTP {exc.code}: {detail}"
        if exc.code in _RETRYABLE_STATUS:
            raise RetryableError(message) from None
        raise RuntimeError(message) from None
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
        raise RetryableError(f"{type(exc).__name__}: {exc}") from None

    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        detail = payload.decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"invalid JSON response: {exc}; body={detail}") from None


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode()

    attempts = retries + 1
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            return _post_once(url, headers, body, timeout)
        except RetryableError as exc:
            last_error = str(exc)
            if attempt < attempts:
                time.sleep(min(2**attempt, 10))
        except RuntimeError:
            raise
    raise RuntimeError(f"{last_error} (after {attempts} attempts)")


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def run_one(item: tuple[dict[str, Any], int], settings: dict[str, Any]) -> dict[str, Any]:
    row, sample_idx = item
    result: dict[str, Any] = {
        "task_id": row["task_id"],
        "sample_idx": sample_idx,
        "gold": row["answer"],
        "raw_text": "",
        "finish_reason": None,
        "error": None,
    }
    try:
        data = chat_completion(
            base_url=settings["base_url"],
            api_key=settings["api_key"],
            model=settings["model"],
            prompt=build_prompt(row["question"]),
            temperature=settings["temperature"],
            max_tokens=settings["max_tokens"],
            timeout=settings["timeout"],
            retries=settings["retries"],
        )
        choice = data["choices"][0]
        message = choice.get("message") or {}
        text = _as_text(message.get("content")).strip()
        if not text:
            text = _as_text(message.get("reasoning_content")).strip()
        if not text:
            text = _as_text(message.get("reasoning")).strip()
        result["raw_text"] = text
        result["finish_reason"] = choice.get("finish_reason")
    except Exception as exc:  # one failed request must not abort the batch
        result["finish_reason"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"

    pred, source = extract_answer_with_source(result["raw_text"])
    result["pred"] = pred
    result["extract_source"] = source
    result["correct"] = is_correct(pred, row["answer"])
    return result


# --------------------------------------------------------------------------- #
# CLI / report
# --------------------------------------------------------------------------- #


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:  # argparse would exit 2; spec wants 1
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(description="Standalone GPQA Diamond rule-scoring runner")
    parser.add_argument("--base-url", default=os.environ.get("LLM_EVAL_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("LLM_EVAL_API_KEY"))
    parser.add_argument("--model", default=os.environ.get("LLM_EVAL_MODEL"))
    parser.add_argument("--data-file", default=None)

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample", type=int, default=None)
    selection.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)

    parser.add_argument("--n-sample", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--min-pass1", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _fail(message: str) -> None:
    print(f"[错误] {message}", file=sys.stderr)
    raise SystemExit(1)


def _validate_args(args: argparse.Namespace, total: int) -> None:
    if not args.base_url:
        _fail("缺少 base_url（--base-url 或 LLM_EVAL_BASE_URL）")
    if not args.api_key:
        _fail("缺少 api_key（--api-key 或 LLM_EVAL_API_KEY）")
    if not args.model:
        _fail("缺少 model（--model 或 LLM_EVAL_MODEL）")
    if args.sample is not None and not 1 <= args.sample <= total:
        _fail(f"--sample 必须在 1..{total} 之间，收到 {args.sample}")
    if args.limit is not None and not 1 <= args.limit <= total:
        _fail(f"--limit 必须在 1..{total} 之间，收到 {args.limit}")
    if args.n_sample < 1:
        _fail(f"--n-sample 必须 >= 1，收到 {args.n_sample}")
    if args.concurrency < 1:
        _fail(f"--concurrency 必须 >= 1，收到 {args.concurrency}")
    if args.timeout <= 0:
        _fail(f"--timeout 必须 > 0，收到 {args.timeout}")
    if args.retries < 0:
        _fail(f"--retries 必须 >= 0，收到 {args.retries}")
    if args.max_tokens < 1:
        _fail(f"--max-tokens 必须 >= 1，收到 {args.max_tokens}")
    if args.temperature < 0:
        _fail(f"--temperature 必须 >= 0，收到 {args.temperature}")
    if args.min_pass1 is not None and not 0 <= args.min_pass1 <= 100:
        _fail(f"--min-pass1 必须在 0..100 之间，收到 {args.min_pass1}")


def _print_report(
    *,
    args: argparse.Namespace,
    data_name: str,
    total: int,
    set_label: str,
    selected_count: int,
    results: list[dict[str, Any]],
    elapsed: float,
    pass1_percent: float,
    coverage_complete: bool,
) -> None:
    expected = len(results)
    successful = sum(1 for item in results if item["error"] is None)
    failed_requests = expected - successful
    correct = sum(1 for item in results if item["correct"])

    finish_counts: dict[str, int] = {}
    for item in results:
        reason = item["finish_reason"] or "unknown"
        finish_counts[reason] = finish_counts.get(reason, 0) + 1
    reason_order = ["stop", "length", "error"] + sorted(
        key for key in finish_counts if key not in {"stop", "length", "error"}
    )
    reason_text = ", ".join(f"{key}={finish_counts.get(key, 0)}" for key in reason_order)

    source_counts = {"answer_line": 0, "boxed": 0, "tail_letter": 0, "none": 0}
    for item in results:
        if item["error"] is None:
            source_counts[item["extract_source"]] += 1
    extraction_failed = source_counts["none"]
    extraction_failed_pct = (extraction_failed / successful * 100) if successful else 0.0

    print("================ GPQA Diamond 质量评测 ================")
    print(f"模型: {args.model}        端点: {args.base_url} (key=***)")
    print(f"数据: {data_name} ({total} 题)")
    print(f"题集: {set_label} ({selected_count}/{total})")
    print(
        f"采样: n_sample={args.n_sample} | temperature={args.temperature} | "
        f"max_tokens={args.max_tokens}"
    )
    print(f"并发: {args.concurrency}  超时: {args.timeout:g}s  耗时: {elapsed:.1f}s")
    print("------------------------------------------------------")
    print(f"请求: {successful}/{expected} 成功   调用失败: {failed_requests}")
    print(f"finish_reason: {reason_text}")
    print(
        "抽取来源: "
        f"answer_line={source_counts['answer_line']}, boxed={source_counts['boxed']}, "
        f"tail_letter={source_counts['tail_letter']}, none={source_counts['none']}"
    )
    coverage_text = "是" if coverage_complete else "否"
    if successful:
        print(f"抽取失败: {extraction_failed_pct:.2f}%   覆盖完整: {coverage_text}")
    else:
        print(f"抽取失败: n/a   覆盖完整: {coverage_text}")
    print("------------------------------------------------------")
    print(f"Pass@1: {pass1_percent:.2f}%  ({correct}/{expected})")
    print("------------------------------------------------------")

    failures = [item for item in results if not item["correct"]]
    if not failures:
        print("全部通过")
    else:
        limit = len(failures) if args.verbose else min(20, len(failures))
        if args.verbose:
            print(f"未通过（全部 {len(failures)} 条）:")
        else:
            print("未通过（前 20 条）:")
        for item in failures[:limit]:
            pred = item["pred"] if item["pred"] is not None else "None"
            if item["error"] is not None:
                source = f"error: {item['error'][:80]}"
            else:
                source = item["extract_source"]
            line = (
                f"  {item['task_id']}  gold={item['gold']}  pred={pred}  "
                f"finish={item['finish_reason'] or 'unknown'}  ({source})"
            )
            if args.n_sample > 1:
                line += f"  sample={item['sample_idx']}"
            print(line)
    print("======================================================")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_questions(args.data_file)
    total = len(rows)
    _validate_args(args, total)

    if args.sample is not None:
        selected = select_deterministic(rows, args.sample, args.sample_seed)
        set_label = (
            f"sample={args.sample} seed={args.sample_seed} "
            f"ids_sha256={ids_sha256(selected)[:12]}"
        )
    elif args.limit is not None:
        selected = rows[: args.limit]
        set_label = f"limit={args.limit} ids_sha256={ids_sha256(selected)[:12]}"
    else:
        selected = rows
        set_label = f"all ids_sha256={ids_sha256(selected)[:12]}"

    settings = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "retries": args.retries,
    }
    plan = [(row, sample_idx) for row in selected for sample_idx in range(args.n_sample)]
    expected = len(plan)
    print(
        f"[gpqa] requests={expected} model={args.model} data={Path(args.data_file).name if args.data_file else DEFAULT_DATA_FILE.name}",
        flush=True,
    )

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda item: run_one(item, settings), plan))
    elapsed = time.monotonic() - started

    successful = sum(1 for item in results if item["error"] is None)
    coverage_complete = successful == expected
    correct = sum(1 for item in results if item["correct"])
    pass1_percent = (correct / expected * 100) if expected else 0.0

    _print_report(
        args=args,
        data_name=Path(args.data_file).name if args.data_file else DEFAULT_DATA_FILE.name,
        total=total,
        set_label=set_label,
        selected_count=len(selected),
        results=results,
        elapsed=elapsed,
        pass1_percent=pass1_percent,
        coverage_complete=coverage_complete,
    )

    if not coverage_complete:
        return 2
    if args.min_pass1 is not None and round(pass1_percent, 2) < args.min_pass1:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
