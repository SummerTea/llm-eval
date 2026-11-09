from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from common.client import chat_completion  # noqa: E402
from common.io import write_jsonl  # noqa: E402
from common.loader import SUPPORTED, load_bench  # noqa: E402
from common.paths import resolve_out_dir  # noqa: E402

BENCH_DEFAULTS = {
    "aime": {"n_sample": 32, "temperature": 1.0, "top_p": 1.0},
    "hle": {"n_sample": 1, "temperature": 1.0, "top_p": 1.0},
    "gpqa": {"n_sample": 8, "temperature": 1.0, "top_p": 1.0},
}


def _read_data_manifest(data_file: str) -> dict[str, Any]:
    path = Path(str(data_file) + ".meta.json")
    if not path.is_file():
        return {}
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


_IMAGE_MAGIC = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG", "image/png"),
    (b"GIF8", "image/gif"),
    (b"BM", "image/bmp"),
)


def _data_uri(path: str) -> str:
    data = Path(path).read_bytes()
    mime = next((m for magic, m in _IMAGE_MAGIC if data.startswith(magic)), None)
    if mime is None and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime = "image/webp"
    return f"data:{mime or 'image/png'};base64,{base64.b64encode(data).decode()}"


def _content_parts(text: str, image_paths: list[str]) -> list[dict[str, Any]]:
    """Split the question on the <image> placeholder and interleave OpenAI vision parts."""
    parts: list[dict[str, Any]] = []
    segments = text.split("<image>")
    for index, segment in enumerate(segments):
        if segment.strip():
            parts.append({"type": "text", "text": segment})
        if index < len(image_paths):
            parts.append({"type": "image_url", "image_url": {"url": _data_uri(image_paths[index])}})
    for extra in image_paths[len(segments) :]:
        parts.append({"type": "image_url", "image_url": {"url": _data_uri(extra)}})
    return parts or [{"type": "text", "text": text}]


def _build_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = [dict(message) for message in record["messages"]]
    image_paths = record.get("image_paths") or []
    if not image_paths:
        return messages
    for message in messages:
        content = message.get("content") or ""
        if isinstance(content, str) and "<image>" in content:
            message["content"] = _content_parts(content, image_paths)
            return messages
    # No placeholder: attach the images to the first user message.
    for message in messages:
        if message.get("role") == "user":
            message["content"] = _content_parts(message.get("content") or "", image_paths)
            break
    return messages


def _parse_test_sets(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[dict[str, Any]]:
    specs = list(args.test_set or [])
    if not specs:
        if args.bench and args.data_file:
            return [{"name": args.bench, "bench": args.bench, "data_file": args.data_file}]
        parser.error("provide --test-set BENCH:FILE (repeatable) or --bench + --data-file")

    test_sets: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for spec in specs:
        if ":" not in spec:
            parser.error(f"--test-set expects BENCH:FILE, got {spec!r}")
        bench_name, data_file = spec.split(":", 1)
        if bench_name not in SUPPORTED:
            parser.error(f"unknown bench {bench_name!r}; expected one of {', '.join(SUPPORTED)}")
        count = seen.get(bench_name, 0) + 1
        seen[bench_name] = count
        test_sets.append(
            {"name": bench_name if count == 1 else f"{bench_name}_{count}", "bench": bench_name, "data_file": data_file}
        )
    return test_sets


def _load_test_sets(test_sets: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for test_set in test_sets:
        bench = load_bench(test_set["bench"])
        problems = bench.load(test_set["data_file"])
        records = bench.build_records(problems, limit=args.limit)
        if args.text_only:
            before = len(records)
            records = [record for record in records if not record.get("image_paths")]
            print(f"[gen] {test_set['name']}: text-only skipped {before - len(records)} image tasks", flush=True)
        if not records:
            raise SystemExit(
                f"no problem left in test set {test_set['name']} (check data file / --limit / --text-only)"
            )
        manifest = _read_data_manifest(test_set["data_file"])
        loaded.append(
            {
                **test_set,
                "records": records,
                "sample_ratio": float(manifest.get("sample_ratio", 1.0)),
                "expected_tasks": len(records),
            }
        )
    return loaded


def run_rank(test_sets: list[dict[str, Any]], args: argparse.Namespace, out_dir: str) -> None:
    plan = [
        (set_index, problem, sample)
        for set_index, test_set in enumerate(test_sets)
        for problem in range(len(test_set["records"]))
        for sample in range(args.n_sample)
    ]
    if not plan:
        os.makedirs(out_dir, exist_ok=True)
        write_jsonl(os.path.join(out_dir, "raw_rank000.jsonl"), [])
        print("[gen] no request to run", flush=True)
        return

    def _one(item: tuple[int, int, int]) -> dict[str, Any]:
        set_index, problem, sample = item
        test_set = test_sets[set_index]
        record = test_set["records"][problem]
        payload: dict[str, Any] = {
            "model": args.model,
            "messages": _build_messages(record),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        }
        if args.reasoning_effort:
            payload["reasoning_effort"] = args.reasoning_effort
        if args.seed is not None:
            payload["seed"] = (args.seed + problem * args.n_sample + sample) % (2**31)
        row: dict[str, Any] = {
            "test_set": test_set["name"],
            "task_id": record["task_id"],
            "sample_idx": sample,
            "raw_text": "",
            "num_output_tokens": 0,
            "prompt_len": 0,
            "finish_reason": None,
        }
        try:
            data = chat_completion(
                base_url=args.base_url,
                api_key=args.api_key,
                payload=payload,
                timeout=args.timeout,
                attempts=args.attempts,
            )
            choice = data["choices"][0]
            message = choice.get("message") or {}
            text = (message.get("content") or "").strip()
            source = "content"
            if not text:
                text = (message.get("reasoning_content") or message.get("reasoning") or "").strip()
                source = "reasoning" if text else "empty"
            usage = data.get("usage") or {}
            row.update(
                {
                    "raw_text": text,
                    "num_output_tokens": usage.get("completion_tokens") or 0,
                    "prompt_len": usage.get("prompt_tokens") or 0,
                    "finish_reason": choice.get("finish_reason"),
                    "text_source": source,
                }
            )
        except Exception as exc:  # one failed request must not abort the batch
            row["finish_reason"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    names = ",".join(test_set["name"] for test_set in test_sets)
    print(
        f"[gen] model={args.model} test_sets=[{names}] requests={len(plan)} n_sample={args.n_sample} "
        f"concurrency={args.concurrency} text_only={args.text_only}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(_one, plan))

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "raw_rank000.jsonl")
    write_jsonl(path, rows)
    errors = sum(1 for row in rows if row.get("error"))
    print(f"[gen] wrote {path} rows={len(rows)} errors={errors}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="run one evaluation task over one or more test sets")
    parser.add_argument("--task", default=None, help="task name; the output dir is outputs/<task>")
    parser.add_argument(
        "--test-set",
        action="append",
        default=[],
        metavar="BENCH:FILE",
        help="test set as BENCH:FILE (aime/gpqa/hle); repeatable, e.g. aime:data/aime_2026.jsonl",
    )
    parser.add_argument("--bench", default=None, choices=list(SUPPORTED), help="legacy single test set")
    parser.add_argument("--data-file", default=None, help="legacy single test set data file")
    parser.add_argument("--limit", type=int, default=None, help="first N problems of each test set")
    parser.add_argument("--out-dir", default=None, help="override the output dir")
    parser.add_argument("--base-url", default=os.environ.get("LLM_EVAL_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("LLM_EVAL_API_KEY"))
    parser.add_argument("--model", default=os.environ.get("LLM_EVAL_MODEL"))
    parser.add_argument("--n-sample", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument(
        "--reasoning-effort", default=None, help="passed through when set (OpenAI: low/medium/high)"
    )
    parser.add_argument("--seed", type=int, default=None, help="base seed; each request gets seed+index")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("LLM_EVAL_TIMEOUT", "600")))
    parser.add_argument("--attempts", type=int, default=int(os.environ.get("LLM_EVAL_ATTEMPTS", "3")))
    parser.add_argument(
        "--text-only", action="store_true", help="skip image questions in every test set (only HLE has images)"
    )
    args = parser.parse_args()

    test_sets = _parse_test_sets(args, parser)

    missing = [name for name in ("base_url", "api_key", "model") if not getattr(args, name)]
    if missing:
        parser.error(
            "missing "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            + " (or LLM_EVAL_BASE_URL / LLM_EVAL_API_KEY / LLM_EVAL_MODEL)"
        )

    if args.n_sample is None:
        args.n_sample = BENCH_DEFAULTS[test_sets[0]["bench"]]["n_sample"] if len(test_sets) == 1 else 1
    if args.temperature is None:
        args.temperature = 1.0
    if args.top_p is None:
        args.top_p = 1.0

    loaded = _load_test_sets(test_sets, args)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")  # noqa: UP017
    task = args.task or f"{args.model.replace('/', '_')}-n{args.n_sample}-{stamp}"
    out_dir = resolve_out_dir(args.out_dir, task=task)

    meta = {
        "task": task,
        "model_name": args.model,
        "base_url": args.base_url,
        "n_sample": args.n_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "seed": args.seed,
        "text_only": args.text_only,
        "limit": args.limit,
        "concurrency": args.concurrency,
        "test_sets": [
            {
                "name": test_set["name"],
                "bench": test_set["bench"],
                "data_file": test_set["data_file"],
                "sample_ratio": test_set["sample_ratio"],
                "expected_tasks": test_set["expected_tasks"],
            }
            for test_set in loaded
        ],
    }
    os.makedirs(out_dir, exist_ok=True)
    for old in Path(out_dir).glob("raw_rank*.jsonl"):
        old.unlink()
    with open(os.path.join(out_dir, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    print(f"OUT_DIR={out_dir}\nmeta={json.dumps(meta, ensure_ascii=False)}", flush=True)

    run_rank(loaded, args, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
