from __future__ import annotations

import argparse
import glob
import io
import json
import random
import re
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DEST = EVAL_DIR / "data"

# ModelScope dataset ids (Verified against modelscope.cn dataset cards).
DATASETS = {
    "aime": "evalscope/aime26",
    "gpqa": "AI-ModelScope/gpqa_diamond",
    "hle": "cais/hle",
}
EXPECTED_ROWS = {"aime": 30, "gpqa": 198, "hle": 2500}
OUTPUT_NAME = {"aime": "aime_2026.jsonl", "gpqa": "gpqa_diamond.jsonl", "hle": "hle_test.jsonl"}


def _download(dataset_id: str, dest: Path) -> Path:
    from modelscope import dataset_snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    return Path(dataset_snapshot_download(dataset_id, local_dir=str(dest)))


def _read_rows(raw_dir: Path) -> list[dict[str, Any]]:
    parquet = sorted(glob.glob(str(raw_dir / "**" / "*.parquet"), recursive=True))
    if parquet:
        from datasets import load_dataset

        return list(load_dataset("parquet", data_files=parquet, split="train"))
    jsonl = sorted(glob.glob(str(raw_dir / "**" / "*.jsonl"), recursive=True))
    rows: list[dict[str, Any]] = []
    for path in jsonl:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no parquet/jsonl data found under {raw_dir}")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_aime(rows: list[dict[str, Any]], **_kwargs: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        problem = row.get("problem") or row.get("question")
        answer = row.get("answer")
        if problem is None or answer is None:
            continue
        out.append(
            {
                "task_id": str(row.get("id") or row.get("task_id") or f"aime_2026_{index}"),
                "question": str(problem),
                "answer": str(answer),
            }
        )
    return out


def _shuffle_options(correct: str, wrong: list[str], question: str, seed: int) -> tuple[str, str]:
    options = [correct, *wrong[:3]]
    random.Random(seed).shuffle(options)
    letter = "ABCD"[options.index(correct)]
    body = question.strip() + "\n\n" + "\n".join(
        f"({label}) {option}" for label, option in zip("ABCD", options, strict=True)
    )
    return body, letter


def convert_gpqa(rows: list[dict[str, Any]], **_kwargs: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        question = row.get("Question") or row.get("question") or row.get("problem")
        if question is None:
            continue
        domain = row.get("domain") or row.get("category") or row.get("subdomain") or "unknown"
        correct = row.get("Correct Answer") or row.get("correct_answer") or row.get("answer")
        wrong = [
            row.get(key)
            for key in (
                "Incorrect Answer 1",
                "Incorrect Answer 2",
                "Incorrect Answer 3",
                "incorrect_answer_1",
                "incorrect_answer_2",
                "incorrect_answer_3",
            )
        ]
        wrong = [str(value) for value in wrong if value not in (None, "")]
        if correct is not None and len(wrong) >= 3:
            body, letter = _shuffle_options(str(correct), wrong, str(question), index)
        else:
            # Already normalized: the answer is a letter and the question carries the choices.
            raw = str(correct or "")
            match = (
                re.search(r"\\boxed\{\s*\(?([A-D])\)?\s*\}", raw, re.I)
                or re.search(r"(?i)answer\s*[:：]\s*\(?([A-D])\b", raw)
                or re.fullmatch(r"\s*\(?([A-D])\)?\s*", raw)
            )
            if not match or "(A)" not in str(question):
                continue
            body, letter = str(question), match.group(1).upper()
        out.append(
            {
                "task_id": str(row.get("task_id") or row.get("id") or f"gpqa_diamond_{index}"),
                "question": body,
                "answer": letter,
                "category": str(domain),
            }
        )
    return out


def _image_bytes(value: Any) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("bytes")
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if hasattr(value, "save"):  # PIL.Image from datasets
        buffer = io.BytesIO()
        value.save(buffer, format="PNG")
        return buffer.getvalue()
    return None


def convert_hle(
    rows: list[dict[str, Any]],
    *,
    dest: Path,
    no_images: bool,
    **_kwargs: Any,
) -> list[dict[str, Any]]:
    images_dir = dest / "hle_images"
    out: list[dict[str, Any]] = []
    saved = 0
    for index, row in enumerate(rows):
        question = row.get("question")
        answer = row.get("answer")
        if question is None or answer is None:
            continue
        task_id = str(row.get("id") or row.get("task_id") or f"hle_{index:04d}")
        image_path = None
        data = _image_bytes(row.get("image_preview"))
        if data:
            if no_images:
                continue
            images_dir.mkdir(parents=True, exist_ok=True)
            path = images_dir / f"{task_id}.png"
            path.write_bytes(data)
            image_path = str(path)
            saved += 1
        record = {
            "task_id": task_id,
            "question": str(question),
            "answer": str(answer),
            "category": str(row.get("category") or row.get("raw_subject") or "unknown"),
        }
        if image_path:
            record["image"] = image_path
        out.append(record)
    print(f"[hle] image questions saved: {saved}", flush=True)
    return out


CONVERTERS = {"aime": convert_aime, "gpqa": convert_gpqa, "hle": convert_hle}


def prepare(bench: str, dest: Path, *, force: bool, no_images: bool) -> int:
    out_path = dest / OUTPUT_NAME[bench]
    if out_path.is_file() and not force:
        print(f"[{bench}] exists, skip: {out_path} (use --force to rebuild)", flush=True)
        return 0

    print(f"[{bench}] downloading {DATASETS[bench]} ...", flush=True)
    raw_dir = _download(DATASETS[bench], dest / "_raw" / bench)
    rows = _read_rows(raw_dir)
    converted = CONVERTERS[bench](rows, dest=dest, no_images=no_images)
    if not converted:
        raise SystemExit(f"[{bench}] converted 0 rows from {raw_dir}")

    expected = EXPECTED_ROWS[bench]
    if not no_images and len(converted) != expected:
        print(f"[{bench}] warning: converted {len(converted)} rows, expected {expected}", flush=True)

    _write_jsonl(out_path, converted)
    print(f"[{bench}] wrote {len(converted)} rows -> {out_path}", flush=True)
    return len(converted)


def main() -> int:
    parser = argparse.ArgumentParser(description="download benchmark datasets from ModelScope and convert to jsonl")
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    parser.add_argument("--benches", default="aime,gpqa,hle", help="comma-separated subset of aime,gpqa,hle")
    parser.add_argument("--force", action="store_true", help="rebuild even when the output already exists")
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="skip HLE image questions entirely (smaller data dir; use with run_eval --text-only)",
    )
    args = parser.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    benches = [name.strip() for name in args.benches.split(",") if name.strip()]
    unknown = sorted(set(benches) - set(DATASETS))
    if unknown:
        parser.error(f"unknown benchmark: {unknown}; expected aime,gpqa,hle")

    total = 0
    for bench in benches:
        total += prepare(bench, dest, force=args.force, no_images=args.no_images)
    print(f"[done] {total} rows under {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
