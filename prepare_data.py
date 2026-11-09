from __future__ import annotations

import argparse
import glob
import hashlib
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

SAMPLE_SEED = 20260101
DEFAULT_RATIOS = "0.01,0.1,0.2"


def _manifest_path(jsonl_path: Path) -> Path:
    return Path(str(jsonl_path) + ".meta.json")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _sample_key(task_id: str) -> bytes:
    return hashlib.sha256(f"{SAMPLE_SEED}:{task_id}".encode()).digest()


def _task_ids_sha(records: list[dict]) -> str:
    return hashlib.sha256("\n".join(sorted(r["task_id"] for r in records)).encode()).hexdigest()


def _ratio_path(full_path: Path, ratio: float) -> Path:
    if ratio >= 1.0:
        return full_path
    return full_path.with_name(f"{full_path.stem}_r{ratio:g}{full_path.suffix}")


def _select(records: list[dict[str, Any]], ratio: float) -> list[dict[str, Any]]:
    if ratio >= 1.0:
        return records
    keep_n = max(1, round(ratio * len(records)))
    ranked = sorted(records, key=lambda record: _sample_key(record["task_id"]))
    keep = {record["task_id"] for record in ranked[:keep_n]}
    return [record for record in records if record["task_id"] in keep]


def _raw_ready(raw_dir: Path) -> bool:
    if not raw_dir.is_dir():
        return False
    return any(raw_dir.rglob("*.parquet")) or any(raw_dir.rglob("*.jsonl"))


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


def ensure_full(
    bench: str, dest: Path, *, force: bool, no_images: bool, allow_partial: bool
) -> tuple[Path, list[dict[str, Any]]]:
    full_path = dest / OUTPUT_NAME[bench]
    manifest_path = _manifest_path(full_path)
    if not force and full_path.is_file():
        manifest = _read_manifest(manifest_path)
        if manifest.get("sample_ratio") == 1.0 and manifest.get("no_images") == no_images:
            try:
                records = _read_jsonl(full_path)
            except (OSError, ValueError):
                records = None
            if (
                records is not None
                and manifest.get("num_tasks") == len(records)
                and manifest.get("task_ids_sha256") == _task_ids_sha(records)
            ):
                if manifest.get("partial"):
                    print(f"[{bench}] reusing a PARTIAL dataset (allow-partial was used)", flush=True)
                return full_path, records
        print(f"[{bench}] full JSONL/manifest mismatch, rebuilding", flush=True)

    raw_dir = dest / "_raw" / bench
    # Download only when the raw cache is missing; --force rebuilds from the existing cache.
    if not _raw_ready(raw_dir):
        print(f"[{bench}] downloading {DATASETS[bench]} ...", flush=True)
        raw_dir = _download(DATASETS[bench], raw_dir)
    rows = _read_rows(raw_dir)
    converted = CONVERTERS[bench](rows, dest=dest, no_images=no_images)
    if not converted:
        raise SystemExit(f"[{bench}] converted 0 rows from {raw_dir}")

    expected = EXPECTED_ROWS[bench]
    partial = False
    text_only_hle = bench == "hle" and no_images
    if not text_only_hle and len(converted) != expected:
        message = f"[{bench}] converted {len(converted)} rows, expected {expected}; the raw download may be incomplete"
        if allow_partial:
            print(f"{message} (continuing: --allow-partial)", flush=True)
            partial = True
        else:
            raise SystemExit(f"{message}\nDelete {raw_dir} and retry, or pass --allow-partial to continue")

    _write_jsonl(full_path, converted)
    _write_manifest(
        manifest_path,
        {
            "bench": bench,
            "dataset_id": DATASETS[bench],
            "sample_ratio": 1.0,
            "no_images": no_images,
            "num_tasks": len(converted),
            "task_ids_sha256": _task_ids_sha(converted),
            "partial": partial,
        },
    )
    return full_path, converted


def derive_ratios(
    bench: str,
    full_path: Path,
    full_records: list[dict[str, Any]],
    ratios: list[float],
    *,
    force: bool,
    no_images: bool,
) -> list[tuple[Path, int]]:
    full_sha = _task_ids_sha(full_records)
    full_manifest = _read_manifest(_manifest_path(full_path))
    source_partial = bool(full_manifest.get("partial"))
    if source_partial:
        print(f"[{bench}] warning: deriving ratios from a PARTIAL full dataset", flush=True)
    made: list[tuple[Path, int]] = []
    for ratio in ratios:
        if ratio >= 1.0:
            continue
        path = _ratio_path(full_path, ratio)
        mpath = _manifest_path(path)
        if not force and path.is_file():
            manifest = _read_manifest(mpath)
            if (
                manifest.get("sample_ratio") == ratio
                and manifest.get("no_images") == no_images
                and manifest.get("source_full_task_ids_sha256") == full_sha
            ):
                print(f"[{bench}] ratio {ratio:g} exists, skip: {path}", flush=True)
                continue
        selected = _select(full_records, ratio)
        _write_jsonl(path, selected)
        _write_manifest(
            mpath,
            {
                "bench": bench,
                "dataset_id": DATASETS[bench],
                "sample_ratio": ratio,
                "no_images": no_images,
                "num_tasks": len(selected),
                "task_ids_sha256": _task_ids_sha(selected),
                "source_full_task_ids_sha256": full_sha,
                "source_full_partial": source_partial,
            },
        )
        made.append((path, len(selected)))
    return made


def prepare(
    bench: str, dest: Path, *, ratios: list[float], force: bool, no_images: bool, allow_partial: bool
) -> tuple[int, int]:
    full_path, full_records = ensure_full(
        bench, dest, force=force, no_images=no_images, allow_partial=allow_partial
    )
    print(f"[{bench}] full: {len(full_records)} rows -> {full_path}", flush=True)
    made = derive_ratios(bench, full_path, full_records, ratios, force=force, no_images=no_images)
    for path, count in made:
        print(f"[{bench}] ratio copy: {count} rows -> {path}", flush=True)
    return len(full_records), len(made)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="download full benchmark datasets from ModelScope once and write deterministic ratio copies"
    )
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    parser.add_argument("--benches", default="aime,gpqa,hle", help="comma-separated subset of aime,gpqa,hle")
    parser.add_argument(
        "--sample-ratios",
        default=DEFAULT_RATIOS,
        help="comma-separated ratios to generate as physical copies (default 0.01,0.1,0.2); the full dataset is always kept",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even when the output already exists (does not re-download when the raw cache exists)",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="skip HLE image questions entirely (text-only full dataset; ratios derive from it)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="warn instead of failing when the full dataset row count differs from the expected count",
    )
    args = parser.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    benches = [name.strip() for name in args.benches.split(",") if name.strip()]
    unknown = sorted(set(benches) - set(DATASETS))
    if unknown:
        parser.error(f"unknown benchmark: {unknown}; expected aime,gpqa,hle")

    ratios: list[float] = []
    for value in (item.strip() for item in args.sample_ratios.split(",")):
        if not value:
            continue
        try:
            ratio = float(value)
        except ValueError:
            parser.error(f"--sample-ratios must be numbers: {value!r}")
        if not (0 < ratio <= 1):
            parser.error(f"--sample-ratios must be in (0, 1]: {value!r}")
        if ratio not in ratios:
            ratios.append(ratio)

    full_total = 0
    made_total = 0
    for bench in benches:
        full_count, made_count = prepare(
            bench,
            dest,
            ratios=ratios,
            force=args.force,
            no_images=args.no_images,
            allow_partial=args.allow_partial,
        )
        full_total += full_count
        made_total += made_count
    print(f"[done] {full_total} full rows, {made_total} ratio copies under {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
