from __future__ import annotations

from pathlib import Path

COMMON_DIR = Path(__file__).resolve().parent
EVAL_DIR = COMMON_DIR.parent
DATA_DIR = EVAL_DIR / "data"
DEFAULT_OUT_ROOT = EVAL_DIR / "outputs"


def resolve_out_dir(out_dir: str | None, *, bench: str, tag: str) -> str:
    return out_dir or str(DEFAULT_OUT_ROOT / bench / tag)
