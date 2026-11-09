from __future__ import annotations

import json
import time
from typing import Any

import requests


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
    attempts: int,
) -> dict[str, Any]:
    """POST an OpenAI-compatible /chat/completions request, retrying transient failures.

    Retries every attempt but the last with exponential backoff, then re-raises so the
    caller can record the failure against a single item instead of aborting the batch.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_err = ""
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # network / timeout / 5xx land here
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(last_err)
