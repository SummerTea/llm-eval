from __future__ import annotations

import html
import json
import os
from datetime import UTC, datetime
from typing import Any

_BAD_FINISH_REASONS = {"length", "prompt_length_exceeded", "ignored", "aborted", "error", "preempted"}
_MAX_PROBLEM_ROWS = 500

_STYLE = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 24px;
  background: #f6f7f9;
  color: #1d2129;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.5;
}
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 28px 0 10px; }
.muted { color: #646a73; font-size: 13px; }
.meta { display: flex; flex-wrap: wrap; gap: 6px 18px; margin-top: 8px; font-size: 13px; }
.meta span { color: #424852; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 14px; }
.card { background: #fff; border: 1px solid #e3e6eb; border-radius: 8px; padding: 10px 12px; }
.card .label { font-size: 12px; color: #646a73; text-transform: uppercase; letter-spacing: .03em; }
.card .value { margin-top: 4px; font-size: 20px; font-weight: 600; }
table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e3e6eb; border-radius: 8px; overflow: hidden; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #eef0f3; font-size: 13px; vertical-align: top; }
th { background: #fafbfc; color: #424852; font-weight: 600; }
tr:last-child td { border-bottom: 0; }
code, .mono { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 12px; }
.track { display: inline-block; width: 180px; height: 10px; background: #eef0f3; border-radius: 3px; overflow: hidden; vertical-align: middle; }
.bar { display: inline-block; height: 10px; background: #2f6feb; border-radius: 3px; }
.diag { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 8px; }
.diag div { background: #fff; border: 1px solid #e3e6eb; border-radius: 6px; padding: 8px 10px; font-size: 13px; }
.diag b { display: block; color: #424852; font-weight: 600; margin-bottom: 2px; }
details { background: #fff; border: 1px solid #e3e6eb; border-radius: 8px; padding: 10px 14px; }
summary { cursor: pointer; font-weight: 600; }
"""


def _e(value: Any) -> str:
    return html.escape(str(value))


def _text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return _e(value)


def _val(value: Any, default: str = "—") -> str:
    return default if value is None else _e(value)


def _pct(value: Any, default: str = "—") -> str:
    if value is None:
        return default
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return _e(value)


def _width(value: Any) -> float:
    try:
        return max(0.0, min(100.0, float(value) * 100))
    except (TypeError, ValueError):
        return 0.0


def _bar(value: Any) -> str:
    return f'<div class="track"><div class="bar" style="width:{_width(value):.2f}%"></div></div>'


def _kv(obj: Any) -> str:
    if not isinstance(obj, dict) or not obj:
        return "—"
    return ", ".join(f"{_e(key)}: {_e(obj[key])}" for key in sorted(obj, key=lambda k: str(k)))


def _excerpt(raw: Any, limit: int = 200) -> str:
    return _e(" ".join(str("" if raw is None else raw).split())[:limit])


def write_report(out_dir: str, summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    meta_items = [
        ("bench", summary.get("bench")),
        ("model_name", summary.get("model_name")),
        ("judge", summary.get("judge")),
        ("sample_ratio", summary.get("sample_ratio")),
        ("generated", generated),
    ]
    if summary.get("judge_model"):
        meta_items.append(("judge_model", summary.get("judge_model")))
    meta_html = "".join(f"<span><b>{_e(label)}</b>: {_val(value)}</span>" for label, value in meta_items)

    cards = [
        ("Pass@1", _pct(summary.get("Pass@1"))),
        ("score", _pct(summary.get("score"))),
        ("num_tasks", _val(summary.get("num_tasks"))),
        ("num_rollouts", _val(summary.get("num_rollouts"))),
        ("n_sample", _val(summary.get("n_sample"))),
        ("coverage_complete", _val(summary.get("coverage_complete"))),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="label">{_e(label)}</div><div class="value">{value}</div></div>'
        for label, value in cards
    )

    def _pass_order(key: Any) -> tuple[int, str]:
        suffix = str(key).split("@", 1)[1]
        return (int(suffix), str(key)) if suffix.isdigit() else (1 << 30, str(key))

    pass_keys = sorted((key for key in summary if str(key).startswith("Pass@")), key=_pass_order)
    if pass_keys:
        pass_html = (
            "<table><thead><tr><th>Metric</th><th>Bar</th><th>Value</th></tr></thead><tbody>"
            + "".join(
                f"<tr><td><b>{_e(key)}</b></td><td>{_bar(summary.get(key))}</td><td>{_pct(summary.get(key))}</td></tr>"
                for key in pass_keys
            )
            + "</tbody></table>"
        )
    else:
        pass_html = '<p class="muted">No Pass@k metrics in summary.</p>'

    diag_items: list[tuple[str, str]] = [
        ("finish_reasons", _kv(summary.get("finish_reasons"))),
        ("extract_sources", _kv(summary.get("extract_sources"))),
        ("truncated_rollouts", _val(summary.get("truncated_rollouts"))),
        ("invalid_rollouts", _val(summary.get("invalid_rollouts"))),
        ("extract_failed", _pct(summary.get("extract_failed"))),
        ("rule_vs_llm_agreement", _pct(summary.get("rule_vs_llm_agreement"))),
        ("judge_unparsed", _pct(summary.get("judge_unparsed"))),
        ("judge_verdict_unparsed", _pct(summary.get("judge_verdict_unparsed"))),
    ]
    if summary.get("expected_tasks") is not None:
        diag_items.append(("expected_tasks", _val(summary.get("expected_tasks"))))
    if summary.get("expected_rollouts") is not None:
        diag_items.append(("expected_rollouts", _val(summary.get("expected_rollouts"))))
    if summary.get("missing_tasks"):
        diag_items.append(("missing_tasks", _val(summary.get("missing_tasks"))))
    diag_html = '<div class="diag">' + "".join(
        f"<div><b>{_e(label)}</b>{value}</div>" for label, value in diag_items
    ) + "</div>"

    counts: dict[str, int] = {}
    corrects: dict[str, int] = {}
    for row in results:
        tid = str(row.get("task_id"))
        counts[tid] = counts.get(tid, 0) + 1
        corrects[tid] = corrects.get(tid, 0) + (1 if row.get("correct") else 0)

    accuracy = summary.get("task_accuracy")
    if not isinstance(accuracy, dict) or not accuracy:
        accuracy = {tid: corrects[tid] / counts[tid] for tid in counts}

    task_table = (
        f"<details><summary>Per-task accuracy ({len(accuracy)} tasks)</summary>"
        "<table><thead><tr><th>task_id</th><th>Bar</th><th>Accuracy</th><th>Rollouts</th></tr></thead><tbody>"
        + "".join(
            f"<tr><td class='mono'>{_e(tid)}</td><td>{_bar(accuracy.get(tid, 0.0))}</td>"
            f"<td>{_pct(accuracy.get(tid, 0.0))}</td><td>{_val(counts.get(tid, 0))}</td></tr>"
            for tid in sorted(accuracy)
        )
        + "</tbody></table></details>"
    )

    problems = [
        row
        for row in results
        if (not row.get("correct"))
        or str(row.get("finish_reason")) in _BAD_FINISH_REASONS
        or row.get("judge_error")
    ]
    problems.sort(key=lambda row: (str(row.get("task_id")), str(row.get("sample_idx"))))
    shown = problems[:_MAX_PROBLEM_ROWS]
    problem_rows = "".join(
        "<tr>"
        f"<td class='mono'>{_e(row.get('task_id'))}</td>"
        f"<td>{_e(row.get('sample_idx'))}</td>"
        f"<td>{_e(row.get('correct'))}</td>"
        f"<td class='mono'>{_text(row.get('pred'))}</td>"
        f"<td class='mono'>{_text(row.get('gold'))}</td>"
        f"<td>{_e(row.get('extract_source'))}</td>"
        f"<td>{_e(row.get('finish_reason'))}</td>"
        f"<td class='mono'>{_excerpt(row.get('raw_text'))}</td>"
        "</tr>"
        for row in shown
    )
    if problems:
        note = (
            f"<p class='muted'>showing first {_MAX_PROBLEM_ROWS} of {len(problems)} problem rollouts</p>"
            if len(problems) > _MAX_PROBLEM_ROWS
            else ""
        )
        problem_html = (
            note
            + "<table><thead><tr>"
            "<th>task_id</th><th>sample_idx</th><th>correct</th><th>pred</th><th>gold</th>"
            "<th>extract_source</th><th>finish_reason</th><th>raw excerpt</th>"
            "</tr></thead><tbody>"
            + problem_rows
            + "</tbody></table>"
        )
    else:
        problem_html = "<p class='muted'>No problem rollouts.</p>"

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(summary.get("bench"))} report</title>
<style>{_STYLE}</style>
</head>
<body>
<main>
<h1>{_e(summary.get("bench"))} scoring report</h1>
<div class="muted">{_e(summary.get("model_name"))}</div>
<div class="meta">{meta_html}</div>
<div class="cards">{cards_html}</div>
<h2>Pass@k</h2>
{pass_html}
<h2>Diagnostics</h2>
{diag_html}
<h2>Per-task accuracy</h2>
{task_table}
<h2>Problem rollouts</h2>
{problem_html}
</main>
</body>
</html>
"""

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(doc)
    return path
