from __future__ import annotations

import html
import json
import os
from datetime import UTC, datetime
from typing import Any

_BAD_FINISH_REASONS = {"length", "prompt_length_exceeded", "ignored", "aborted", "error", "preempted"}
_MAX_PROBLEM_ROWS = 500

_FINISH_ZH = {
    "stop": "正常结束",
    "length": "达到最大长度",
    "prompt_length_exceeded": "提示词过长",
    "ignored": "被忽略",
    "aborted": "中止",
    "error": "错误",
    "preempted": "被抢占",
}
_EXTRACT_ZH = {
    "boxed": "\\boxed{}",
    "tail_number": "末尾数字",
    "answer_line": "Answer 行",
    "tail_letter": "末尾字母",
    "none": "未抽取到",
}
_JUDGE_ZH = {"rule": "规则", "llm": "裁判", "auto": "自动"}

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


def _map_keys(obj: Any, mapping: dict[str, str]) -> Any:
    if not isinstance(obj, dict):
        return obj
    return {mapping.get(str(key), key): value for key, value in obj.items()}


def _kv(obj: Any) -> str:
    if not isinstance(obj, dict) or not obj:
        return "—"
    return ", ".join(f"{_e(key)}: {_e(obj[key])}" for key in sorted(obj, key=lambda k: str(k)))


def _bool_zh(value: Any) -> str:
    if value is True:
        return "是"
    if value is False:
        return "否"
    return _val(value)


def _correct_zh(value: Any) -> str:
    if value is True:
        return "正确"
    if value is False:
        return "错误"
    return _val(value)


def _judge_zh(value: Any) -> str:
    return _val(_JUDGE_ZH.get(str(value), value))


def _excerpt(raw: Any, limit: int = 200) -> str:
    return _e(" ".join(str("" if raw is None else raw).split())[:limit])


def write_report(out_dir: str, summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    meta_items: list[tuple[str, str]] = [
        ("基准", _val(summary.get("bench"))),
        ("模型", _val(summary.get("model_name"))),
        ("判分方式", _judge_zh(summary.get("judge"))),
        ("采样比例", _val(summary.get("sample_ratio"))),
        ("生成时间", _val(generated)),
    ]
    if summary.get("judge_model"):
        meta_items.append(("判题模型", _val(summary.get("judge_model"))))
    meta_html = "".join(f"<span><b>{_e(label)}</b>: {value}</span>" for label, value in meta_items)

    cards = [
        ("Pass@1", _pct(summary.get("Pass@1"))),
        ("平均得分", _pct(summary.get("score"))),
        ("题目数", _val(summary.get("num_tasks"))),
        ("样本数", _val(summary.get("num_rollouts"))),
        ("每题采样", _val(summary.get("n_sample"))),
        ("覆盖率完整", _bool_zh(summary.get("coverage_complete"))),
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
            "<table><thead><tr><th>指标</th><th>柱状图</th><th>数值</th></tr></thead><tbody>"
            + "".join(
                f"<tr><td><b>{_e(key)}</b></td><td>{_bar(summary.get(key))}</td><td>{_pct(summary.get(key))}</td></tr>"
                for key in pass_keys
            )
            + "</tbody></table>"
        )
    else:
        pass_html = '<p class="muted">汇总中没有 Pass@k 指标。</p>'

    diag_items: list[tuple[str, str]] = [
        ("结束原因分布", _kv(_map_keys(summary.get("finish_reasons"), _FINISH_ZH))),
        ("答案抽取来源", _kv(_map_keys(summary.get("extract_sources"), _EXTRACT_ZH))),
        ("截断样本数", _val(summary.get("truncated_rollouts"))),
        ("无效样本数", _val(summary.get("invalid_rollouts"))),
        ("抽取失败率", _pct(summary.get("extract_failed"))),
        ("规则/裁判一致率", _pct(summary.get("rule_vs_llm_agreement"))),
        ("裁判未解析率", _pct(summary.get("judge_unparsed"))),
        ("裁判结论未解析率", _pct(summary.get("judge_verdict_unparsed"))),
    ]
    if summary.get("expected_tasks") is not None:
        diag_items.append(("期望题目数", _val(summary.get("expected_tasks"))))
    if summary.get("expected_rollouts") is not None:
        diag_items.append(("期望样本数", _val(summary.get("expected_rollouts"))))
    if summary.get("missing_tasks"):
        diag_items.append(("缺失题目数", _val(summary.get("missing_tasks"))))
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
        f"<details><summary>逐题正确率（共 {len(accuracy)} 题）</summary>"
        "<table><thead><tr><th>题目</th><th>柱状图</th><th>正确率</th><th>样本数</th></tr></thead><tbody>"
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
        f"<td>{_correct_zh(row.get('correct'))}</td>"
        f"<td class='mono'>{_text(row.get('pred'))}</td>"
        f"<td class='mono'>{_text(row.get('gold'))}</td>"
        f"<td>{_e(_EXTRACT_ZH.get(str(row.get('extract_source')), row.get('extract_source')))}</td>"
        f"<td>{_e(_FINISH_ZH.get(str(row.get('finish_reason')), row.get('finish_reason')))}</td>"
        f"<td class='mono'>{_excerpt(row.get('raw_text'))}</td>"
        "</tr>"
        for row in shown
    )
    if problems:
        note = (
            f"<p class='muted'>仅显示前 {_MAX_PROBLEM_ROWS} 条，共 {len(problems)} 条问题样本</p>"
            if len(problems) > _MAX_PROBLEM_ROWS
            else ""
        )
        problem_html = (
            note
            + "<table><thead><tr>"
            "<th>题目</th><th>样本序号</th><th>是否正确</th><th>预测</th><th>标准答案</th>"
            "<th>抽取来源</th><th>结束原因</th><th>输出摘录</th>"
            "</tr></thead><tbody>"
            + problem_rows
            + "</tbody></table>"
        )
    else:
        problem_html = "<p class='muted'>没有异常或错误的样本。</p>"

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(summary.get("bench"))} 评分报告</title>
<style>{_STYLE}</style>
</head>
<body>
<main>
<h1>{_e(summary.get("bench"))} 评分报告</h1>
<div class="muted">{_e(summary.get("model_name"))}</div>
<div class="meta">{meta_html}</div>
<div class="cards">{cards_html}</div>
<h2>Pass@k 指标</h2>
{pass_html}
<h2>诊断信息</h2>
{diag_html}
<h2>逐题正确率</h2>
{task_table}
<h2>异常 / 翻车样本</h2>
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
