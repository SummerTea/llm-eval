# llm-eval

## 简介

独立的模型质量评测：通过任意 OpenAI 兼容的 `/chat/completions` 端点生成答案，然后对答案评分。
本项目抽取自 DeepInfer 的 `eval/`，与引擎解耦——只通过 HTTP 通信。

基准：AIME 2026（N 次采样上的 Pass@1）、GPQA Diamond、HLE。

## 双端点说明

评测使用两个互相独立的端点：

| 角色 | 环境变量 | 使用方 |
| :--- | :--- | :--- |
| 被测模型 | `LLM_EVAL_BASE_URL` / `LLM_EVAL_API_KEY` / `LLM_EVAL_MODEL` | `run_eval.py` |
| 判题模型 | `EVAL_JUDGE_BASE_URL` / `EVAL_JUDGE_API_KEY` / `EVAL_JUDGE_MODEL` | `score.py --check llm` |

判题模型绝不回退使用被测模型的凭据。

## 1. 安装

```bash
pip install -r llm-eval/requirements.txt        # runtime: requests, pillow
```

## 2. 准备数据（手动，从 ModelScope）

数据集**不随仓库分发**。请一次性下载并转换：

```bash
pip install -r llm-eval/requirements-data.txt   # modelscope, datasets
python llm-eval/prepare_data.py                 # all three benchmarks
python llm-eval/prepare_data.py --benches aime,gpqa
python llm-eval/prepare_data.py --no-images     # text-only: skip HLE image questions
```

下载来源（ModelScope）：

| 基准 | 数据集 id | 输出 |
| :--- | :--- | :--- |
| AIME 2026 | `evalscope/aime26` | `llm-eval/data/aime_2026.jsonl` (30) |
| GPQA Diamond | `AI-ModelScope/gpqa_diamond` | `llm-eval/data/gpqa_diamond.jsonl` (198) |
| HLE | `cais/hle` | `llm-eval/data/hle_test.jsonl` (2500) + `data/hle_images/` |

`data/` 与 `outputs/` 已被 git 忽略：数据集带有各自的许可证
（HLE 要求不得再分发；AIME 为非商业用途），因此仅保留在本地。

## 3. 运行

```bash
export LLM_EVAL_BASE_URL=https://api.openai.com/v1
export LLM_EVAL_API_KEY=sk-...
export LLM_EVAL_MODEL=gpt-4o

export EVAL_JUDGE_BASE_URL=https://api.deepseek.com/v1
export EVAL_JUDGE_API_KEY=...
export EVAL_JUDGE_MODEL=deepseek-chat

MODEL="$LLM_EVAL_MODEL" bash llm-eval/aime/run.sh
MODEL="$LLM_EVAL_MODEL" bash llm-eval/gpqa/run.sh
MODEL="$LLM_EVAL_MODEL" bash llm-eval/hle/run.sh
```

也可以直接分别调用两个阶段：

```bash
python llm-eval/run_eval.py --bench hle --data-file llm-eval/data/hle_test.jsonl \
    --model "$LLM_EVAL_MODEL" --n-sample 1 --text-only

python llm-eval/score.py --out-dir llm-eval/outputs/hle/<tag> --check llm
```

快速回归示例（确定性的 20% 子集）：

```bash
SAMPLE_RATIO=0.2 MODEL="$LLM_EVAL_MODEL" bash llm-eval/gpqa/run.sh
```

`SAMPLE_RATIO=0.2` 是推荐的快速检查比例。GPQA 默认使用确定性的 `rule` 判题，
因此该命令**不会产生任何判题调用**。

### 判题

AIME 与 GPQA 默认使用确定性的 `rule` 判题。HLE 自动使用 `llm`
（其答案多为自由文本，精确匹配会系统性低估得分）。使用 `--check llm`
可将 AIME/GPQA 切换为 LLM 判题；`--judge` 为兼容旧用法保留，两者同时指定时
`--check` 优先生效。

## 参数

`run_eval.py`：

| 选项 | 含义 |
| :--- | :--- |
| `--text-only` | 跳过含图片的题目（当前仅 HLE 有图片）。评分阶段会应用相同过滤，因此覆盖率仍然正确。 |
| `--sample-ratio` | 参与评测的题目比例，`0 < r <= 1`（默认 1.0）。确定性固定种子选择（对 task id 做 SHA-256）：同一数据集 + 同一比例，每次运行选出的子集完全相同。 |
| `--n-sample` | 每题采样次数（默认：aime 32 / gpqa 8 / hle 1）。 |
| `--temperature`, `--top-p`, `--max-tokens` | 原样透传。 |
| `--reasoning-effort` | 仅在设置时发送（OpenAI：`low`/`medium`/`high`）。 |
| `--seed` | 基础种子；每个请求使用 `seed + index`（尽力而为，取决于端点）。 |
| `--concurrency` | 并发请求数（默认 16）。 |
| `--base-url`, `--api-key`, `--model` | 覆盖 `LLM_EVAL_*` 环境变量。 |

**采样比例建议（`--sample-ratio`）**

| 比例 | 用途 |
| :--- | :--- |
| `0.01` | 仅用于冒烟/存活性检查（GPQA≈2 题，AIME 因 `max(1)` 下限=1 题，HLE≈25 道题，默认含图题；纯文本需加 `--text-only`）；统计上无法用于准确率回归（置信区间约 ±50%）。 |
| `0.1` | 粗筛，只能发现明显的大幅下跌。 |
| `0.2` | 推荐的准确率回归检查比例。 |

`score.py`：`--check rule|llm`（默认：`rule`，HLE 除外，HLE 使用 `llm`），
`--judge auto|rule|llm`（为兼容保留；`--check` 优先生效），
`--judge-concurrency`、`--data-file`。

### 快速回归检查

- 推荐比例：`0.2`（`SAMPLE_RATIO=0.2`）。
- GPQA 是主要探针：198 题，约选中 40 题，使用 rule 判题 => 零判题调用。
- AIME 接近上限（约 99.9%），只能作为冒烟信号。
- HLE 需要 LLM 判题（以及判题凭据）。

### 运维说明 / 已接受的风险

- `meta.json.expected_tasks` 是**选中的**题目数，而非完整数据集大小。
- 用不同比例重跑到同一输出目录会覆盖 `raw_rank*.jsonl`
  与 `meta.json`；请在 `TAG` 中包含比例以隔离不同运行。
- 量化与非量化对比必须使用同一数据集版本、同一比例和同一环境；
  `meta.selected_task_ids` 是对齐键。
- 代码支持 GPQA 数据集与 HLE 图片/视觉路径，但本阶段仅做过 mock 测试。

## 4. 输出

`llm-eval/outputs/<bench>/<tag>/`：

| 文件 | 内容 |
| :--- | :--- |
| `meta.json` | 运行配置（不含机密）。 |
| `raw_rank000.jsonl` | 模型原始输出。 |
| `scored.jsonl` | 每次 rollout 的抽取答案与判定结果。 |
| `summary.json` | `Pass@1`、`Pass@k`、覆盖率与诊断信息。 |

只有当 `summary.json.coverage_complete` 为 `true` 时，`Pass@1` 才有意义。

## 5. 目录结构

```text
llm-eval/
├── prepare_data.py     # ModelScope 下载 + 转换（手动，一次性）
├── run_eval.py         # OpenAI 兼容生成
├── score.py            # 抽取、rule/LLM 判题、Pass@k
├── common/             # client, io, loader, paths, passk, judge
├── aime/ gpqa/ hle/     # bench.py（prompt/抽取/评分）+ run.sh
└── data/ outputs/      # 本地生成，已被 git 忽略
```

## 工程参考

本项目抽取自 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 的 `eval/` 评测框架，并改造为仅依赖 OpenAI 兼容 API 的独立模型评测工具。
