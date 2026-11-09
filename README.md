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

数据集**不随仓库分发**。`prepare_data.py` 会从 ModelScope **下载一次完整数据集**，
然后写出完整 JSONL 以及若干**物理比例副本**：

```bash
pip install -r llm-eval/requirements-data.txt   # modelscope, datasets
python llm-eval/prepare_data.py                 # all three benchmarks
python llm-eval/prepare_data.py --benches aime,gpqa
python llm-eval/prepare_data.py --no-images     # text-only: skip HLE image questions
```

默认生成的比例为 `0.01,0.1,0.2`（逗号分隔的 `--sample-ratios`，默认值即 `0.01,0.1,0.2`），
完整数据集始终保留。显式传入 `--sample-ratios` 会**替换**该次调用的默认集合；
若要在保留默认比例的同时增加 `0.5`：

```bash
python llm-eval/prepare_data.py --sample-ratios 0.01,0.1,0.2,0.5
```

之后若只想补出 `0.5`：

```bash
python llm-eval/prepare_data.py --sample-ratios 0.5
```

如果完整 JSONL / raw 缓存已存在于本地，上述补出 `0.5` 的操作完全在本地派生，
**不会重新下载**；只有 raw 缓存缺失时才会触发下载。

文件命名：完整文件为 `aime_2026.jsonl` / `gpqa_diamond.jsonl` / `hle_test.jsonl`；
比例副本为 `aime_2026_r0.2.jsonl` 等。每个文件都有一个旁车
`<file>.meta.json`，记录 `sample_ratio`、`num_tasks`、`task_ids_sha256`。

抽样是确定性的：按 `SHA-256(固定种子 : task_id)` 排序，取前 `k = max(1, round(r * N))` 个；
同一数据集 + 同一比例 => 每次得到的子集完全相同。

`prepare_data.py` 选项：

| 选项 | 含义 |
| :--- | :--- |
| `--sample-ratios` | 逗号分隔的比例列表（默认 `0.01,0.1,0.2`），生成对应物理副本；完整数据集始终保留。 |
| `--no-images` | 先过滤掉 HLE 含图题，得到纯文本完整集，再从中派生各比例。 |
| `--force` | 用**已存在的 raw 缓存**重建完整 JSONL 与比例副本，不重新下载；如需强制全新下载，请删除 `data/_raw/<bench>`。 |
| `--allow-partial` | 行数不符时降级为警告；完整 manifest 记录 `partial: true`，复用与比例派生时都会告警。 |
| `--benches`, `--dest` | 选择基准子集（`aime,gpqa,hle`）与输出目录。 |

HLE 图片只对完整数据集写一次；比例 JSONL 引用同一批图片路径，不复制图片。

> 诚实说明：HLE 上游 parquet（约 274MB，图片包含在内）仍会整包下载；
> 抽样减少的是准备后的 JSONL/图片文件体积与运行期调用次数，而不是上游下载量。

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

通过 `RATIO` 选择预先准备的比例文件（省略 `RATIO` 则使用完整数据集）：

```bash
RATIO=0.2 MODEL="$LLM_EVAL_MODEL" bash llm-eval/gpqa/run.sh
```

上例会使用 `data/gpqa_diamond_r0.2.jsonl`。

`CHECK` 仍用于选择判题方式：为空 => 默认（AIME/GPQA 用 `rule`，HLE 用 `llm`）；
`CHECK=llm` 可让 AIME/GPQA 走 LLM 判题。

每次运行都会得到**独立的输出目录**：

```text
outputs/<bench>/<model>-n<N>-r<ratio|full>-<UTC timestamp>/
```

可用 `TAG=my-run` 覆盖该目录名。仍可用 `DATA_FILE` 显式指定数据文件。

也可以直接分别调用两个阶段：

```bash
python llm-eval/run_eval.py --bench hle --data-file llm-eval/data/hle_test.jsonl \
    --model "$LLM_EVAL_MODEL" --n-sample 1 --text-only

python llm-eval/score.py --out-dir llm-eval/outputs/hle/<tag> --check llm
```

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
| `--n-sample` | 每题采样次数（默认：aime 32 / gpqa 8 / hle 1）。 |
| `--temperature`, `--top-p`, `--max-tokens` | 原样透传。 |
| `--reasoning-effort` | 仅在设置时发送（OpenAI：`low`/`medium`/`high`）。 |
| `--seed` | 基础种子；每个请求使用 `seed + index`（尽力而为，取决于端点）。 |
| `--concurrency` | 并发请求数（默认 16）。 |
| `--base-url`, `--api-key`, `--model` | 覆盖 `LLM_EVAL_*` 环境变量。 |

运行期不再做抽样；`meta.json` 中的 `sample_ratio` 来自数据旁车 manifest，仅用于溯源。

`score.py`：`--check rule|llm`（默认：`rule`，HLE 除外，HLE 使用 `llm`），
`--judge auto|rule|llm`（为兼容保留；`--check` 优先生效），
`--judge-concurrency`、`--data-file`。

**采样比例建议（数据准备阶段的 `--sample-ratios`）**

| 比例 | 用途 |
| :--- | :--- |
| `0.01` | 仅用于冒烟/存活性检查（GPQA≈2 题，AIME 因 `max(1)` 下限=1 题，HLE≈25 道题，默认含图题；纯文本需加 `--no-images`）；统计上无法用于准确率回归（置信区间约 ±50%）。 |
| `0.1` | 粗筛，只能发现明显的大幅下跌。 |
| `0.2` | 推荐的准确率回归检查比例。 |

### 快速回归检查

- 使用预生成的比例文件：`RATIO=0.2 MODEL=... bash llm-eval/gpqa/run.sh`。
- 推荐比例 `0.2`：GPQA 是主要探针（198 题，约选中 40 题），使用 rule 判题 => 零判题调用。
- AIME 接近上限（约 99.9%），只能作为冒烟信号。
- HLE 需要 LLM 判题（以及判题凭据）。
- 量化与非量化对比必须使用**同一个比例文件 + 同一环境**。

### 运维说明 / 已接受的风险

- `meta.json.expected_tasks` 是实际参与本次运行的题目数，而非完整数据集大小。
- 运行输出目录默认带 UTC 时间戳，天然隔离；如自定义 `TAG` 请自行避免覆盖。
- `RATIO` 必须与 `prepare_data.py` 生成的文件拼写一致（例如用 `0.1` 而不是 `0.10`）；省略 `RATIO` 即使用完整数据集。
- 同时使用 `RATIO` 与 `DATA_FILE` 时请保持二者一致，否则输出目录名中的比例标签可能与实际数据不符（分数仍按真实数据计算）。
- 时间戳精度为秒；同一秒内并发运行相同配置可能撞目录。本工具按"每轮串行"使用，串行多轮不受影响。
- 比例副本与完整文件由 `prepare_data.py` 维护，同一比例重复生成会覆盖对应 JSONL 与旁车 manifest。
- 代码支持 GPQA 数据集与 HLE 图片/视觉路径，但本阶段仅做过 mock 测试。

## 4. 输出

`llm-eval/outputs/<bench>/<tag>/`：

| 文件 | 内容 |
| :--- | :--- |
| `meta.json` | 运行配置（不含机密）。 |
| `raw_rank000.jsonl` | 模型原始输出。 |
| `scored.jsonl` | 每次 rollout 的抽取答案与判定结果。 |
| `summary.json` | `Pass@1`、`Pass@k`、覆盖率与诊断信息。 |
| `report.html` | 自包含 HTML 报告（内联 CSS，无外部资源/脚本），每次评分运行都会生成。 |

`report.html` 包含指标卡片、Pass@k、诊断信息、`<details>` 中的逐题准确率，以及题目 rollout
（原始输出摘录 ≤200 字符）。重新评分会重新生成该报告；报告生成失败只告警，不会中断评分。

只有当 `summary.json.coverage_complete` 为 `true` 时，`Pass@1` 才有意义。

## 5. 目录结构

```text
llm-eval/
├── prepare_data.py     # ModelScope 全量下载 + 转换 + 确定性比例副本（手动，一次性）
├── run_eval.py         # OpenAI 兼容生成
├── score.py            # 抽取、rule/LLM 判题、Pass@k、report.html
├── common/             # client, io, loader, paths, passk, judge
├── aime/ gpqa/ hle/     # bench.py（prompt/抽取/评分）+ run.sh
└── data/ outputs/      # 本地生成，已被 git 忽略
```

## 工程参考

本项目抽取自 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 的 `eval/` 评测框架，并改造为仅依赖 OpenAI 兼容 API 的独立模型评测工具。
