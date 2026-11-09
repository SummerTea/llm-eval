# llm-eval

一个可以单独拷走使用的**模型质量评测工具**：只用 OpenAI 兼容的 `/chat/completions`
调用被测模型，生成答案后打分。内置三个基准：**AIME 2026、GPQA Diamond、HLE**。

> 以下命令都在**本仓库根目录**执行（克隆后的 `llm-eval` 目录）。需要 Python 3.11+。
> 如果你把本项目嵌在别的仓库里，请自行给相对路径加上所在目录前缀。

---

## 最快上手（5 分钟，以 GPQA 为例）

### 1. 安装运行依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据（先只下 GPQA）

```bash
pip install -r requirements-data.txt     # modelscope, datasets
python prepare_data.py --benches gpqa
```

会得到：

- `data/gpqa_diamond.jsonl`：全量 198 题
- `data/gpqa_diamond_r0.01.jsonl`、`_r0.1`、`_r0.2`：三份比例副本

### 3. 配置被测模型

```bash
export LLM_EVAL_BASE_URL=https://api.openai.com/v1
export LLM_EVAL_API_KEY=sk-...
export LLM_EVAL_MODEL=gpt-4o
```

### 4. 跑一次 20% 快检

```bash
RATIO=0.2 N_SAMPLE=1 bash gpqa/run.sh
```

> GPQA 默认用确定性的 `rule` 判分，**不需要判题模型**。

### 5. 看结果

终端应看到类似输出：

```text
[gpqa] model=... n_sample=1 ratio=0.2 check=auto text_only=0 -> .../outputs/gpqa/...
...
gpqa  <model>  judge=rule
tasks=40  rollouts=40  n_sample=1
  Pass@1     =  93.40
coverage: expected 40 rollouts / 40 tasks, got 40 / 40, complete=True
wrote .../outputs/gpqa/<model>-n1-r0.2-<UTC>/report.html
```

用浏览器打开 `report.html` 即可看到完整报告；`summary.json` 是机器可读的汇总。

---

## 你会得到什么

- **Pass@1**：平均每题正确率（常用主指标）。
- **coverage_complete**：是否所有题/样本都跑完。**只有为 `true` 时 Pass@1 才可信**。
- **report.html**：自包含 HTML 报告（指标卡、Pass@k、诊断、逐题正确率、翻车样本），无需联网。
- **scored.jsonl**：每条 rollout 的抽取答案与判定，便于排查。

---

## 前置条件

1. **Python 3.11+**（`report.py` 使用了 `datetime.UTC`）。
2. 能访问 **ModelScope**（国内网络可用）。
3. 一个 **OpenAI 兼容**的模型端点（被测模型）。
4. 需要 LLM 判题时（HLE 必须、AIME/GPQA 可选）：另一个 OpenAI 兼容的**判题模型**端点。

判题与被测是**两套独立凭据，绝不互相回退**：

| 角色 | 环境变量 | 什么时候需要 |
| :--- | :--- | :--- |
| 被测模型 | `LLM_EVAL_BASE_URL` / `LLM_EVAL_API_KEY` / `LLM_EVAL_MODEL` | 始终 |
| 判题模型 | `EVAL_JUDGE_BASE_URL` / `EVAL_JUDGE_API_KEY` / `EVAL_JUDGE_MODEL` | HLE，或 AIME/GPQA 使用 `CHECK=llm` 时 |

---

## 常见任务

### 跑全量

```bash
bash gpqa/run.sh
```

### 跑 0.01 / 0.1 / 0.2

副本已由 `prepare_data.py` 生成，直接用 `RATIO` 选文件：

```bash
RATIO=0.01 N_SAMPLE=1 bash gpqa/run.sh   # 冒烟
RATIO=0.1  N_SAMPLE=1 bash gpqa/run.sh   # 粗筛
RATIO=0.2  N_SAMPLE=1 bash gpqa/run.sh   # 推荐回归
```

### 追加一个 0.5 比例

```bash
python prepare_data.py --benches gpqa --sample-ratios 0.01,0.1,0.2,0.5
RATIO=0.5 N_SAMPLE=1 bash gpqa/run.sh
```

本地已有全量时，追加比例**不会重新下载**，直接派生。

### HLE（需要判题模型）

```bash
python prepare_data.py --benches hle          # 约 274MB，含图片
export EVAL_JUDGE_BASE_URL=...
export EVAL_JUDGE_API_KEY=...
export EVAL_JUDGE_MODEL=...

RATIO=0.2 bash hle/run.sh
```

只测文本（跳过图片题，数据也更小）：

```bash
python prepare_data.py --benches hle --no-images
RATIO=0.2 bash hle/run.sh
```

### AIME

```bash
python prepare_data.py --benches aime
RATIO=0.2 N_SAMPLE=8 bash aime/run.sh
```

### 换裁判重新评分（不重新生成）

```bash
EVAL_JUDGE_BASE_URL=... EVAL_JUDGE_API_KEY=... EVAL_JUDGE_MODEL=... \
python score.py --out-dir outputs/gpqa/<tag> --check llm
```

### 自定义数据文件

数据格式为 JSONL，最少包含 `task_id`、`question`、`answer`（GPQA 的 `answer` 为字母 A–D）：

```bash
DATA_FILE=/path/to/my.jsonl N_SAMPLE=1 bash gpqa/run.sh
```

---

## 判题方式

| 基准 | 默认 | 说明 |
| :--- | :--- | :--- |
| AIME / GPQA | `rule` | 确定性抽取 + 比较，不调用判题模型 |
| HLE | `llm` | 答案多为自由文本，必须用判题模型 |

- `CHECK=llm`：让 AIME/GPQA 也走 LLM 判题。
- 报告会给出 `rule` 与 `llm` 的一致率（`rule_vs_llm_agreement`），用于判断裁判是否可靠。

---

## 采样比例怎么选

| 比例 | GPQA（198 题） | 用途 |
| :--- | ---: | :--- |
| `0.01` | ≈2 题 | 只验证链路，不能判断精度 |
| `0.1` | ≈20 题 | 只能发现明显大跌 |
| `0.2` | ≈40 题 | **推荐的精度回归比例** |

量化前后对比时，务必使用**同一个比例文件**、同一套参数、同一环境。

---

## 输出目录

每轮运行都会生成独立目录（默认带 UTC 时间戳，不会互相覆盖）：

```text
outputs/<bench>/<model>-n<N>-r<ratio|full>-<UTC>/
```

| 文件 | 内容 |
| :--- | :--- |
| `meta.json` | 本次运行配置（不含密钥） |
| `raw_rank000.jsonl` | 模型原始输出 |
| `scored.jsonl` | 抽取答案 + 判定结果 |
| `summary.json` | Pass@1、Pass@k、覆盖率、诊断信息 |
| `report.html` | 自包含 HTML 报告 |

可用 `TAG=my-run` 指定固定目录名（复跑会覆盖同名目录）。

---

## 参数速查

### `prepare_data.py`

| 选项 | 含义 |
| :--- | :--- |
| `--benches` | `aime,gpqa,hle` 的子集 |
| `--sample-ratios` | 要生成的比例副本，默认 `0.01,0.1,0.2`；全量始终保留 |
| `--no-images` | 跳过 HLE 图片题（先滤图，再派生比例） |
| `--force` | 用本地 raw 缓存重建，不重新下载 |
| `--allow-partial` | 行数不符时只告警（manifest 记 `partial:true`） |
| `--dest` | 数据输出目录，默认 `data` |

### `run.sh` 环境变量

| 变量 | 含义 |
| :--- | :--- |
| `RATIO` | 选哪份比例文件；省略=全量 |
| `N_SAMPLE` | 每题采样次数（默认：aime 32 / gpqa 8 / hle 1） |
| `TEMPERATURE` | 默认 `1.0`；复现实验建议 `0` |
| `REASONING_EFFORT` | 可选，透传给端点（如 `low`/`medium`/`high`） |
| `CHECK` | 空=默认判题；`llm`=强制 LLM 判题 |
| `TEXT_ONLY` | `1`=跳过图片题 |
| `TAG` | 自定义输出目录名 |
| `DATA_FILE` | 直接指定数据文件（会覆盖 `RATIO` 推导的路径） |

### `run_eval.py` / `score.py`（高级用法）

`run_eval.py` 关键参数：`--bench`、`--data-file`、`--model`、`--n-sample`、`--temperature`、
`--top-p`、`--max-tokens`、`--reasoning-effort`、`--seed`、`--concurrency`、`--text-only`、`--limit`。
完整列表：`python run_eval.py --help`。

`score.py`：`--out-dir`、`--check rule|llm`、`--judge-concurrency`、`--data-file`。

---

## 故障排查

| 现象 | 原因 / 解决 |
| :--- | :--- |
| `ModuleNotFoundError: modelscope` / `datasets` | 未装数据依赖：`pip install -r requirements-data.txt` |
| `FileNotFoundError: ..._r0.2.jsonl` | `RATIO` 拼写或该比例未生成：用 `0.2` 而非 `0.20`，或先 `python prepare_data.py --sample-ratios 0.2` |
| `401` / `403` | 检查 `LLM_EVAL_BASE_URL`、`LLM_EVAL_API_KEY`（HLE 还要检查 `EVAL_JUDGE_*`） |
| `coverage_complete=false` | 有请求失败或被截断：看 `summary.json` 的 `finish_reasons`、`errors` |
| HLE 下载很慢/很大 | 正常，HLE 原始包约 274MB（含图片）；可先用 `--benches aime,gpqa` |
| HLE 判题解析失败 | 判题模型太弱或输出被截断：换更强裁判，或提高其输出预算 |
| 数据想重新下载 | 删除 `data/_raw/<bench>` 后重跑 `prepare_data.py` |
| 同一秒启动两次同配置 | 目录可能撞名：串行运行，或设置不同的 `TAG` |

---

## 原理与限制

- **两阶段**：先生成并落盘原始输出，再评分。换裁判/改抽取规则时只需重新评分，不必重新生成。
- **采样在数据准备阶段完成**：`prepare_data.py` 下载一次全量，生成固定比例的物理副本；
  运行期只选文件，不再抽样。选择是确定性的，同一比例每次都是同一批题。
- **HLE 上游仍是整包下载**：抽样减少的是准备后的文件体积与运行调用量，不是上游下载量。
- **数据不随仓库分发**：`data/`、`outputs/` 已被 git 忽略。HLE 要求不得再分发；AIME 为非商业用途。
- GPQA 数据集与 HLE 图片/视觉路径在代码中已支持，但本项目仅在 mock 环境验证过。

---

## 目录结构

```text
.
├── prepare_data.py     # ModelScope 全量下载 + 转换 + 确定性比例副本
├── run_eval.py         # OpenAI 兼容生成
├── score.py            # 抽取、rule/LLM 判题、Pass@k、report.html
├── common/             # client, io, loader, paths, passk, judge, report
├── aime/ gpqa/ hle/     # bench.py（prompt/抽取/评分）+ run.sh
└── data/ outputs/      # 本地生成，已被 git 忽略
```

## 工程参考

本项目抽取自 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 的 `eval/` 评测框架，并改造为仅依赖 OpenAI 兼容 API 的独立模型评测工具。
