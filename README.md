# llm-eval

一个可以单独拷走使用的**模型质量评测工具**：只用 OpenAI 兼容的 `/chat/completions`
调用被测模型，生成答案后打分。内置三个基准：**AIME 2026、GPQA Diamond、HLE**。

> 以下命令都在**本仓库根目录**执行（克隆后的 `llm-eval` 目录）。需要 Python 3.11+。
> 如果你把本项目嵌在别的仓库里，请自行给相对路径加上所在目录前缀。

## 核心概念

- **任务（task）是顶层单位**：一次评测 = 一个任务 = 一个输出目录 `outputs/<task>/`。
- 一个任务可以包含 **1 个或多个测试集**（test set）；测试集数量由任务决定。
- 生成和评分都是"任务级"：**一次命令跑完所有测试集**，最后只产出一份 `summary.json`
  和一份 `report.html`。
- 报告内部**按测试集分开展示**；只有多个测试集时才会额外显示一个"总体"汇总。

---

## 最快上手（5 分钟，单测试集）

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

### 4. 跑一个任务（GPQA 20%）

```bash
TEST_SETS="gpqa:data/gpqa_diamond_r0.2.jsonl" \
TASK=my-gpqa-run N_SAMPLE=1 bash run.sh
```

> `TEST_SETS` 的格式是 `bench:数据文件`，多个测试集用空格分隔。
> GPQA 默认用确定性的 `rule` 判分，**不需要判题模型**。

### 5. 看结果

终端应看到类似输出：

```text
[task] task=my-gpqa-run model=gpt-4o test_sets=gpqa:data/gpqa_diamond_r0.2.jsonl ...
...
gpqa  <model>  judge=rule
tasks=40  rollouts=40  n_sample=1
  Pass@1     =  93.40
...
overall: test_sets=1 tasks=40 rollouts=40 Pass@1=93.40% complete=True
done: .../outputs/my-gpqa-run/summary.json
```

用浏览器打开 `outputs/my-gpqa-run/report.html` 即可看到完整报告。

---

## 多测试集任务（一个任务、一份报告）

```bash
TEST_SETS="aime:data/aime_2026.jsonl gpqa:data/gpqa_diamond_r0.1.jsonl" \
TASK=qwen-regression N_SAMPLE=1 MAX_TOKENS=8192 bash run.sh
```

- 所有测试集在**同一个任务目录** `outputs/qwen-regression/` 下生成与评分。
- `summary.json` 里同时有 `overall`（合并）和 `test_sets`（每个测试集单独结果）。
- `report.html` 会依次展示：`总体`（仅多测试集时）、`测试集：aime`、`测试集：gpqa`。

---

## 你会得到什么

- **Pass@1**：平均每题正确率（常用主指标）。
- **coverage_complete**：是否所有题/样本都跑完。**只有为 `true` 时 Pass@1 才可信**。
- **report.html**：一份中文自包含报告；多测试集时按测试集分节展示。
- **scored.jsonl**：每条 rollout 带 `test_set`，便于排查与配对比较。

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
TEST_SETS="gpqa:data/gpqa_diamond.jsonl" TASK=gpqa-full bash run.sh
```

### 跑 0.01 / 0.1 / 0.2

副本已由 `prepare_data.py` 生成，直接在 `TEST_SETS` 里选文件：

```bash
TEST_SETS="gpqa:data/gpqa_diamond_r0.01.jsonl" TASK=gpqa-001 N_SAMPLE=1 bash run.sh   # 冒烟
TEST_SETS="gpqa:data/gpqa_diamond_r0.1.jsonl"  TASK=gpqa-01  N_SAMPLE=1 bash run.sh   # 粗筛
TEST_SETS="gpqa:data/gpqa_diamond_r0.2.jsonl"  TASK=gpqa-02  N_SAMPLE=1 bash run.sh   # 推荐回归
```

### 追加一个 0.5 比例

```bash
python prepare_data.py --benches gpqa --sample-ratios 0.01,0.1,0.2,0.5
TEST_SETS="gpqa:data/gpqa_diamond_r0.5.jsonl" TASK=gpqa-05 N_SAMPLE=1 bash run.sh
```

本地已有全量时，追加比例**不会重新下载**，直接派生。

### HLE（需要判题模型）

```bash
python prepare_data.py --benches hle          # 约 274MB，含图片
export EVAL_JUDGE_BASE_URL=...
export EVAL_JUDGE_API_KEY=...
export EVAL_JUDGE_MODEL=...

TEST_SETS="hle:data/hle_test_r0.1.jsonl" TASK=hle-01 bash run.sh
```

只测文本（跳过图片题，数据也更小）：

```bash
python prepare_data.py --benches hle --no-images
TEST_SETS="hle:data/hle_test_r0.2.jsonl" TASK=hle-text-02 bash run.sh
```

### 换裁判重新评分（不重新生成）

```bash
EVAL_JUDGE_BASE_URL=... EVAL_JUDGE_API_KEY=... EVAL_JUDGE_MODEL=... \
python score.py --out-dir outputs/<task> --check llm
```

### 单测试集快捷脚本

`aime/run.sh`、`gpqa/run.sh`、`hle/run.sh` 仍然保留，内部等价于一个只含该测试集的任务：

```bash
RATIO=0.2 N_SAMPLE=1 bash gpqa/run.sh
```

### 自定义数据文件

数据格式为 JSONL，最少包含 `task_id`、`question`、`answer`（GPQA 的 `answer` 为字母 A–D）：

```bash
TEST_SETS="gpqa:/path/to/my.jsonl" TASK=my-data N_SAMPLE=1 bash run.sh
```

---

## 判题方式

每个测试集按自己的基准选择默认判题方式：

| 基准 | 默认 | 说明 |
| :--- | :--- | :--- |
| AIME / GPQA | `rule` | 确定性抽取 + 比较，不调用判题模型 |
| HLE | `llm` | 答案多为自由文本，必须用判题模型 |

- `CHECK=llm`：让所有测试集都走 LLM 判题。
- 报告会给出每个测试集的 `rule` 与 `llm` 一致率（`rule_vs_llm_agreement`）。

---

## 采样比例怎么选

| 比例 | GPQA（198 题） | 用途 |
| :--- | ---: | :--- |
| `0.01` | ≈2 题 | 只验证链路，不能判断精度 |
| `0.1` | ≈20 题 | 只能发现明显大跌 |
| `0.2` | ≈40 题 | **推荐的精度回归比例** |

量化前后对比时，务必使用**同一批数据文件**、同一套参数、同一环境。

---

## 输出目录（任务级）

```text
outputs/<task>/
├── meta.json          # 任务配置 + test_sets 列表
├── raw_rank000.jsonl  # 每行带 test_set
├── scored.jsonl       # 每行带 test_set
├── summary.json       # overall + 每个 test_set 的汇总
└── report.html        # 一份报告，按测试集分节
```

可用 `TASK=my-run` 指定任务名（即目录名）；省略时按 `<model>-<UTC>` 自动生成。

历史结果可直接删除 `outputs/`（不影响 `data/`）；`data/` 是准备数据，删除后需重新运行 `prepare_data.py`。

---

## 参数速查

### `run.sh`（任务级）

| 变量 | 含义 |
| :--- | :--- |
| `TEST_SETS` | 必填，`bench:path` 列表，如 `aime:data/aime_2026.jsonl gpqa:data/gpqa_diamond_r0.1.jsonl` |
| `TASK` | 任务名/输出目录名；默认 `<model>-<UTC>` |
| `MODEL` | 被测模型名（也可用 `LLM_EVAL_MODEL`） |
| `N_SAMPLE` | 每题采样次数；单测试集默认用基准默认值，多测试集默认 1 |
| `TEMPERATURE` | 默认 `1.0`；复现实验建议 `0` |
| `MAX_TOKENS` | 透传给端点；默认 32768 |
| `LIMIT` | 每个测试集只跑前 N 题 |
| `CONCURRENCY` | 并发请求数（默认 16） |
| `REASONING_EFFORT` | 可选，透传（如 `low`/`medium`/`high`） |
| `TEXT_ONLY` | `1`=跳过所有测试集的图片题 |
| `CHECK` | 空=按测试集默认判题；`llm`=全部走 LLM 判题 |

### `prepare_data.py`

| 选项 | 含义 |
| :--- | :--- |
| `--benches` | `aime,gpqa,hle` 的子集 |
| `--sample-ratios` | 要生成的比例副本，默认 `0.01,0.1,0.2`；全量始终保留 |
| `--no-images` | 跳过 HLE 图片题（先滤图，再派生比例） |
| `--force` | 用本地 raw 缓存重建，不重新下载 |
| `--allow-partial` | 行数不符时只告警（manifest 记 `partial:true`） |
| `--dest` | 数据输出目录，默认 `data` |

### `run_eval.py` / `score.py`（高级用法）

```bash
python run_eval.py --task my-task \
  --test-set aime:data/aime_2026.jsonl \
  --test-set gpqa:data/gpqa_diamond_r0.1.jsonl \
  --model "$LLM_EVAL_MODEL" --n-sample 1 --temperature 0

python score.py --out-dir outputs/my-task
```

- `run_eval.py` 还支持旧参数 `--bench` + `--data-file`（等价于单个测试集）。
- `score.py`：`--check rule|llm`、`--judge-concurrency`；`--data-file` 仅用于单测试集覆盖。

---

## 故障排查

| 现象 | 原因 / 解决 |
| :--- | :--- |
| `ModuleNotFoundError: modelscope` / `datasets` | 未装数据依赖：`pip install -r requirements-data.txt` |
| `--test-set expects BENCH:FILE` | `TEST_SETS` 项写成了 `path`，应为 `bench:path`（如 `gpqa:data/gpqa_diamond_r0.2.jsonl`） |
| `FileNotFoundError: ..._r0.2.jsonl` | 该比例未生成：先 `python prepare_data.py --sample-ratios 0.2` |
| `401` / `403` | 检查 `LLM_EVAL_BASE_URL`、`LLM_EVAL_API_KEY`（HLE 还要检查 `EVAL_JUDGE_*`） |
| `coverage_complete=false` | 有请求失败或被截断：看 `summary.json` 对应测试集的 `finish_reasons` |
| HLE 下载很慢/很大 | 正常，HLE 原始包约 274MB（含图片）；可先用 `--benches aime,gpqa` |
| HLE 判题解析失败 | 判题模型太弱或输出被截断：换更强裁判，或提高其输出预算 |
| 数据想重新下载 | 删除 `data/_raw/<bench>` 后重跑 `prepare_data.py` |

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
├── run.sh              # 任务级入口：一个任务跑 1..N 个测试集
├── prepare_data.py     # ModelScope 全量下载 + 转换 + 确定性比例副本
├── run_eval.py         # OpenAI 兼容生成（支持多测试集）
├── score.py            # 抽取、rule/LLM 判题、Pass@k、report.html
├── common/             # client, io, loader, paths, passk, judge, report
├── aime/ gpqa/ hle/     # bench.py（prompt/抽取/评分）+ 单测试集 run.sh
└── data/ outputs/      # 本地生成，已被 git 忽略
```

## 工程参考

本项目抽取自 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 的 `eval/` 评测框架，并改造为仅依赖 OpenAI 兼容 API 的独立模型评测工具。
