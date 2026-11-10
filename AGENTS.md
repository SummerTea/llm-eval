# AGENTS.md

独立的模型质量评测 CLI：只通过 OpenAI 兼容 `/chat/completions` 调用被测模型，两阶段执行
（`run_eval.py` 生成 → `score.py` 评分），基准为 AIME / GPQA Diamond / HLE。

## 常用命令

```bash
pip install -r requirements.txt            # 运行依赖（requests, pillow）
pip install -r requirements-data.txt       # 仅准备数据时（modelscope, datasets）
python prepare_data.py --benches gpqa      # 生成 data/*.jsonl 与比例副本
TEST_SETS="gpqa:data/gpqa_diamond_r0.2.jsonl" TASK=my-run N_SAMPLE=1 bash run.sh
python score.py --out-dir outputs/<task> --check llm   # 只换裁判重评，不重新生成
```

- Python 3.11+（`common/report.py` 使用 `datetime.UTC`）。
- 仓库**没有测试套件、lint、格式化或 CI 配置**。改动后的最小验证：`python -m compileall .`，
  外加一次 GPQA `r0.01` 冒烟（需要真实端点和 `data/`）。

## 必知机制

- `data/`、`outputs/` 被 git 忽略，仓库不含数据集；任何评测前先跑 `prepare_data.py`。
- 两套凭据严格分离、互不回退：被测 `LLM_EVAL_BASE_URL/API_KEY/MODEL`，裁判
  `EVAL_JUDGE_BASE_URL/API_KEY/MODEL`（HLE 或 `CHECK=llm` 才需要）。
- 判题默认：AIME/GPQA 走确定性 `rule`，HLE 强制 `llm`；对 HLE 指定 `rule` 会直接抛错退出。
- `N_SAMPLE` 默认值取决于测试集数量：单测试集用基准默认（aime 32 / gpqa 8 / hle 1），多测试集恒为 1。
- 同一 `TASK` 重跑会先删除 `outputs/<task>/raw_rank*.jsonl`，旧生成结果不可恢复，只能重新生成。
- `coverage_complete=false` 的 Pass@1 不可信；判分结果看 `summary.json` 的 `finish_reasons` / `judge_unparsed`。
- `score.py` 依赖 `outputs/<task>/meta.json`，只能评分已完成的生成任务。
- 比例副本由固定种子（`SAMPLE_SEED=20260101`）按 `task_id` 哈希确定性抽取，是物理文件，
  运行期只选文件不重抽样；`--force` 只重建转换产物，**不重新下载**（删 `data/_raw/<bench>` 才会重下）。
- 数据旁车 `*.jsonl.meta.json` 记录 `sample_ratio` 与完整性哈希；`run_eval.py` 从它读 `sample_ratio`，
  手工改动数据文件会破坏复用校验。

## 新增基准的注册点（容易漏）

新建 `<bench>/bench.py` 实现 `load` / `build_records` / `score_rollout`，并同步修改：

1. `common/loader.py` 的 `SUPPORTED`
2. `run_eval.py` 的 `BENCH_DEFAULTS`
3. `score.py` 的 `BENCH_DEFAULT_JUDGE`
4. `prepare_data.py` 的 `DATASETS` / `EXPECTED_ROWS` / `OUTPUT_NAME` / `CONVERTERS`

`bench.py` 由 `common/loader.py` 按文件路径动态导入为 `eval_bench_<name>`，只能靠各入口手动插入
`sys.path` 后做绝对导入（`from common.io import ...`），不能用相对导入。可选再加 `<bench>/run.sh`。

## 约定

- `run.sh` 用环境变量（`TEST_SETS="bench:path"` 空格分隔），`run_eval.py` 用可重复的 `--test-set BENCH:FILE`。
- 报告 `report.html` 与 README 面向中文读者；代码注释用英文。
