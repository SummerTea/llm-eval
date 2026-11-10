# GPQA Standalone

单目录、零第三方依赖；Python 3.11+。`gpqa_diamond.jsonl` 必须与 `gpqa_single.py` 同级。

## 通用测试配置（推荐固定使用）

以下命令是内网实测通过的稳定配置，除 `<host>` / `<model>` / `<key-or-dummy>` 外不要改：

```bash
NO_PROXY=<host> python3 gpqa_single.py \
  --base-url http://<host>:8000/v1 --api-key <key-or-dummy> \
  --model <model> \
  --sample 20 --sample-seed 20260101 --n-sample 1 \
  --temperature 0 --max-tokens 32768 \
  --concurrency 4 --timeout 300 --retries 1 --verbose
```

参数固定理由：

- `--sample 20 --sample-seed 20260101`：确定性子集，等价主仓库 r0.1，每次同一批题；本配置指纹为 `ids_sha256=ee1c465fe1dc`，报告可直接核对。
- `--n-sample 1 --temperature 0`：固定输出口径，便于横向比较。
- `NO_PROXY=<host>`：仅在有 `http_proxy` / `https_proxy` 的环境下必要；无代理可省略。
- `--max-tokens 32768`：内网 vLLM 已验证可接受；降到 4096 会触发 `length` 截断。
- `--concurrency 4 --timeout 300 --retries 1`：20 题实测约 3 分钟。
- `--verbose`：打印全部未通过题，便于定位。

## 其他场景

- 冒烟 2 题：`--sample 2 --max-tokens 4096 --concurrency 1 --timeout 60`
- 回归 40 题：`--sample 40 --timeout 600`
- 全量 198 题：`--sample 198 --timeout 600`

## 口径

- 仅 rule 判分，不需要裁判模型。
- 本工具默认 `n_sample=1`、`temperature=1.0`；主仓库 `gpqa` 默认 `n_sample=8`，分数不可直接对表。
- 选题确定性：同一 `--sample` + `--sample-seed` 每次同一批题；报告 `ids_sha256` 可核对。
- 退出码：0 达标；1 参数/数据错误；2 请求失败、覆盖不完整或低于 `--min-pass1`。
- 参考量级：Qwen3.5-2B 在上述 20 题配置下 Pass@1 50%（10/20，n=1）；仅作链路/量级参考，不是客户质量结论。
