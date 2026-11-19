# Nano Model Experiment Framework

用同一个入口运行小模型预训练和 SFT，验证数据、下一 token 损失、验证集评估和 checkpoint 流程。PyTorch 后端复用仓库的 Dense GPT，Pai-Megatron-Patch 后端使用上游 Qwen3 预训练入口构建小型稠密模型。内置实验使用分开的合成训练集和验证集，无需下载模型权重或语料。

## PyTorch CPU 实验

在仓库根目录运行。已有 PyTorch 2.6 或更新版本时可以直接执行训练命令。

```bash
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu

python -m src.train --stage pretrain --output runs/torch-pretrain
python -m src.train --stage sft --init-from runs/torch-pretrain/checkpoint.pt \
  --output runs/torch-sft

python -m unittest discover -s tests -v
```

默认每阶段训练 40 步，序列长度 64，global batch size 4，学习率 0.001，seed 42。可通过 `--steps`、`--seq-len`、`--batch-size`、`--lr`、`--seed` 和 `--eval-batches` 调整。SFT 的 `--init-from` 加载同后端的预训练权重并初始化新的优化器。PyTorch 的可学习位置嵌入要求两阶段的序列长度一致。省略 `--init-from` 可单独检查 SFT 的数据和训练路径。

在 PyTorch 2.7.1 CPU 上运行默认实验得到以下合成验证集 loss。

| 阶段 | 初始 | 40 步后 |
| --- | ---: | ---: |
| 预训练 | 5.5799 | 1.4930 |
| SFT，加载预训练权重 | 2.5277 | 0.0341 |

## Pai-Megatron-Patch 单卡实验

使用支持 BF16 的 NVIDIA GPU，例如 A800，以及包含 CUDA、NCCL、Triton、Transformer Engine 和编译工具的 Pai-Megatron-Patch 环境。此例的 tokenizer 使用 Transformers 4.57.6，JSON-SFT 加载器需要 Hugging Face Datasets。[上游 Qwen3 指南和 PAI 25.04 镜像](https://github.com/alibaba/Pai-Megatron-Patch/blob/a098ca5acbdeaf7172ce0393fa309b39a77506db/examples/qwen3/README.md)

```bash
git clone https://github.com/alibaba/Pai-Megatron-Patch.git ../Pai-Megatron-Patch
git -C ../Pai-Megatron-Patch checkout a098ca5acbdeaf7172ce0393fa309b39a77506db
git -C ../Pai-Megatron-Patch submodule update --init --depth 1 backends/megatron/Megatron-LM-250624

CUDA_VISIBLE_DEVICES=0 python -m src.train --backend megatron-patch \
  --patch-root ../Pai-Megatron-Patch --stage pretrain --steps 4 --output runs/patch-pretrain

CUDA_VISIBLE_DEVICES=0 python -m src.train --backend megatron-patch \
  --patch-root ../Pai-Megatron-Patch --stage sft --steps 4 \
  --init-from runs/patch-pretrain/checkpoint --output runs/patch-sft
```

适配器通过 `torch.distributed.run` 调用上游 `examples/qwen3/pretrain_qwen.py`，使用两个 Transformer 层、hidden size 128 和 TP/PP/CP=1。预训练输入通过 Megatron-Core 写为 `.bin/.idx`。SFT 用 Patch 的 JSON-SFT 加载器读取转换后的 `query` 和 `response`，并应用本地模板。两种后端共享语料与 SFT loss mask 语义，各自使用自己的模型结构、采样实现和 checkpoint 格式。

无 GPU 时可加入 `--prepare-only` 生成 tokenizer、数据和 `launch.json` 中的实际参数与环境变量。该模式的 `summary.json` 标记为 `prepared`。已在 CPU 上检查上游参数和两阶段数据读取，GPU 训练、分布式 checkpoint 保存和 SFT 热启仍需在 CUDA 环境验证。

## 自定义数据和产物

成对传入 `--train-data` 和 `--valid-data`。预训练输入是 UTF-8 文本，每个非空行作为一篇文档并追加 EOS。SFT 输入是 UTF-8 JSONL，每行包含 `prompt` 和 `response` 字符串，例如 `{"prompt":"颜色是什么？","response":"蓝色"}`。SFT 对 response 和 EOS 计算损失，prompt 与 padding 均被 mask。按 UTF-8 字节计算长度，超过 `--seq-len` 时会报告对应行号。

每次实验使用新输出目录，保留训练和验证输入快照、`config.json`、`metrics.jsonl`、`summary.json` 及 checkpoint。Patch 训练另外保留 `megatron.log` 和 `launch.json`。两阶段分别在各自的验证协议下统计 loss。

## RSI 递归自训练

使用 Python 3.11 或更新版本。CPU demo 先用固定的算术样本训练种子模型，再用当前晋级 checkpoint 采样。控制器用整数答案判分，混入 replay，复用现有训练入口做下一轮 SFT，并在独立的 dev 集上决定是否晋级。

```bash
python -m src.rsi.run_round --demo --rounds 2 --output runs/rsi-demo
python -m src.rsi.run_round --demo --rounds 2 --generator-mode frozen --output runs/rsi-frozen
```

`frozen` 始终用种子权重生成数据，各轮 SFT 仍从该实验臂当前晋级权重开始。对照时核对两臂的实际训练和采样预算，以及触发提前停止的轮次。内置题目只有个位数加零和分开的标签，适合验证流程与 checkpoint 流转。

使用 PyTorch 2.7.1 CPU、固定配比和 seed 42 各跑四轮，两组各消耗 29,440 个训练 token 位置、7,680 个采样 token 位置。合成盲测的 `pass@1` 从种子模型的 15.00% 提升至递归组的 83.75% 和冻结生成器组的 71.25%。盲测仅含 10 道模板题。另一次把晋级门槛调至 20 个百分点，后两轮未达标，控制器停止并保留第二轮权重。

自定义实验通过 `--data-dir DIR --init-from WEIGHTS.pt` 指定初始 Torch checkpoint 和数据。该 checkpoint 的模型结构和 `--seq-len` 应与 Torch 训练入口一致。目录下有五个 JSONL 文件，每行形如 `{"id":"q01","family":"signed-add","prompt":"-3 + 2 = ?","answer":-1}`。同一个 ID、规范化 prompt 或题族不得跨 split 重用。

| 文件 | 用途 |
| --- | --- |
| `rollout.jsonl` | 采样 prompt，答案用于外部整数验证 |
| `replay.jsonl` | 固定的高质量监督样本 |
| `monitor.jsonl` | 训练期间记录 SFT validation loss |
| `dev.jsonl` | 每轮晋级判定 |
| `blind.jsonl` | 停止后比较初始与最终选定 checkpoint |

判分接受整数，或最后一行形如 `Final: -1` 的完整回复。采样序列需要 EOS，长回复需增大 `--max-new-tokens` 并留出上下文空间。每题最多入选两条不同的正确回复。触及新数据 token 上限时优先覆盖部分采样正确的题。Replay 比例按入库的有效监督 token 估算，`decision.json` 记录实际入库比例。

晋级要求 dev 的 `pass@1` 增益达到 `--min-gain`，按题配对 bootstrap 的置信区间下界大于零，各题族回退低于 `--max-regression`。评估同时记录 `pass@k` 和原本稳定题的退化情况。用 `--eval-samples 32` 或更高计算 `pass@32`。当正样本不足、连续未晋级、轮数或 token 预算触顶时停止。预算覆盖初始评估、训练轮次和最后的盲测。训练失败会保留此前晋级的 `best.json`。

使用 DSH 时，先在运行环境中配置 DSH CLI 和模型凭据，例如默认 DeepSeek API 路由使用 `DEEPSEEK_API_KEY`。Headless 模式接收聚合指标，通过 stdout 返回 JSON。控制器仅采纳 0.1–0.5 范围内的 `replay_fraction`，超时或无效 JSON 时沿用本轮比例。[DeepSeek Harness headless CLI](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/cli-help.md#dsh-headless)

```bash
python -m src.rsi.run_round --demo --rounds 3 --planner dsh --output runs/rsi-dsh
```

`run.json` 记录配置和数据哈希，`round_*/decision.json` 保存候选、生成器、父 checkpoint、入库统计与晋级依据，`best.json` 指向已晋级模型，`summary.json` 汇总 token 预算和盲测结果。RSI 的训练和采样当前使用 Torch CPU 后端。Pai-Megatron-Patch 可用现有独立训练入口，接入 RSI 还需要对接可供 rollout 使用的 Megatron checkpoint。

## 后续实现

- 接入 verl，记录 rollout、reward 和训练日志，先跑通可复现的 RL 实验。
- 增加 OPD 训练，记录 teacher/student、轨迹来源、分词对齐和蒸馏损失配置。
