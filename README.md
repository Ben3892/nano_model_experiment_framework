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

## 后续实现

- 接入 verl，记录 rollout、reward 和训练日志，先跑通可复现的 RL 实验。
- 增加 OPD 训练，记录 teacher/student、轨迹来源、分词对齐和蒸馏损失配置。
