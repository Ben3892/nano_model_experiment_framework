"""一个小型、完整的 decoder-only GPT 模型。

模型结构如下：

    token embedding + position embedding
                    ↓
       N × TransformerBlock
                    ↓
              final RMSNorm
                    ↓
                 LM head

每个 TransformerBlock 都使用 Pre-Norm 结构：先归一化，再依次执行因果
自注意力和 SwiGLU 前馈网络，并在两个子层外添加残差连接。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """GPT 的结构和正则化配置。

    默认值刻意保持较小，便于在单卡或 CPU 上做教学实验。实际使用时至少应让
    ``vocab_size`` 与 tokenizer 的词表大小保持一致。
    """

    vocab_size: int = 50_304
    max_seq_len: int = 1_024
    num_layers: int = 6
    num_heads: int = 6
    hidden_size: int = 384
    intermediate_size: Optional[int] = None
    dropout: float = 0.0
    bias: bool = False
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    init_std: float = 0.02

    def __post_init__(self) -> None:
        """补全派生配置，并尽早暴露不合法的超参数。"""
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

        positive_fields = {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} 必须大于 0，当前值为 {value}")

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size 必须能被 num_heads 整除")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 区间")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps 必须大于 0")
        if self.init_std <= 0:
            raise ValueError("init_std 必须大于 0")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization。

    归一化计算临时提升到 float32，以减小 fp16/bf16 下的数值误差；输出仍恢复
    为输入 dtype。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        normalized = x.float()
        normalized = normalized * torch.rsqrt(
            normalized.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        # 参数可能仍是 float32（例如 autocast 场景），最后显式恢复输入 dtype。
        return (self.weight.float() * normalized).to(input_dtype)


class CausalSelfAttention(nn.Module):
    """带因果掩码的多头自注意力。

    因果掩码保证位置 ``t`` 只能读取位置 ``<= t`` 的 token，从而满足自回归
    语言模型的训练要求。PyTorch 2.0+ 会自动使用高效的 SDPA/Flash Attention
    内核；旧版本则回退到显式注意力实现。
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout = config.dropout

        # 一次线性投影同时计算 Q、K、V，减少小算子的调用开销。
        self.qkv_proj = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=config.bias
        )
        self.out_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.bias
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, hidden_size = x.shape

        qkv = self.qkv_proj(x).view(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        # (B, T, H, D) -> (B, H, T, D)，让注意力头成为独立批次维度。
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:  # pragma: no cover - 仅供 PyTorch 2.0 以下版本使用
            scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal_mask = torch.ones(
                seq_len, seq_len, device=x.device, dtype=torch.bool
            ).tril()
            scores = scores.masked_fill(~causal_mask, float("-inf"))
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = F.dropout(
                attention_weights, p=self.dropout, training=self.training
            )
            attended = attention_weights @ value

        # 重新拼接所有注意力头：(B, H, T, D) -> (B, T, C)。
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )
        return self.residual_dropout(self.out_proj(attended))


class SwiGLU(nn.Module):
    """SwiGLU 前馈网络，比单一激活函数多一条可学习的门控分支。"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        intermediate_size = config.intermediate_size
        if intermediate_size is None:  # __post_init__ 已补全；用于静态类型收窄。
            raise ValueError("intermediate_size 不能为空")

        self.gate_proj = nn.Linear(
            config.hidden_size, intermediate_size, bias=config.bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, intermediate_size, bias=config.bias
        )
        self.down_proj = nn.Linear(
            intermediate_size, config.hidden_size, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))


class TransformerBlock(nn.Module):
    """一个 Pre-Norm Transformer 解码器层。"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, x: Tensor) -> Tensor:
        # Pre-Norm + residual：深层网络通常比 Post-Norm 更容易稳定训练。
        x = x + self.attention(self.attention_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GPT(nn.Module):
    """用于 next-token prediction 的小型 GPT 语言模型。

    输入是形状 ``(batch_size, sequence_length)`` 的 token id，输出是形状
    ``(batch_size, sequence_length, vocab_size)`` 的未归一化 logits。损失计算
    留给训练代码处理，模型本身保持简单且易于复用。
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_seq_len, config.hidden_size
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        # 词嵌入与输出分类器共享权重，可减少参数并常常改善语言建模效果。
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        """采用 GPT 风格的正态分布初始化。"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def _scale_residual_projections(self) -> None:
        """按残差层数缩小输出投影初始化，避免残差流方差持续增长。"""
        std = self.config.init_std / math.sqrt(2 * self.config.num_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.out_proj.weight, mean=0.0, std=std)
            nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=std)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids 必须是 (batch_size, sequence_length) 的二维张量"
            )
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids 必须包含整数 token id")

        _, seq_len = input_ids.shape
        if seq_len == 0:
            raise ValueError("sequence_length 必须大于 0")
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"输入长度 {seq_len} 超过 max_seq_len={self.config.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=input_ids.device)
        hidden_states = self.token_embedding(input_ids)
        hidden_states = hidden_states + self.position_embedding(positions)
        hidden_states = self.embedding_dropout(hidden_states)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.final_norm(hidden_states)
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        """使用温度采样和可选 top-k 过滤自回归生成 token。"""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens 不能小于 0")
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        for _ in range(max_new_tokens):
            # 绝对位置嵌入只能覆盖 max_seq_len，因此仅保留最近的上下文。
            context = input_ids[:, -self.config.max_seq_len :]
            next_token_logits = self(context)[:, -1, :] / temperature

            if top_k is not None:
                k = min(top_k, next_token_logits.size(-1))
                threshold = torch.topk(next_token_logits, k).values[:, [-1]]
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < threshold, float("-inf")
                )

            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)

        return input_ids

    def num_parameters(self, exclude_position_embeddings: bool = False) -> int:
        """返回参数量；可选不统计位置嵌入，便于比较不同上下文长度。"""
        count = sum(parameter.numel() for parameter in self.parameters())
        if exclude_position_embeddings:
            count -= self.position_embedding.weight.numel()
        return count


# 兼容原文件对外暴露的类名，同时推荐新代码使用语义更明确的 GPT。
Transformer = GPT
TransformerConfig = GPTConfig


__all__ = [
    "GPTConfig",
    "GPT",
    "TransformerConfig",
    "Transformer",
    "RMSNorm",
    "CausalSelfAttention",
    "SwiGLU",
    "TransformerBlock",
]
