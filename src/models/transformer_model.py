import torch
from torch import nn
import torch.nn.functional as F

'''
一个基本的moe模型的架构
tokenizer
｜
embedding层
｜
dense层
｜
x * hidden 层
｜
lm head 层 - mtp层

其中最重要的是hidden层，包括：self self attention-> ffn -> residual -> rmsnorm
'''

class Embedding(nn.Module):
    def __init__(self, config):
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

    def forward(self, x):
        return self.embedding(x)

class Expert(nn.Module):
    def __init__(self, config):
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act = nn.SwiGLU()

    def forward(self, x):
        return self.act(self.down_proj(self.up_proj(x)))

class Multi_Head_Attention(nn.Module):
    def __init__(self, config):
        self.hidden_size = config.hidden_size
        self.attention_head = config.attention_head
        self.head_dim = self.hidden_size // self.attention_head
        self.W_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        b, s, h = q.shape

        q_view = q.view(b, self.attention_head, s, self.head_dim)
        k_view = k.view(b, self.attention_head, s, self.head_dim)
        v_view = v.view(b, self.attention_head, s, self.head_dim)
        attention_score = nn.Softmax(q_view @ k_view.transpose(-2, -1) / self.scale, dim=-1)
        output = attention_score @ v_view
        # (b, a, s, s) -> (b, a, s, h//a)
        output = output.transpose(1, 2).view(b, s, h)
        return output

class RMSNorm(nn.Module):
    def __init__(self, config, eps=1e-6):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.hidden_size))

    def forward(self, x):
        input_type = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_type)



class Dense(nn.Module):
    def __init__(self, config):
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.attention = Multi_Head_Attention(config)
        self.ffn = Expert(config)
        self.rmsnorm1 = RMSNorm(config)
        self.rmsnorm2 = RMSNorm(config)

    def forward(self, x):
        x_residual = x
        x = self.rmsnorm1.forward(x)
        attention_output = x_residual + self.rmsnorm2.forward(self.attention.forward(x))
        ffn_output = attention_output + self.ffn.forward(attention_output)
        return ffn_output


class MoE_ffn(nn.Module):
    def __init__(self, config):
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.expert_num = config.expert_num
        self.activate_expert_num = config.activate_expert_num
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.expert_num)])
        self.score_calculater = nn.Linear(self.hidden_size, self.expert_num)

    def forward(self, x):
        scores = self.score_calculater(x)
        topk_scores, topk_indices = torch.topk(scores, k=self.activate_expert_num, dim=-1)
        topk_scores = nn.softmax(topk_scores, dim=-1)



class Moe_layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Multi_Head_Attention(config)
        self.moe = MoE_ffn(config)
        self.rmsnorm1 = RMSNorm(config)
        self.rmsnorm2 = RMSNorm(config)
        self.act_proj = nn.SwiGLU()
    def forward(self, x):
        x = self.rmsnorm1.forward(x)
        x = self.attention.forward(x)
        x = self.rmsnorm2.forward(x)
        x = self.moe.forward(x)
        x = self.act_proj.forward(x)
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config)
        self.dense = nn.ModuleList([Dense(config) for _ in range(config.dense_layer_num)])
        self.moe_layers = nn.ModuleList([Moe_layer(config) for _ in range(config.moe_layer_num)])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
    def forward(self, x):
        x = self.embedding.forward(x)
        for layer in self.dense:
            x = x + layer.forward(x)
        for layer in self.moe_layers:
            x = x + layer.forward(x)
        x = self.lm_head.forward(x)
        return x

class mhc_layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mix_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.rmsnorm = RMSNorm(config)
        self.attention = Multi_Head_Attention(config)
    def forward(self, residual):
        mixes = self.mix_proj(residual)
        pre, post, comb = sinkhorn_normalize(mixes)
        y = (pre * residual).sum(dim=2)
        y = self.rmsnorm.forward(y)
        y = self.attention.forward(y)
        new_residual = post * y.unsqueeze(2) + (comb * residual).sum(dim=2)
        return new_residual
