"""Position embeddings and attention layers from MuJoCo-VISTA-ACT."""
import math
import torch
from torch import nn

def sine_1d(position, dim):
    if dim % 2:
        raise ValueError("Sine embedding dimension must be even")
    frequency = torch.exp(
        -math.log(10000.0) * torch.arange(0, dim, 2, device=position.device) / dim
    )
    angle = position.float().unsqueeze(-1) * frequency
    return torch.stack((angle.sin(), angle.cos()), dim=-1).flatten(-2)


def sine_2d(height, width, dim, device):
    # 与特征 flatten(2) 一致：行优先；两个坐标各占一半通道。
    if dim % 4:
        raise ValueError("2D sine embedding dimension must be divisible by 4")
    y = (torch.arange(height, device=device) + 0.5) / height * 2 * math.pi
    x = (torch.arange(width, device=device) + 0.5) / width * 2 * math.pi
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.cat((sine_1d(yy, dim // 2), sine_1d(xx, dim // 2)), -1).reshape(-1, dim)


class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.hidden_dim
        self.attn = nn.MultiheadAttention(d, cfg.nheads, dropout=cfg.dropout,
                                          batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, cfg.dim_feedforward), nn.ReLU(),
                                 nn.Dropout(cfg.dropout), nn.Linear(cfg.dim_feedforward, d))
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, pos, is_pad=None):
        qk = x + pos
        update = self.attn(qk, qk, x, key_padding_mask=is_pad, need_weights=False)[0]
        x = self.norm1(x + self.drop(update))
        return self.norm2(x + self.drop(self.ffn(x)))


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.hidden_dim
        self.self_attn = nn.MultiheadAttention(d, cfg.nheads, dropout=cfg.dropout,
                                               batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, cfg.nheads, dropout=cfg.dropout,
                                                batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, cfg.dim_feedforward), nn.ReLU(),
                                 nn.Dropout(cfg.dropout), nn.Linear(cfg.dim_feedforward, d))
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, h, query_pos, memory, memory_pos):
        qk = h + query_pos
        update = self.self_attn(qk, qk, h, need_weights=False)[0]  # 没有 causal mask
        h = self.norms[0](h + self.drop(update))
        update = self.cross_attn(h + query_pos, memory + memory_pos, memory,
                                 need_weights=False)[0]
        h = self.norms[1](h + self.drop(update))
        return self.norms[2](h + self.drop(self.ffn(h)))
