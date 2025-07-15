import torch 
import torch.nn as nn
import torch.nn.functional as F
from scaledotattention import ScaleDotattention
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, heads):
        super().__init__()
        assert d_k % heads == 0
        self.heads = heads
        self.d_k = d_k
        self.head_dim = d_k // heads
        self.q_linear = nn.Linear(d_k, d_k)
        self.k_linear = nn.Linear(d_k, d_k)
        self.v_linear = nn.Linear(d_k, d_k)
        self.w_o = nn.Linear(d_k, d_k)
        self.attention = ScaleDotattention(self.head_dim)

    def forward(self, x, k=None, v=None, mask=None):
        B, T, C = x.shape
        k = x if k is None else k
        v = x if v is None else v
        q = self.q_linear(x)
        k = self.k_linear(k)
        v = self.v_linear(v)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, k.size(1), self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, v.size(1), self.heads, self.head_dim).transpose(1, 2)
        out, attn = self.attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.w_o(out)
        return out, attn
