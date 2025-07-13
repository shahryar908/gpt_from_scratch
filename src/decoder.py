import torch 
import torch.nn as nn
import torch.nn.functional as F
from multiheadattention import MultiHeadAttention
from encoder import  FeedForward


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        # Masked Self-Attention
        _x, _ = self.self_attn(x, tgt_mask)
        x = self.norm1(x + self.dropout(_x))

        # Cross-Attention with encoder output
        _x, _ = self.cross_attn(x= x, mask=memory_mask)  # query = x, keys/values = enc_out
        x = self.norm2(x + self.dropout(_x))

        # Feed-Forward
        _x = self.ffn(x)
        x = self.norm3(x + self.dropout(_x))

        return x
