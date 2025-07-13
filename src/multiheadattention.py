import torch 
import torch.nn as nn
import torch.nn.functional as F
from scaledotattention import ScaleDotattention
class MultiHeadAttention(nn.Module):
  def __init__(self,d_k,heads):
    super().__init__()
    assert d_k%heads==0
    self.heads=heads
    self.d_k=d_k
    self.head_dim=d_k//heads

    self.q_linear=nn.Linear(d_k,d_k)
    self.k_linear=nn.Linear(d_k,d_k)
    self.v_linear=nn.Linear(d_k,d_k)
    self.w_o=nn.Linear(d_k,d_k)
    self.attention=ScaleDotattention(self.head_dim)
  def forward(self,x,mask=None):
    B,T,C=x.shape
    q=self.q_linear(x) # Use input x to generate q, k, v
    k=self.k_linear(x)
    v=self.v_linear(x)

    # Reshape for multi-heads: (B, T, H, head_dim) → (B, H, T, head_dim)
    q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
    k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
    v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # Apply attention on all heads
    out, attn = self.attention(q, k, v, mask=mask)  # (B, H, T, head_dim)

        # Concatenate heads: (B, H, T, head_dim) → (B, T, C)
    out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection
    out = self.w_o(out)

    return out, attn