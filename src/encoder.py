import torch 
import torch.nn as nn
import torch.nn.functional as F
from multiheadattention import MultiHeadAttention
class FeedForward(nn.Module):
  def __init__(self,embed_dim,ff_dim,dropout=0.1):
    super().__init__()
    self.net=nn.Sequential(
        nn.Linear(embed_dim,ff_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(ff_dim,embed_dim),
        nn.Dropout(dropout)
    )
  def forward(self,x):
    return self.net(x)


class TransformerEncoderBlock(nn.Module):
  def __init__(self,embed_dim,heads,ff_dim,dropout=0.1):
    super().__init__()
    self.mha=MultiHeadAttention(embed_dim,heads)
    self.ff=FeedForward(embed_dim,ff_dim,dropout)
    self.norm1=nn.LayerNorm(embed_dim)
    self.norm2=nn.LayerNorm(embed_dim)
    self.dropout=nn.Dropout(dropout)
  def forward(self,x,mask=None):
    attn_output, _ = self.mha(x,mask) # Unpack the tuple and get the output tensor
    x = self.norm1(x + self.dropout(attn_output))

        # Feedforward sublayer
    ff_output = self.ff(x) # Corrected the feedforward layer name
    x = self.norm2(x + self.dropout(ff_output))

    return x