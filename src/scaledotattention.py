import torch 
import torch.nn as nn
import torch.nn.functional as F
class ScaleDotattention(nn.Module):
  def __init__(self,d_k):
    super().__init__()
    self.d_k=d_k
  def forward(self,q,k,v ,mask=None):
    scores=torch.matmul(q,k.transpose(-2,-1)) /torch.sqrt(torch.tensor(self.d_k,dtype=torch.float))

    if mask is not None:
      scores=scores.masked_fill(mask==0,float('-inf'))
    attn=F.softmax(scores,dim=-1)
    output=torch.matmul(attn,v)

    return output,attn