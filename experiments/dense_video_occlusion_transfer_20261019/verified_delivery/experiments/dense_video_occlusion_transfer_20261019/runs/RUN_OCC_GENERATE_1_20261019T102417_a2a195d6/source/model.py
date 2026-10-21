"""Causal visibility head. All three arms have exactly the same parameters."""
import torch
from torch import nn

class VisibilityHead(nn.Module):
 def __init__(self):
  super().__init__()
  self.project=nn.Sequential(nn.Linear(392,128),nn.ReLU())
  self.gru=nn.GRU(128,64,batch_first=True)
  self.state=nn.Linear(64,4);self.visibility=nn.Linear(64,1)
 def forward(self,x,arm='FULL'):
  if arm=='NOCTX':x=torch.cat([x[...,:128],x[...,:128],x[...,256:]],-1)
  if arm=='NOW':x=x[:,-1:].expand(-1,8,-1)
  h,_=self.gru(self.project(x));h=h[:,-1]
  return self.state(h),self.visibility(h).sigmoid().squeeze(-1)

def windows(x,ids):
 # id boundaries are explicit, never use another individual's history.
 index=[];starts={}
 for i,k in enumerate(ids):
  k=int(k);start=starts.setdefault(k,i)
  index.append([max(start,i-7+j) for j in range(8)])
 return torch.as_tensor(index,dtype=torch.long)
