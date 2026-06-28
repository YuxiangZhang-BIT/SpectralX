import torch
import torch.nn as nn
from src.adapters.MMOE import MMoE

class ConvFusionLayer(nn.Module):
    """
    """

    def __init__(
        self,
        dim=1024,
        r = 32,
      
    ):
        super().__init__()
        self.dim = dim
        kersize =3
        self.conv1 = nn.Conv2d(dim,dim//4,(1, 1),bias=False,padding=0)
        self.conv2 = nn.Conv2d(dim//4,dim//4,(kersize, kersize),bias=False,padding=1)
        self.conv3 = nn.Conv2d(dim//4,dim,(1, 1),bias=False,padding=0)
        
    def forward(self, x,H,W):

        feature = x
        B,L,N = feature.shape
        feature = feature.permute(0,2,1).view(B,N,H,W)
        feature = self.conv1(feature)
        feature = self.conv2(feature)
        feature = self.conv3(feature)
        x = feature.view(B,N,H*W).permute(0,2,1)
        return x

class AoMixtureOfAdapters(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """

    def __init__(
        self,
        dim=1024,
        out_dim=1024,
        task_num=2,
    ):
        super().__init__()
        self.dim = dim
        self.dimReduction = nn.Linear(dim, dim//4, bias=False)
        self.dimIncrease = nn.Linear(dim//4, out_dim, bias=False)
        self.MoA = MMoE(dim//4, dim//4, 4, dim//32, noisy_gating=True, k=2,task_num=task_num)
        
        #print()
        self.modal_shifts = [nn.Parameter(torch.zeros(dim))  for i in range(2*task_num)]
        self.MoA_relu = nn.ReLU()
        self.MoA_sigmoid = nn.Sigmoid()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim//4)
        self.init_scale_shift()

    def init_scale_shift(self):
        for layer in self.modal_shifts:
            nn.init.normal_(layer, std=.02)
        torch.nn.init.xavier_uniform_(self.dimReduction.weight)
        torch.nn.init.xavier_uniform_(self.dimIncrease.weight)
       
    def forward(self, y, task_index, loss_coef=1e-3):
        B,N,C = y.shape
        y = self.norm1(y)
        y = self.dimReduction(y)
        y = self.norm2(y)
        y = y.view(B*N,C//4)
        y = self.MoA(y, task_index, loss_coef)
        y = y.view(B,N,C//4)
        out_x = self.dimIncrease(y)
        return out_x