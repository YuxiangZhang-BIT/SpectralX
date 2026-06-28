import torch
import torch.nn as nn
import math
from torch import Tensor
from src.utils import *

class Attribute_refinedAdapters(nn.Module):
    def __init__(
        self,
        scale_init: float = 0.01,
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(512)
        num = 28
        self.spa_match_attn = PerceiverAttention(512, out_dim=512, dim_head=512 // 2, heads=1)
        self.spe_match_attn = PerceiverAttention(num**2, out_dim=512, dim_head=num**2 // 2, heads=1)
        self.spa_mlp = nn.Linear(num**2, 196)
        self.spe_mlp = nn.Linear(512, 196)

        self.refine_spa = nn.Sequential(
            nn.Conv2d(512, 256, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.refine_spe = nn.Sequential(
            nn.Conv1d(512, 196, 1, bias=False),
            nn.BatchNorm1d(196),
            nn.ReLU(True),
        )
        self.scale_init = scale_init
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(
        self, feats: Tensor, layer: int, conv_feats: Tensor, batch_first=False, has_cls_token=True
    ) -> Tensor:
        
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=1)
        delta_feat, conv_feats = self.forward_delta_feat(
            feats,
            conv_feats,
            layer,
        )

        feats = feats + delta_feat * self.scale

        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=1)
            
        return feats, conv_feats

    
    def forward_delta_feat(self, feats: Tensor, conv_feats: Tensor,  layers: int) -> Tensor:

        spa_feats, spe_feats = torch.tensor_split(feats, [feats.shape[-1]//2], dim=-1)
        spa_tokens, attn_spa = self.spa_match_attn(conv_feats['spa_fea'], spa_feats)
        spe_tokens, attn_spe = self.spe_match_attn(conv_feats['spe_fea'], spe_feats)

        max_indices = torch.argmax(attn_spa.squeeze(), dim=1)
        expanded_indices = max_indices.unsqueeze(-1).expand(-1, -1, spa_feats.size(-1))
        similar_tokens_spa = self.norm(torch.gather(spa_feats, 1, expanded_indices))
        h = w = int(math.sqrt(similar_tokens_spa.shape[1]))
        re_spa_fea = einops.rearrange(similar_tokens_spa, "b (h w) d -> b d h w", h=h, w=w)
        re_spa_fea = einops.rearrange(self.refine_spa(re_spa_fea), "b d h w -> b (h w) d", h=h//2, w=w//2)
        spa_feats = self.gap(re_spa_fea)*spa_feats

        max_indices = torch.argmax(attn_spe.squeeze(), dim=1)
        expanded_indices = max_indices.unsqueeze(-1).expand(-1, -1, spe_feats.size(-1))
        similar_tokens_spe = self.norm(torch.gather(spe_feats, 1, expanded_indices))
        re_spe_fea = self.refine_spe(similar_tokens_spe)
        spe_feats = self.gap(re_spe_fea)*spe_feats

        return torch.cat([spa_feats, spe_feats], dim=-1), conv_feats