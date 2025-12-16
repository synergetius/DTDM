import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, einsum

class DTDM(nn.Module):
    def __init__(self, 
        projection = 256, 
        embed_dim = [24, 48, 96, 128],
        num_classes = 200
        ):
        super().__init__()
        self.num_classes = num_classes
        fusion_dim = embed_dim[-1]
        self.head = nn.Sequential(
            nn.Conv2d(fusion_dim, projection, kernel_size=1, bias=False),
            nn.BatchNorm2d(projection),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(projection, num_classes, kernel_size=1) if num_classes > 0 else nn.Identity()
        )
    def forward_unfold(self, x):
        return x
    def forward_fuse(self, x):
        return x
    def forward(self, x):
        x = self.forward_unfold(x)
        x = self.forward_aggr(x)
        x = self.head(x)
        return x
def dtdm_xxt(num_classes = 200):
    model = DTDM(
        projection = 256,
        embed_dim = [24, 48, 96, 128], 
        num_classes = num_classes
    )
    return model
