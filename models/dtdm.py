import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, einsum
class UnfoldLayer(nn.Module):
    def __init__(self, in_channels, top_channels, down_channels, branch_ratio, num_rects, group_channels = 4):
        self.in_channels = in_channels
        self.branch_ratio = branch_ratio
        self.num_rects = num_rects
        self.index_rect = nn.Linear(top_channels, branch_ratio * num_rects * 4)
        self.project_rect = nn.Sequential(
            nn.Linear(in_channels, down_channels),
            nn.GELU(),
            # 暂未加norm
        )
        self.group_channels = 4 # 在进行调制标量的计算时用到的分组中，每个组含有的通道数量（据此决定group的数量）
        self.groups = down_channels // group_channels
        self.mask = nn.Linear(top_channels, branch_ratio * num_rects * self.groups)
    def sample_rect_topleft(self, I_tuple, x, y):
        I, IX, IY, IT = I_tuple
        # I: (B, in_channels, H, W)
        # x, y:(B, N)
        # -> s: (B, in_channels, N)
        B, N = x.shape
        x = x.view(B, 1, N)
        y = y.view(B, 1, N)
        X = torch.ceil(x)
        Y = torch.ceil(y)
        zeros = torch.zeros_like(X)
        dx = X - x
        dy = Y - y
        # 索引广播
        ind_b = torch.arange(B, device = I.device).view(B, 1, 1)
        ind_c = torch.arange(self.in_channels, device = I.device).view(1, self.in_channels, 1)
        
        wx1 = 0.5 * dx ** 2
        wx2 = dx - wx1
        wy1 = 0.5 * dy ** 2
        wy2 = dy - wy1
        
        s_a = IT[ind_b, ind_c, X, Y]
        s_e1 = (
            wy1 * (IX[ind_b, ind_c, X, Y - 1] - 0.5 * I[ind_b, ind_c, X, Y - 1] - 0.5 * I[ind_b, ind_c, zeros, Y - 1]) +
            wy2 * (IX[ind_b, ind_c, X, Y] - 0.5 * I[ind_b, ind_c, X, Y] - 0.5 * I[ind_b, ind_c, zeros, Y])
        )
        s_e2 = (
            wx1 * (IY[ind_b, ind_c, X - 1, Y] - 0.5 * I[ind_b, ind_c, X - 1, Y] - 0.5 * I[ind_b, ind_c, X - 1, zeros]) +
            wx2 * (IY[ind_b, ind_c, X, Y] - 0.5 * I[ind_b, ind_c, X, Y] - 0.5 * I[ind_b, ind_c, X, zeros]) 
        )
        s_c = (
            wx1 * wy1 * I[ind_b, ind_c, X - 1, Y - 1] + wx2 * wy1 * I[ind_b, ind_c, X, Y - 1] + 
            wx1 * wy2 * I[ind_b, ind_c, X - 1, Y] + wx2 * wy2 * I[ind_b, ind_c, X, Y]
        )
        return s_a - s_e1 - s_e2 + s_c
    def sample_rect(self, I_tuple, p):
        # I: (B, in_channels, H, W), p: (B, D, branch_ratio, num_rects, 4) 
        # 4-tuple = (x1, y1, x2, y2)
        # -> r: (B, D, branch_ratio, num_rects, in_channels)
        I, IX, IY, IT = I_tuple
        B, D, branch_ratio, num_rects, _ = p.shape
        _, __, H, W = I.shape
        fac = torch.tensor([H - 1, W - 1, H - 1, W - 1]).view(1, 1, 1, 1, 4).to(I.device)
        # 缩放因子，这里避免边角处的越界问题
        p = F.sigmoid(p) * fac
        p = p.view(B, D * self.branch_ratio * self.num_rects, 4) # 重塑形状便于索引
        x1, y1, x2, y2 = torch.split(p, 1, dim = -1)
        S = (
            self.sample_rect_topleft(I_tuple, x2, y2) 
            - self.sample_rect_topleft(I_tuple, x1, y2) 
            - self.sample_rect_topleft(I_tuple, x2, y1)
            + self.sample_rect_topleft(I_tuple, x1, y1)
        )
        S = S.view(B, self.in_channels, D, self.branch_ratio, self.num_rects)
        S = S.permute(0, 2, 3, 4, 1) # (B, D, branch_ratio, num_rects, in_channels)
        # 考虑计算方法：
        # 先写一个能够计算浮点坐标到左上角位置构成的矩形内均值的函数
        # 然后再使用割补法计算任意两个浮点坐标组成的矩形内均值（有向）的函数
        return S
    def forward(self, x, Ituple):
        # x: (B, D, C) -> UnfoldLayer(x): (B, D', C')
        # D' = D * branch_ratio
        
        # x -> p(x): (B, D, branch_ratio, num_rects, 4)
        B, D, C = x.shape
        p = self.index_rect(x) # (B, D, branch_ratio * num_rects * 4)
        p = p.view(B, D, self.branch_ratio, self.num_rects, 4)
        rects = self.sample_rect(Ituple, p) # (B, D, branch_ratio, num_rects, in_channels)
        rects = self.project_rect(rects) # (B, D, branch_ratio, num_rects, down_channels)
        # x -> m(x): (B, D, branch_ratio, num_rects, groups)
        mask = self.mask(x) #(B, D, branch_ratio * num_rects * groups)
        mask = mask.view(B, D, self.branch_ratio, self.num_rects, self.groups, 1)
        mask = F.softmax(mask, 3)
        rects = rects.view(B, D, self.branch_ratio, self.num_rects, self.groups, self.group_channels)
        output = (mask * rects).sum(3).view(B, D, self.branch_ratio, self.down_channels)
        return output
        
class DTDM(nn.Module):
    def __init__(self, 
        projection = 256,
        in_channels = 3,         
        query_channels = 12, # 初始的query嵌入维度
        unfold_channels = [24, 48, 96, 128],
        branch_ratio = [2, 2, 2, 2],
        num_rects = [3, 3, 3, 3],
        num_classes = 200,
        group_channels = [4, 4, 4, 4]
        ):
        super().__init__()
        self.num_classes = num_classes
        self.branch_ratio = branch_ratio
        self.num_branches = [None] * len(self.branch_ratio)
        self.unfold_layers = nn.ModuleList()
        for i, v in enumerate(self.branch_ratio):
            if i == 0:
                self.num_branches[i] = v
                self.unfold_layers.append(UnfoldLayer(
                    in_channels = in_channels, 
                    top_channels = query_channels, 
                    down_channels = unfold_channels[i],
                    branch_ratio = branch_ratio[i],
                    num_rects = num_rects[i],
                    group_channels = group_channels[i]
                ))
            else:
                self.num_branches[i] = self.num_branches[i - 1] * v
                self.unfold_layers.append(UnfoldLayer(
                    in_channels = in_channels, 
                    top_channels = unfold_channels[i - 1], 
                    down_channels = unfold_channels[i],
                    branch_ratio = branch_ratio[i],
                    num_rects = num_rects[i],
                    group_channels = group_channels[i]
                ))
        
        self.in_channels = in_channels
        self.query = nn.Parameter(torch.zeros(1, 1, query_channels)) # (1, D_0 = 1, C_0)
        
        # fusion_dim = embed_dim[-1]
        # self.head = nn.Sequential(
            # nn.Conv2d(fusion_dim, projection, kernel_size=1, bias=False),
            # nn.BatchNorm2d(projection),
            # nn.SiLU(),
            # nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(projection, num_classes, kernel_size=1) if num_classes > 0 else nn.Identity()
        # )
        
    def forward_unfold(self, I):
        # I: (B, in_channels, H, W)
        IX = torch.cumsum(I, dim = 2)
        IY = torch.cumsum(I, dim = 3)
        IT = torch.cumsum(IX, dim = 3)
        outputs = []
        for i, layer in enumerate(self.unfold_layers):
            if i == 0:
                outputs.append(layer(x = self.query, (I, IX, IY, IT)))
            else:
                output = layer(x = outputs[-1], (I, IX, IY, IT))
                outputs.append(output)
        return outputs
    def forward_fuse(self, x):
        return x
    def forward(self, I):
        x = self.forward_unfold(I)
        x = self.forward_fuse(x)
        x = self.head(x)
        return x
def dtdm_xxt(num_classes = 200):
    model = DTDM(
        projection = 256,
        embed_channels = [24, 48, 96, 128], 
        num_classes = num_classes
    )
    return model
