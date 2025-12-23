import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, einsum
class UnfoldLayer(nn.Module):
    def __init__(
        self, 
        in_channels, 
        top_channels, 
        down_channels, 
        branch_ratio, 
        num_rects, 
        groups
        ):
        # group_channels = 4):
        super().__init__()
        self.in_channels = in_channels
        self.branch_ratio = branch_ratio
        self.num_rects = num_rects
        self.top_channels = top_channels
        self.down_channels = down_channels
        self.index_rect = nn.Linear(top_channels, branch_ratio * num_rects * 4)
        self.project_rect = nn.Sequential(
            nn.Linear(in_channels, down_channels),
            nn.GELU(),
            # 暂未加norm
        )
        self.groups = groups
        self.group_channels = down_channels // self.groups # 在进行调制标量的计算时用到的分组中，每个组含有的通道数量（据此决定group的数量）
        # self.groups = down_channels // group_channels
        self.mask = nn.Linear(top_channels, branch_ratio * num_rects * self.groups)
    def sample_rect_topleft(self, I_tuple, x, y):
        I, IX, IY, IT = I_tuple
        # I: (B, in_channels, H, W)
        # x, y:(B, N)
        # -> s: (B, in_channels, N)
        B, N = x.shape
        x = x.view(B, 1, N)
        y = y.view(B, 1, N)
        X = torch.ceil(x).to(torch.int)
        Y = torch.ceil(y).to(torch.int)
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
        x1 = p[:, :, 0]
        y1 = p[:, :, 1]
        x2 = p[:, :, 2]
        y2 = p[:, :, 3]
        # x1, y1, x2, y2 = torch.split(p, 1, dim = -1)
        # print(x2.shape)
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
        mask = mask.view(B, D * self.branch_ratio, self.num_rects, self.groups, 1)
        mask = F.softmax(mask, 2)
        rects = rects.view(B, D * self.branch_ratio, self.num_rects, self.groups, self.group_channels)
        output = (mask * rects).sum(2).view(B, D * self.branch_ratio, self.down_channels)
        return output
class FuseLayer(nn.Module):
    def __init__(self, down_channels, top_channels, out_channels, branch_ratio, groups):
        super().__init__()
        self.branch_ratio = branch_ratio
        self.down_channels = down_channels # u_{i+1} 通道数
        self.top_channels = top_channels # x_{i}
        self.out_channels = out_channels # u_{i}
        self.project_key = nn.Linear(down_channels, top_channels) # 从u_{i+1}投影得到key
        self.project_query = nn.Linear(top_channels, top_channels) # 暂时以top_channels作为q,k统一的维度
        self.project_value = nn.Linear(down_channels, out_channels)
        self.groups = groups
        # self.group_channels = top_channels // groups
        ## 在进行了投影之后，计算attn之前再进行分组
        self.project_res = nn.Linear(out_channels, out_channels)
    def forward(self, u, x):
        # 接受输入：u_{i+1}, x_{i}，
        # 输出： u_{i}
        # x_{i} : (B, D, C)
        # u_{i} : (B, D, E)
        # u_{i + 1} : (B, D', E') # E'是U的特征通道数
        k = self.project_key(u) # (B, D', C)
        q = self.project_query(x) # (B, D, C)
        v = self.project_value(u) # (B, D', E)
        B, D, _ = x.shape
        k = k.view(B, D, self.branch_ratio, self.groups, self.top_channels // self.groups) # 拆分分支维度和注意力头的分组
        # k: (B, D, branch_ratio, G, C/G)
        q = q.view(B, D, self.groups, top_channels // self.groups) # 拆分分支维度和注意力头的分组
        # q: (B, D, G, C/G)
        v = v.view(B, D, self.branch_ratio, self.groups, self.out_channels // self.groups) # 拆分分支维度和注意力头的分组
        # v: (B, D, branch_ratio, G, C/G)
        # q @ k :
        attn = einsum(q, k, 'b d g c, b d r g c -> b d r g c')
        attn = torch.softmax(attn, dim = 2)
        res = einsum(attn, v, 'b d r g c, b d r g c -> b d g c')
        res = res.view(B, D, self.out_channels)
        res = self.project_res(res)
        return res        
        # k = rearrange(k, "b d m -> b d0 ")
class DTDM(nn.Module):
    def __init__(self, 
        projection = 256,
        in_channels = 3,         
        query_channels = 12, # 初始的query嵌入维度
        # unfold_channels = [128, 96, 48, 24],
        unfold_channels = [64, 32, 16, 8],
        unfold_groups = [16, 8, 4, 2], 
        fuse_channels = [64, 32, 16, 8], # fuse_channels的顺序是倒着来的（U_i与X_i对应）
        fuse_groups = [16, 8, 4, 2], 
        branch_ratio = [2, 2, 2, 2],
        num_rects = [3, 3, 3, 3],
        num_classes = 200,
        
        # group_channels = [4, 4, 4, 4]
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
                    groups = unfold_groups[i],
                    # group_channels = group_channels[i]
                ))
            else:
                self.num_branches[i] = self.num_branches[i - 1] * v
                self.unfold_layers.append(UnfoldLayer(
                    in_channels = in_channels, 
                    top_channels = unfold_channels[i - 1], 
                    down_channels = unfold_channels[i],
                    branch_ratio = branch_ratio[i],
                    num_rects = num_rects[i],
                    groups = unfold_groups[i],
                    # group_channels = group_channels[i]
                ))
        self.fuse_layers = nn.ModuleList()
        for i, v in enumerate(self.branch_ratio):
            if i == len(fuse_channels) - 1:
                # middle projection: X_{n - 1} -> U_{n - 1}
                self.fuse_layers.append(nn.Linear(unfold_channels[i], fuse_channels[i]))
            else:
                self.fuse_layers.append(FuseLayer(
                    top_channels = fuse_channels[i],
                    down_channels = unfold_channels[i + 1],
                    out_channels = unfold_channels[i]
                    branch_ratio = branch_ratio[i], 
                    groups = fuse_groups[i]
                ))
                
        self.in_channels = in_channels
        self.query = nn.Parameter(torch.zeros(1, 1, query_channels)) # (1, D_0 = 1, C_0)

        # u_0 : (B, D_0, E_0) # 如果计算成功，应有 D_0 = 1
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(fuse_channels[0], num_classes)
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
                outputs.append(layer(self.query, (I, IX, IY, IT)))
            else:
                output = layer(outputs[-1], (I, IX, IY, IT))
                outputs.append(output)
        return outputs
    def forward_fuse(self, xs):
        # xs: n层特征，按照自顶向下的顺序的列表
        us = []
        n = len(xs)
        for i in range(n - 1, -1, -1):
            x = xs[i]
            if i == n - 1:
                u = self.fuse_layers[i](x)
            else:
                u = self.fuse_layers[i](u, x)
            us.append(u)
        return us
    def forward(self, I):
        xs = self.forward_unfold(I)
        us = self.forward_fuse(xs)
        u_0 = us[-1].squeeze() # (B, E_0)
        cls = self.head(u_0) # 最后一层（最终的融合结果）输入head进行分类
        return cls
def dtdm_xxt(num_classes = 200):
    model = DTDM(
        # projection = 256,
        # unfold_channels = [24, 48, 96, 128], 
        # num_classes = num_classes
    )
    return model
