import torch
import torch.nn.functional as F
from torch import nn
import itertools
from einops import rearrange
class Split(nn.Module):
    # 复制元素并调整通道数
    def __init__(self, in_channels, out_channels, factor = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.project = nn.Linear(in_channels, out_channels)
        self.factor = factor
    def forward(self, x, I_tuple = None):
        B, N, C = x.shape
        x = x.view(B, 1, N, C).expand(B, self.factor, N, C).view(B, self.factor * N, C)
        # 注意view的机制让重复的元素按特定顺序排列
        x = self.project(x)
        return x
class Merge(nn.Module):
    # 合并元素并调整通道数
    def __init__(self, in_channels, out_channels, factor = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.project = nn.Linear(in_channels, out_channels)
        self.factor = factor
    def forward(self, x, I_tuple = None):
        B, N, C = x.shape
        M = N // self.factor
        x = torch.mean(x.view(B, self.factor, M, C), dim = 1)
        x = self.project(x)
        return x
def mlp(depth, channels):
    # 先考虑不使用维度展开的最简单的mlp
    seq = [nn.Linear(channels, channels)]
    for i in range(depth - 1):
        seq.append(nn.GELU())
        seq.append(nn.Linear(channels, channels))
    return nn.Sequential(*seq)
class Process(nn.Module):
    # 核心的加工
    def __init__(self, channels, depths = (2, 3, 4, 5)):
        super().__init__()
        self.channels = channels
        self.bunches = len(depths)
        self.mlps = nn.ModuleList()
        for depth in depths:
            self.mlps.append(mlp(depth, channels))
        self.norm = nn.LayerNorm(channels)
        # 要求depths数组长度可整除输入的元素数量
    def forward(self, x, I_tuple = None):
        B, N, C = x.shape
        xs = torch.chunk(x, self.bunches, dim = 1) # 默认同一个bunch为连续一块
        ys = []
        for mlp_i, x_i in zip(self.mlps, xs):
            ys.append(mlp_i(x_i))
        y = torch.cat(ys, dim = 1)
        y = self.norm(y) + x
        return y
class Interact(nn.Module):
    def __init__(self, channels, bunches, groups):
        super().__init__()
        self.channels = channels
        self.bunches = bunches# 分别进行交互的每个部分中元素的个数
        self.groups = groups
        self.project_key = nn.Linear(channels, channels)
        self.project_query = nn.Linear(channels, channels)
        self.project_value = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
    def forward(self, x, I_tuple = None):
        B, N, C = x.shape
        parts = N // self.bunches
        group_channels = C // self.groups
        x_ = x.view(B, parts, self.bunches, C)
        k = self.project_key(x_).view(B, parts, self.bunches, self.groups, group_channels)
        q = self.project_query(x_).view(B, parts, self.bunches, self.groups, group_channels)
        v = self.project_value(x_).view(B, parts, self.bunches, self.groups, group_channels)
        attn = torch.einsum("b p t g c, b p s g c -> b p t s g", q, k)
        weight = torch.softmax(attn / group_channels ** 2, dim = 3)
        y = torch.einsum("b p t s g, b p s g c -> b p t g c", weight, v)
        y = self.norm(y.reshape(B, N, C)) + x
        # y = rearrange(y, "b p t g c -> b (p t) (g c)")
        # y = self.norm(y) + x
        # y = self.norm(y.view(B, N, C)) + x
        return y
class Admit(nn.Module):
    def __init__(self, in_channels, channels, rects, groups):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.rects = rects
        self.groups = groups
        self.group_channels = self.channels // self.groups
        self.project_p = nn.Linear(channels, 4 * rects)
        self.project_m = nn.Linear(channels, rects * groups)
        self.project_s = nn.Linear(in_channels, channels)
        self.norm = nn.LayerNorm(channels)
    def sample_rects_topleft(self, I_tuple, x, y):
        I, IX, IY, IT = I_tuple
        # I: (B, channels, H, W)
        # x, y:(B, number)
        # -> s: (B, channels, number)
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
    def sample_rects(self, p, I_tuple):
        # I: (B, in_channels, H, W), p: (B, M, rects, 4) 
        I, IX, IY, IT = I_tuple
        B, M, rects, _ = p.shape
        _, __, H, W = I.shape
        fac = torch.tensor([H - 1, W - 1, H - 1, W - 1]).view(1, 1, 1, 4).to(I.device)
        p = F.sigmoid(p) * fac
        p = p.view(B, M * rects, 4)
        x1 = p[:, :, 0] # (B, M * rects)
        y1 = p[:, :, 1]
        x2 = p[:, :, 2]
        y2 = p[:, :, 3]
        S = (
            self.sample_rects_topleft(I_tuple, x2, y2) 
            - self.sample_rects_topleft(I_tuple, x1, y2) 
            - self.sample_rects_topleft(I_tuple, x2, y1)
            + self.sample_rects_topleft(I_tuple, x1, y1)
        ) # (B, in_channels, M * rects)
        S = S.view(B, self.in_channels, M, rects)
        S = S.permute(0, 2, 3, 1) # (B, M, rects, in_channels)
        return S
    def forward(self, x, I_tuple):
        B, N, C = x.shape
        
        p = self.project_p(x).view(B, N, self.rects, 4)
        m = self.project_m(x).view(B, N, self.rects, self.groups)
        m = F.softmax(m, dim = 2)
        s = self.project_s(self.sample_rects(p, I_tuple)) # (B, N, rects, channels)
        s = s.view(B, N, self.rects, self.groups, self.group_channels)
        y = torch.einsum("b n s g, b n s g c -> b n g c", m, s)
        y = y.view(B, N, C)
        y = self.norm(y) + x
        return y
class MSM2(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.layers = nn.ModuleList()
        self.bunch_depths = [1, 2, 3, 4]
        self.channels = 64
        self.head_channels = 256
        self.in_channels = in_channels
        self.rects = 4
        self.groups = 16
        self.elements = 16
        # for s in "apiapi":
        for s in "apiapiapi":
            if s == "a":
                self.layers.append(Admit(
                    in_channels = in_channels, 
                    channels = self.channels, 
                    rects = self.rects, 
                    groups = self.groups
                ))
            elif s == "p":
                self.layers.append(Process(
                    channels = self.channels, 
                    depths = self.bunch_depths
                ))
            elif s == "i":
                self.layers.append(Interact(
                    channels = self.channels, 
                    bunches = len(self.bunch_depths), 
                    groups = self.groups
                ))
        self.query = nn.Parameter(torch.zeros(1, self.elements, self.channels))
        self.head = nn.Sequential( #### 仿照OverLoCK的结构
            nn.Linear(self.channels, self.head_channels),
            nn.LayerNorm(self.head_channels),
            nn.SiLU(),
            Merge(in_channels = self.head_channels, out_channels = self.num_classes, factor = self.elements)
        )
    def forward(self, I):
        IX = torch.cumsum(I, dim = 2)
        IY = torch.cumsum(I, dim = 3)
        IT = torch.cumsum(IX, dim = 3)
        I_tuple = (I, IX, IY, IT)
        B = I.shape[0]
        x = self.query.expand(B, self.query.shape[1], self.query.shape[2])
        for layer in self.layers:
            x = layer(x, I_tuple)
        scores = self.head(x).squeeze()
        # print('score:', scores.shape)
        # print('x:', x.shape)
        return scores
# class MSM2(nn.Module):
    # def __init__(
        # self, 
        # channels = [64, 32, 16, 8],
        # bunch_depths = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        # layer_symbols = [1, 1, 1, 1] #
    # ):
        # self.splits = nn.ModuleList()
        
    # def forward(self, I):
        # pass
def msm2_xxt(num_classes = 20):
    model = MSM2(
        in_channels = 3,
        num_classes = num_classes
    )
    return model