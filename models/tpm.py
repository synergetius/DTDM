import torch
import torch.nn.functional as F
from torch import nn
import itertools
class Split(nn.Module):
    # 复制元素并调整通道数
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.project = nn.Linear(in_channels, out_channels)
        self.factor = 2
    def forward(self, x, I_tuple = None):
        B, N, C = x.shape
        x = x.view(B, N, 1, C).expand(B, N, 2, C).reshape(B, N * 2, C)
        # 注意：此处的视图中，每个分支拆分出的两个矩形（子分支）是连续排放在一起的
        x = self.project(x)
        return x
class Merge(nn.Module):
    # 合并元素并调整通道数
    def __init__(self, in_channels, out_channels, factor = 2, use_norm = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.project = nn.Linear(in_channels, out_channels)
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(out_channels)
        self.factor = factor
    def forward(self, x, I_tuple = None):
        B, N, C = x.shape
        M = N // self.factor
        x = torch.mean(x.view(B, M, self.factor, C), dim = 2) 
        # 注意：此处的视图中，待合并的子分支是连续排放在一起的
        x = self.project(x)
        if self.use_norm:
            x = self.norm(x)
        return x
class Admit(nn.Module):
    def __init__(self, in_channels, channels, groups, start_dim, split_levels = 3):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.groups = groups
        self.group_channels = channels // groups
        self.split_levels = split_levels
        self.num_rects = 2 ** split_levels
        self.start_dim = start_dim # 0 or 1, 开始分割的维度
        self.project_ratios = nn.Sequential(
            nn.Linear(channels, 2 ** split_levels - 1),
            nn.Sigmoid()
        )
        self.project_mask = nn.Linear(channels, self.num_rects * groups)
        self.project_samples = nn.Linear(in_channels, channels)
        self.norm = nn.LayerNorm(channels)
    def split_rect(self, r, dim, ratios):
        # r: (B, N, 2 ** i, 4)
        # dim == 0 or dim == 1
        # ratios: (B, N, 2 ** i)
        # out: (B, N, 2 ** (i + 1), 4)
        
        # print('##', r.shape)
        ratios = ratios.unsqueeze(-1)
        # mid = r[:, :, :, dim] * (1 - ratios) + r[:, :, :, dim + 2] * ratios
        pos = torch.chunk(r, chunks = 4, dim = 3)
        mid = pos[dim] * (1 - ratios) + pos[dim + 2] * ratios
        # print(pos[0].shape, mid.shape)
        # mid = (r[:, :, dim] + r[:, :, dim + 2]) / 2
        if dim == 0:
            r1 = torch.cat([pos[0], pos[1], mid, pos[3]], 3)
            r2 = torch.cat([mid, pos[1], pos[2], pos[3]], 3)
        else:
            r1 = torch.cat([pos[0], pos[1], pos[2], mid], 3)
            r2 = torch.cat([pos[0], mid, pos[2], pos[3]], 3)
        out = torch.cat([r1, r2], 2)
        # print("#", out.shape)
        return out
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
        p = p * fac ######### 前面生成的矩形已经是关于整张图的比例值了，所以直接乘以尺寸
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
        eps = 1e-9
        area = torch.abs((x1 - x2) * (y1 - y2)) + eps # (B, M * rects)
        area = area.unsqueeze(1) # (B, 1, M * rects)
        S = S / area ############ 要对每个rect的面积做一下归一化
        S = S.view(B, self.in_channels, M, rects)
        S = S.permute(0, 2, 3, 1) # (B, M, rects, in_channels)
        return S
    def forward(self, x, r, I_tuple):
        # x: (B, N, C)
        # r: (B, N, 4) 代表已分割得到的与x中每个元素（分支）对应的矩形
        #要得到的预览（矩形分割）的形状为：(B, N, 2 ** split_levels, 4)
        B, N, C = x.shape
        ratios = self.project_ratios(x) # (B, N, 2 ** split_levels - 1)
        level_rects = []
        r = r.unsqueeze(2)
        for i in range(self.split_levels):
            r = self.split_rect(r, (self.start_dim + i) % 2, ratios[:, :, (2 ** i - 1):(2 ** (i + 1) - 1)])
            level_rects.append(r)
        # r 为最终得到的预览 (B, N, 2 ** split_levels, 4)
        mask = self.project_mask(x).view(B, N, self.num_rects, self.groups)
        mask = F.softmax(mask, dim = 2)
        s = self.sample_rects(r, I_tuple) 
        s = self.project_samples(s)
        s = s.view(B, N, self.num_rects, self.groups, self.group_channels)
        y = torch.einsum("b n s g, b n s g c -> b n g c", mask, s)
        y = y.view(B, N, C)
        y = self.norm(y) + x
        return y, level_rects

class TPM(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_unfold_layers = 5
        self.unfold_channels = [16, 16, 16, 16, 16] # 各个阶段的channels数量
        self.unfold_groups = [4, 4, 4, 4, 4]
        self.merge_channels = [16, 32, 64, 128] # 按照正常的计算顺序记载
        self.head_channels = 256
        self.split_levels = 3 # 预览的层数
        
        self.query = nn.Parameter(torch.zeros(1, 1, self.unfold_channels[0]))
        self.query_rect = nn.Parameter(torch.tensor([0., 0., 1., 1.], requires_grad = False))
        
        self.admit_layers = nn.ModuleList()
        self.unfold_mlp_layers = nn.ModuleList()
        self.split_layers = nn.ModuleList()
        for i in range(self.num_unfold_layers):
            # admit -> mlp -> split
            self.admit_layers.append(Admit(
                in_channels, 
                self.unfold_channels[i], 
                self.unfold_groups[i], 
                i % 2, 
                split_levels = self.split_levels
            ))
            self.unfold_mlp_layers.append(nn.Sequential(
                nn.Linear(self.unfold_channels[i], 4 * self.unfold_channels[i]),
                nn.GELU(),
                nn.Linear(4 * self.unfold_channels[i], self.unfold_channels[i]),
                nn.LayerNorm(self.unfold_channels[i])
            ))
            if i < self.num_unfold_layers - 1:
                self.split_layers.append(Split(self.unfold_channels[i], self.unfold_channels[i + 1]))
        self.merge_layers = nn.ModuleList()
        in_channels = self.unfold_channels[-1]
        for i in range(self.num_unfold_layers - 1):
            out_channels = self.merge_channels[i]
            self.merge_layers.append(Merge(in_channels, out_channels, 2, use_norm = True)) ###### 用了norm，看看是否合适
            in_channels = out_channels
        self.head = nn.Sequential( #### 仿照OverLoCK的结构
            nn.Linear(self.merge_channels[-1], self.head_channels),
            nn.LayerNorm(self.head_channels),
            nn.SiLU(),
            Merge(in_channels = self.head_channels, out_channels = self.num_classes, factor = 1)
        )
    def forward(self, I):
        IX = torch.cumsum(I, dim = 2)
        IY = torch.cumsum(I, dim = 3)
        IT = torch.cumsum(IX, dim = 3)
        I_tuple = (I, IX, IY, IT)
        B = I.shape[0]
        
        # unfold
        x = self.query.expand(B, self.query.shape[1], self.query.shape[2])
        r = self.query_rect.view(1, 1, 4).expand(B, 1, 4)
        for i in range(self.num_unfold_layers):
            x, level_rects = self.admit_layers[i](x, r, I_tuple)
            x = self.unfold_mlp_layers[i](x) + x ####### 注意残差连接
            if i < self.num_unfold_layers - 1:
                r = level_rects[0].view(x.shape[0], x.shape[1] * 2, 4) # (B, N, 2, 4) -> (B, N * 2, 4) 注意此处同一分支的子分支相连排列
                x = self.split_layers[i](x)
                
        for i in range(self.num_unfold_layers - 1):
            x = self.merge_layers[i](x)
        # fuse
        # print(x.shape)
        scores = self.head(x).squeeze()
        # print('score:', scores.shape)
        # print('x:', x.shape)
        return scores
def tpm_xxt(num_classes = 20):
    model = TPM(
        in_channels = 3,
        num_classes = num_classes
    )
    return model