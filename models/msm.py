import torch
import torch.nn.functional as F
from torch import nn
import itertools
# class UnfoldLayer(nn.Module):
    # def __init__(
        # self, 
        # in_channels, 
        # up_channels, 
        # down_channels, 
        # bunch_relations, 
        # up_bunches,
        # num_rects, 
        # groups
        # ):
        # super().__init__()
        # self.in_channels = in_channels
        # self.num_rects = num_rects
        # self.up_channels = up_channels
        # self.up_bunches = tuple(up_bunches)
        # self.down_channels = down_channels
        # self.bunch_relations = bunch_relations
        # self.groups = groups
        # self.group_channels = down_channels // self.groups
        # self.to_numbers = torch.sum(bunch_relations, dim = 1) # 每个上层元素生成的下层元素数，根据组别区分
        # self.interactions = nn.ModuleList()
        # self.interaction_channels = 2 * up_channels ##############################
        # for to_number in self.to_numbers:
            # self.interactions.append(nn.Sequential(
                # nn.Linear(up_channels, self.interaction_channels * to_number),
                # # nn.GELU()
            # )) 
            # # self.interactions.append(nn.Sequential(
                
                # # # nn.Linear(up_channels, up_channels),
                # # # nn.GELU(),
                # # nn.Linear(up_channels, self.interaction_channels * to_number),
                # # nn.GELU(),
            # # )) 
        # # self.norm0 = nn.LayerNorm(self.interaction_channels)
        # self.index_rect = nn.Linear(self.interaction_channels // 2, num_rects * 4)
        # self.project_rects = nn.Sequential(
            # nn.Linear(in_channels, down_channels),
            # nn.GELU(),
        # )
        # self.mask = nn.Linear(self.interaction_channels // 2, num_rects * self.groups)
        # self.norm1 = nn.LayerNorm(down_channels)
        # self.mlp = nn.Sequential(
            # nn.Linear(down_channels, 4 * down_channels),
            # nn.GELU(),
            # nn.Linear(4 * down_channels, down_channels)
        # )
        # self.norm2 = nn.LayerNorm(down_channels)
    # def sample_rects_topleft(self, I_tuple, x, y):
        # I, IX, IY, IT = I_tuple
        # # I: (B, in_channels, H, W)
        # # x, y:(B, number)
        # # -> s: (B, in_channels, number)
        # B, N = x.shape
        # x = x.view(B, 1, N)
        # y = y.view(B, 1, N)
        # X = torch.ceil(x).to(torch.int)
        # Y = torch.ceil(y).to(torch.int)
        # zeros = torch.zeros_like(X)
        # dx = X - x
        # dy = Y - y
        # # 索引广播
        # ind_b = torch.arange(B, device = I.device).view(B, 1, 1)
        # ind_c = torch.arange(self.in_channels, device = I.device).view(1, self.in_channels, 1)
        
        # wx1 = 0.5 * dx ** 2
        # wx2 = dx - wx1
        # wy1 = 0.5 * dy ** 2
        # wy2 = dy - wy1
        
        # s_a = IT[ind_b, ind_c, X, Y]
        # s_e1 = (
            # wy1 * (IX[ind_b, ind_c, X, Y - 1] - 0.5 * I[ind_b, ind_c, X, Y - 1] - 0.5 * I[ind_b, ind_c, zeros, Y - 1]) +
            # wy2 * (IX[ind_b, ind_c, X, Y] - 0.5 * I[ind_b, ind_c, X, Y] - 0.5 * I[ind_b, ind_c, zeros, Y])
        # )
        # s_e2 = (
            # wx1 * (IY[ind_b, ind_c, X - 1, Y] - 0.5 * I[ind_b, ind_c, X - 1, Y] - 0.5 * I[ind_b, ind_c, X - 1, zeros]) +
            # wx2 * (IY[ind_b, ind_c, X, Y] - 0.5 * I[ind_b, ind_c, X, Y] - 0.5 * I[ind_b, ind_c, X, zeros]) 
        # )
        # s_c = (
            # wx1 * wy1 * I[ind_b, ind_c, X - 1, Y - 1] + wx2 * wy1 * I[ind_b, ind_c, X, Y - 1] + 
            # wx1 * wy2 * I[ind_b, ind_c, X - 1, Y] + wx2 * wy2 * I[ind_b, ind_c, X, Y]
        # )
        # return s_a - s_e1 - s_e2 + s_c
    # def sample_rects(self, p, I_tuple):
        # # I: (B, in_channels, H, W), p: (B, M, num_rects, 4) 
        # I, IX, IY, IT = I_tuple
        # B, M, num_rects, _ = p.shape
        # _, __, H, W = I.shape
        # fac = torch.tensor([H - 1, W - 1, H - 1, W - 1]).view(1, 1, 1, 4).to(I.device)
        # p = F.sigmoid(p) * fac
        # p = p.view(B, M * self.num_rects, 4)
        # x1 = p[:, :, 0] # (B, M * num_rects)
        # y1 = p[:, :, 1]
        # x2 = p[:, :, 2]
        # y2 = p[:, :, 3]
        # S = (
            # self.sample_rects_topleft(I_tuple, x2, y2) 
            # - self.sample_rects_topleft(I_tuple, x1, y2) 
            # - self.sample_rects_topleft(I_tuple, x2, y1)
            # + self.sample_rects_topleft(I_tuple, x1, y1)
        # ) # (B, in_channels, M * num_rects)
        # S = S.view(B, self.in_channels, M, self.num_rects)
        # S = S.permute(0, 2, 3, 1) # (B, M, num_rects, in_channels)
        # return S
    # def forward(self, x, I_tuple):
        # # x: (B, N, C) -> UnfoldLayer(x): (B, M, C')
        # B, N, C = x.shape
        # # interactions 
        # bunches = torch.split(x, self.up_bunches, dim = 1)
        # y_sections = []
        # for i, (bunch, interaction, to_number) in enumerate(zip(bunches, self.interactions, self.to_numbers)):
            # _, n, __ = bunch.shape
            # y_ = interaction(bunch) # (B, n_i, to_number * self.interaction_channels)
            # # print('!', y_.shape)
            # # print('!!', (B, n * to_number, C))
            # y_ = y_.view(B, n * to_number, self.interaction_channels) ##### 2 * 
            # y_sections.append(torch.split(y_, tuple(self.bunch_relations[i] * n), dim = 1)) # 把这一上层组别的n个元素生成的元素按各个下层组别区分开来
        # # y_sections: (k, l)
        # y_sections = list(itertools.chain(*zip(*y_sections)))
        # y = torch.cat(y_sections, dim = 1) #(B, M, C)
        # # y = self.norm0(y) ##### 
        # y1, y2 = torch.chunk(y, 2, dim = 2)
        # # y1 = y2 = y
        # #######
        # M = y.shape[1]
        # # index_rect
        # p = self.index_rect(y1).view(B, y1.shape[1], self.num_rects, 4) # (B, M, num_rects * 4)
        # rects = self.sample_rects(p, I_tuple) # (B, M, num_rects, in_channels)
        # rects = self.project_rects(rects)
        # mask = self.mask(y2) # (B, M, num_rects * groups)
        # mask = mask.view(B, M, self.num_rects, self.groups, 1)
        # mask = F.softmax(mask, 2)
        # # print('!', rects.shape)
        # # print('!!', (B, M, self.num_rects, self.groups, self.group_channels))
        # rects = rects.view(B, M, self.num_rects, self.groups, self.group_channels)
        # z = (mask * rects).sum(2).view(B, M, self.down_channels)
        # z = self.norm1(z)
        # #z = z + y #############  if interaction_channels == down_channels，残差连接
        # identity = z
        # z = self.mlp(z)
        # z = self.norm2(z) + identity
        # return z 
class UnfoldLayer(nn.Module):
    def __init__(
        self, 
        in_channels, 
        up_channels, 
        down_channels, 
        bunch_relations, 
        up_bunches,
        num_rects, 
        groups
        ):
        super().__init__()
        self.in_channels = in_channels
        self.num_rects = num_rects
        self.up_channels = up_channels
        self.up_bunches = tuple(up_bunches)
        self.down_channels = down_channels
        self.bunch_relations = bunch_relations
        self.groups = groups
        self.group_channels = down_channels // self.groups
        self.to_numbers = torch.sum(bunch_relations, dim = 1) # 每个上层元素生成的下层元素数，根据组别区分
        self.interactions = nn.ModuleList()
        # self.interaction_channels = 2 * up_channels ##############################
        self.interaction_channels = self.num_rects * 4 + self.num_rects * self.groups
        for to_number in self.to_numbers:
            self.interactions.append(nn.Sequential(
                nn.Linear(up_channels, self.interaction_channels * to_number),
                # nn.GELU()
            )) 
            # self.interactions.append(nn.Sequential(
                
                # # nn.Linear(up_channels, up_channels),
                # # nn.GELU(),
                # nn.Linear(up_channels, self.interaction_channels * to_number),
                # nn.GELU(),
            # )) 
        # self.norm0 = nn.LayerNorm(self.interaction_channels)
        # self.index_rect = nn.Linear(self.interaction_channels // 2, num_rects * 4)
        self.project_rects = nn.Sequential(
            nn.Linear(in_channels, down_channels),
            nn.GELU(),
        )
        # self.mask = nn.Linear(self.interaction_channels // 2, num_rects * self.groups)
        self.norm1 = nn.LayerNorm(down_channels)
        self.mlp = nn.Sequential(
            nn.Linear(down_channels, 4 * down_channels),
            nn.GELU(),
            nn.Linear(4 * down_channels, down_channels)
        )
        self.norm2 = nn.LayerNorm(down_channels)
    def sample_rects_topleft(self, I_tuple, x, y):
        I, IX, IY, IT = I_tuple
        # I: (B, in_channels, H, W)
        # x, y:(B, number)
        # -> s: (B, in_channels, number)
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
        # I: (B, in_channels, H, W), p: (B, M, num_rects, 4) 
        I, IX, IY, IT = I_tuple
        B, M, num_rects, _ = p.shape
        _, __, H, W = I.shape
        fac = torch.tensor([H - 1, W - 1, H - 1, W - 1]).view(1, 1, 1, 4).to(I.device)
        p = F.sigmoid(p) * fac
        p = p.view(B, M * self.num_rects, 4)
        x1 = p[:, :, 0] # (B, M * num_rects)
        y1 = p[:, :, 1]
        x2 = p[:, :, 2]
        y2 = p[:, :, 3]
        S = (
            self.sample_rects_topleft(I_tuple, x2, y2) 
            - self.sample_rects_topleft(I_tuple, x1, y2) 
            - self.sample_rects_topleft(I_tuple, x2, y1)
            + self.sample_rects_topleft(I_tuple, x1, y1)
        ) # (B, in_channels, M * num_rects)
        S = S.view(B, self.in_channels, M, self.num_rects)
        S = S.permute(0, 2, 3, 1) # (B, M, num_rects, in_channels)
        return S
    def forward(self, x, I_tuple):
        # x: (B, N, C) -> UnfoldLayer(x): (B, M, C')
        B, N, C = x.shape
        # interactions 
        bunches = torch.split(x, self.up_bunches, dim = 1)
        y_sections = []
        for i, (bunch, interaction, to_number) in enumerate(zip(bunches, self.interactions, self.to_numbers)):
            _, n, __ = bunch.shape
            y_ = interaction(bunch) # (B, n_i, to_number * self.interaction_channels)
            # print('!', y_.shape)
            # print('!!', (B, n * to_number, C))
            y_ = y_.view(B, n * to_number, self.interaction_channels) ##### 2 * 
            y_sections.append(torch.split(y_, tuple(self.bunch_relations[i] * n), dim = 1)) # 把这一上层组别的n个元素生成的元素按各个下层组别区分开来
        # y_sections: (k, l)
        y_sections = list(itertools.chain(*zip(*y_sections)))
        y = torch.cat(y_sections, dim = 1) #(B, M, C)
        M = y.shape[1]
        ############### 
        p, mask = torch.split(y, [self.num_rects * 4, self.num_rects * self.groups], dim = 2)
        p = p.view(B, p.shape[1], self.num_rects, 4) # (B, M, num_rects, 4)
        rects = self.sample_rects(p, I_tuple) # (B, M, num_rects, in_channels)
        rects = self.project_rects(rects)
        rects = rects.view(B, M, self.num_rects, self.groups, self.group_channels)
        mask = mask.view(B, M, self.num_rects, self.groups)
        mask = F.softmax(mask, 2)
        z = torch.einsum("b m s g, b m s g c -> b m g c", mask, rects)
        z = z.view(B, M, self.down_channels)
        z = self.norm1(z)
        #z = z + y #############  if interaction_channels == down_channels，残差连接
        identity = z
        z = self.mlp(z)
        z = self.norm2(z) + identity
        return z 
class FuseLayer(nn.Module):
    def __init__(self, down_channels, up_channels, out_channels, bunch_relations, up_bunches, groups):
        super().__init__()
        self.down_channels = down_channels # u_{i} 通道数
        self.up_channels = up_channels # x_{i - 1}
        self.out_channels = out_channels # u_{i - 1}
        self.bunch_relations = bunch_relations
        self.to_numbers = torch.sum(bunch_relations, dim = 1)
        self.up_bunches = tuple(up_bunches)
        
        self.c_q = down_channels ##### 在新的设计中，fuse层因为要累积信息所以通道数更多一些，而unfold层不需要，所以通道数少一些
        self.project_key = nn.Linear(down_channels, self.c_q) # 从u_{i}投影得到key
        self.project_query = nn.Linear(up_channels, self.c_q) # 暂时以top_channels作为q,k统一的维度
        self.project_value = nn.Linear(down_channels, out_channels)
        self.groups = groups
        self.norm1 = nn.LayerNorm(out_channels)
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, 4 * out_channels),
            nn.GELU(),
            nn.Linear(4 * out_channels, out_channels),
        )
        self.norm2 = nn.LayerNorm(out_channels)
        self.shortcut_project = nn.Linear(up_channels, out_channels)
    def forward(self, u, x):
        # 接受输入：u_{i}, x_{i - 1}，
        # 输出： u_{i - 1}
        # u_{i} : (B, M, E') # 下层特征，E'是U的特征通道数
        # x_{i - 1} : (B, N, C) # 上层特征
        # u_{i - 1} : (B, N, E)
        B, N, C = x.shape
        shortcut = self.shortcut_project(x) ####################
        x_ = torch.split(x, self.up_bunches, dim = 1) # 每个bunch为 (B, n_i, C)形状
        u_sections = []
        for i, bunch in enumerate(x_):
            u_sections.append(self.bunch_relations[i] * bunch.shape[1])
        u_sections = list(itertools.chain(*zip(*u_sections))) # 获得用于切分u的dim 1的信息
        u_ = torch.split(u, u_sections, dim = 1) # 其中每个元素形如 (B, c_{ij} * n_i, E)，j相同的连成一片
        k, l = self.bunch_relations.shape
        u_ = [u_[j * k:(j + 1) * k] for j in range(l)]
        output = []
        for i, secs in enumerate(zip(*u_)):
            u_i = torch.cat(secs, dim = 1) # 把所有的c_{ij} * n_i (1 <= j <= l)个元素拼接在一起，它们用于生成与上层计算注意力所需的k和v
            x_i = x_[i]
            u_i = u_i.view(B, x_i.shape[1], self.to_numbers[i], u_i.shape[-1]) # 转换成 (B, n_i, c_{i1} + ... + c_{il}, E')的形状
            
            k = self.project_key(u_i) # (B, sum_j c_{ij} * n_i, C_Q)
            # print('!', (B, u_i.shape[1], u_i.shape[2], self.groups, self.c_q // self.groups) )
            # print(self.groups, self.c_q)
            # print(k.shape)
            k = k.view(B, u_i.shape[1], u_i.shape[2], self.groups, self.c_q // self.groups) 
            #k: (B, n_i, c_{i1} + ... + c_{il}, G, C_Q / G)
            ######### 还要加入分组（多头）注意力
            q = self.project_query(x_i).view(B, x_i.shape[1], self.groups, self.c_q // self.groups) # (B, n_i, G, C_Q / G)
            attn = torch.einsum('b n u g c, b n g c -> b n u g', k, q) / (self.c_q ** 0.5) # (B, n_i, c_{i1} + ... + c_{il}, G)
            attn = torch.softmax(attn, dim = 2) 
            v = self.project_value(u_i).view(B, u_i.shape[1], u_i.shape[2], self.groups, self.out_channels // self.groups) 
            #v : (B, n_i, c_{i1} + ... + c_{il}, G, E / G)
            out = torch.einsum('b n u g, b n u g c -> b n g c', attn, v) # (B, n_i, G, E / G)
            out = out.view(B, out.shape[1], self.out_channels) # 堆叠各组通道 (B, n_i, E)
            output.append(out)
        z = torch.cat(output, dim = 1)
        z = self.norm1(z)
        # identity = z
        z = self.mlp(z)
        # z = self.norm2(z) + identity
        z = self.norm2(z) + shortcut
        return z
class MSM(nn.Module):
    def __init__(self, 
        in_channels = 3,         
        query_channels = 128, # 初始的query嵌入维度
        unfold_channels = [64, 32, 16, 8],
        unfold_groups = [16, 8, 4, 2], 
        fuse_channels = [64, 32, 16, 8], 
        end_channels = 128, 
        fuse_groups = [16, 8, 4, 2], 
        query_bunches = [0, 1], # 默认设置：query有1个2-分支，没有1-分支
        bunch_relations = (
            [[2, 0],
             [1, 2]],
            [[2, 0],
             [1, 2]],
            [[2, 0],
             [1, 2]],
            [[2, 0],
             [1, 2]]
        ),
        num_rects = [3, 3, 3, 3],
        num_classes = 20,

        ):
        super().__init__()
        self.num_classes = num_classes
        self.query = nn.Parameter(torch.zeros(1, sum(query_bunches), query_channels))
        
        self.unfold_layers = nn.ModuleList()
        
        up_channels = query_channels
        up_bunches = torch.tensor(query_bunches)
        
        for i, rel in enumerate(bunch_relations):
            rel = torch.tensor(rel)
            down_bunches = up_bunches @ rel
            down_channels = unfold_channels[i]
            self.unfold_layers.append(UnfoldLayer(
                in_channels = in_channels, 
                up_channels = up_channels, 
                down_channels = down_channels,
                up_bunches = up_bunches,
                bunch_relations = rel,
                num_rects = num_rects[i],
                groups = unfold_groups[i],
            ))
            
            up_channels = down_channels
            up_bunches = down_bunches
        
        self.middle_project = nn.Sequential(
            nn.GELU(), 
            nn.Linear(unfold_channels[-1], fuse_channels[-1]),
        )
        
        self.fuse_layers = nn.ModuleList()
        up_channels = query_channels
        up_bunches = torch.tensor(query_bunches)
        out_channels = end_channels
        for i, rel in enumerate(bunch_relations):
            rel = torch.tensor(rel)
            down_bunches = up_bunches @ rel
            down_channels = fuse_channels[i]
            self.fuse_layers.append(FuseLayer(
                up_channels = up_channels, # X_{i - 1}
                down_channels = down_channels, # U_{i}
                out_channels = out_channels, # U_{i - 1}
                bunch_relations = rel, # i - 1层 到 i层 
                up_bunches = up_bunches,
                groups = fuse_groups[i]
            ))
            up_channels = unfold_channels[i] # X_{i}作为更下层的输入
            out_channels = down_channels # fusepath从下到上加工，更下层的输出(U_{i})作为当前层的输入
            up_bunches = down_bunches
        self.end_channels = end_channels
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(end_channels, num_classes)
        )
    def forward_unfold(self, I):
        # I: (B, in_channels, H, W)
        IX = torch.cumsum(I, dim = 2)
        IY = torch.cumsum(I, dim = 3)
        IT = torch.cumsum(IX, dim = 3)
        I_tuple = (I, IX, IY, IT)
        B = I.shape[0]
        xs = []
        x = self.query.expand(B, -1, self.query.shape[-1])
        for i, layer in enumerate(self.unfold_layers):
            x = layer(x, I_tuple)
            xs.append(x)
        return xs
    def forward_fuse(self, xs):
        us = []
        n = len(xs)
        u = self.middle_project(xs[-1])
        B = u.shape[0]
        
        for i in range(n - 1, -1, -1):
            if i == 0: # 当前u对应于u_i，要与x_{i-1}融合
                x = self.query.expand(B, -1, self.query.shape[-1])
            else:
                x = xs[i - 1]
            u = self.fuse_layers[i](u, x)
            us.append(u)
        return us
    def forward(self, I):
        # print("I:", I.shape)
        ###### 可能得留意空分支的处理（长度为0的Tensor？）
        xs = self.forward_unfold(I)
        us = self.forward_fuse(xs)
        u_0 = us[-1].squeeze() # (B, E_0)
        cls = self.head(u_0) # 最后一层（最终的融合结果）输入head进行分类
        # print('cls:', cls.shape)
        return cls
def msm_xxt(num_classes = 20):
    # model = MSM(
        # in_channels = 3,         
        # query_channels = 8, # 初始的query嵌入维度
        # unfold_channels = [8, 8, 8, 8],
        # unfold_groups = [2, 2, 2, 2], 
        # fuse_channels = [64, 32, 16, 8], 
        # end_channels = 128, 
        # fuse_groups = [16, 8, 4, 2], 
        # query_bunches = [0, 0, 1], # 默认设置：query有1个3-分支，没有1-分支或2-分支
        # bunch_relations = (
            # [[1, 0, 0],
             # [1, 1, 0],
             # [1, 1, 1]],
            # [[1, 0, 0],
             # [1, 1, 0],
             # [1, 1, 1]],
            # [[1, 0, 0],
             # [1, 1, 0],
             # [1, 1, 1]],
            # [[1, 0, 0],
             # [1, 1, 0],
             # [1, 1, 1]],
        # ),
        # num_rects = [3, 3, 3, 3],
        # num_classes = 20,
    # )
    
    # model = MSM(
        # in_channels = 3,         
        # query_channels = 8, # 初始的query嵌入维度
        # unfold_channels = [8, 8, 8, 8],
        # unfold_groups = [2, 2, 2, 2], 
        # fuse_channels = [64, 32, 16, 8], 
        # end_channels = 128, 
        # fuse_groups = [16, 8, 4, 2], 
        # query_bunches = [0, 1], # 默认设置：query有1个2-分支，没有1-分支
        # bunch_relations = (
            # [[1, 0],
             # [1, 1]],
            # [[1, 0],
             # [1, 1]],
            # [[1, 0],
             # [1, 1]],
            # [[1, 0],
             # [1, 1]]
        # ),
        # num_rects = [3, 3, 3, 3],
        # num_classes = 20,
    # )
    # model = MSM(
        # in_channels = 3,         
        # query_channels = 16, # 初始的query嵌入维度
        # unfold_channels = [16, 16, 16, 16, 16, 16],
        # unfold_groups = [4, 4, 4, 4, 4, 4], 
        # fuse_channels = [64, 32, 16, 16, 16, 16], 
        # end_channels = 128, 
        # fuse_groups = [16, 8, 4, 4, 4, 4], 
        # query_bunches = [0, 1], # 默认设置：query有1个2-分支，没有1-分支
        # bunch_relations = (
            # [[1, 0],
             # [1, 1]],
            # [[1, 0],
             # [1, 1]],
            # [[1, 0],
             # [1, 1]],
            # [[1, 0],
             # [1, 1]],
            # [[1, 0],
             # [1, 1]],
            # [[1, 0],
             # [1, 1]]
        # ),
        # num_rects = [3, 3, 3, 3, 3, 3],
        # num_classes = 20,
    # )
    
    # model = MSM(
        # in_channels = 3,         
        # query_channels = 16, # 初始的query嵌入维度
        # unfold_channels = [16, 16, 16, 16],
        # unfold_groups = [4, 4, 4, 4], 
        # fuse_channels = [64, 32, 16, 16], 
        # end_channels = 128, 
        # fuse_groups = [16, 8, 4, 4], 
        # query_bunches = [0, 1], # 默认设置：query有1个2-分支，没有1-分支
        # bunch_relations = (
            # [[0, 0],
             # [0, 2]],
            # [[0, 0],
             # [0, 2]],
            # [[0, 0],
             # [0, 2]],
            # [[0, 0],
             # [0, 2]]
        # ),
        # num_rects = [3, 3, 3, 3],
        # num_classes = 20,
    # )
    # model = MSM(
        # in_channels = 3,         
        # query_channels = 128,
        # unfold_channels = [64, 32, 16, 8],
        # unfold_groups = [16, 8, 4, 2], 
        # fuse_channels = [64, 32, 16, 8], 
        # end_channels = 128, 
        # fuse_groups = [16, 8, 4, 2], 
        # query_bunches = [0, 1], 
        # bunch_relations = (
            # [[1, 0],
             # [1, 1]],
            # [[1, 0],
             # [1, 1]],
            # [[1, 0],
             # [1, 1]],
            # [[1, 0],
             # [1, 1]]
        # ),
        # num_rects = [3, 3, 3, 3],
        # num_classes = 20,
    # )
    # model = MSM(
        # in_channels = 3,         
        # query_channels = 16, # 初始的query嵌入维度
        # unfold_channels = [16, 16, 16, 16],
        # unfold_groups = [4, 4, 4, 4], 
        # fuse_channels = [64, 32, 16, 16], 
        # end_channels = 128, 
        # fuse_groups = [16, 8, 4, 4], 
        # query_bunches = [0, 1], # 默认设置：query有1个2-分支，没有1-分支
        # bunch_relations = (
            # [[1, 0],
             # [1, 1]],
            # [[1, 0],
             # [1, 1]],
            # [[1, 0],
             # [1, 1]],
            # [[1, 0],
             # [1, 1]]
        # ),
        # num_rects = [8, 8, 8, 8],
        # num_classes = 20,
    # )
    model = MSM(
        in_channels = 3,         
        query_channels = 128, 
        unfold_channels = [64, 32, 16, 8],
        unfold_groups = [16, 8, 4, 2], 
        fuse_channels = [64, 32, 16, 8], 
        end_channels = 128, 
        fuse_groups = [16, 8, 4, 2], 
        query_bunches = [1], 
        bunch_relations = (
            [[2]],
            [[2]],
            [[2]],
            [[2]],
        ),
        num_rects = [24, 12, 6, 3],
        num_classes = 20
    )
    # model = MSM(
        # in_channels = 3,         
        # query_channels = 128, 
        # unfold_channels = [64, 32, 16, 8],
        # unfold_groups = [16, 8, 4, 2], 
        # fuse_channels = [64, 32, 16, 8], 
        # end_channels = 128, 
        # fuse_groups = [16, 8, 4, 2], 
        # query_bunches = [0, 1], 
        # bunch_relations = (
            # [[2, 0],
             # [1, 2]],
            # [[2, 0],
             # [1, 2]],
            # [[2, 0],
             # [1, 2]],
            # [[2, 0],
             # [1, 2]]
        # ),
        # num_rects = [24, 12, 6, 3],
        # num_classes = 20
    # )
    # model = MSM(
        # in_channels = 3,         
        # query_channels = 16, # 初始的query嵌入维度
        # unfold_channels = [16, 16, 16, 16],
        # unfold_groups = [4, 4, 4, 4], 
        # fuse_channels = [64, 32, 16, 16], 
        # end_channels = 128, 
        # fuse_groups = [16, 8, 4, 4], 
        # query_bunches = [0, 1], # 默认设置：query有1个2-分支，没有1-分支
        # bunch_relations = (
            # [[1, 0],
             # [1, 1]],
            # [[1, 0],
             # [1, 1]],
            # [[1, 0],
             # [1, 1]],
            # [[1, 0],
             # [1, 1]]
        # ),
        # num_rects = [3, 3, 3, 3],
        # num_classes = 20,
    # )
    return model
