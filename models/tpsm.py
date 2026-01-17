import torch
import torch.nn.functional as F
from torch import nn
import itertools
# 考虑把Admit和MLP合并到一个模块中（TPSBlock）
# 而为了能方便复用矩形的加工计算（矩形拆分、矩形置换），就将它们抽象为独立于各个模块的函数
# 另外，sample_rects和sample_rects也和模块没有什么关系、是无参数的，所以考虑把这两个函数也抽出来
class TPBlock(nn.Module):
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
        self.norm1 = nn.LayerNorm(channels)
class TPSM(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
def tpsm_xxt(num_classes = 20):
    model = TPSM(
        in_channels = 3,
        num_classes = num_classes
    )
    return model