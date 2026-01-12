import models
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torch
import numpy as np
model = models.tpm_xxt(num_classes = 20)
cnt = np.sum([p.numel() for p in model.parameters()]).item()
print("#Param:", cnt)

tensor = (torch.rand(1, 3, 64, 64),)
flops = FlopCountAnalysis(model, tensor)
print("FLOPs:", flops.total())