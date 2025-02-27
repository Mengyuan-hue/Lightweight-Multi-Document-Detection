# import random
#
# from fightingcv_attention.attention.EMSA import EMSA
# import torch
# from torch import nn
# from torch.nn import functional as F
#
# shape = (50, 64, 512)
# generated_tuple = tuple([random.random() for _ in range(shape[0])])
# emsa = EMSA(d_model=512, d_k=512, d_v=512, h=8, H=8, W=8, ratio=2, apply_transform=True)
# output = emsa(input, input, input)
# print(output.shape)

#import torch
# import numpy as np
#
# # 假设你有一个 16*8*8*1280 大小的列表
# your_list = np.random.rand(16, 8, 8, 1280)
#
# # 将 NumPy 数组转换为 PyTorch 张量
# your_tensor = torch.tensor(your_list)
#
# # 打印张量的形状
# print(your_tensor.shape)

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from fightingcv_attention.attention.SelfAttention  import ScaledDotProductAttention
from mmpose.registry import MODELS

tensor = torch.randn(16, 1280, 8, 8)
sub_tensors = torch.split(tensor, 1, dim=0)  # 沿着第一个维度分割成16个张量
# 对每个子张量进行 Self-Attention 计算（示例中采用简单的线性变换代替注意力机制的计算）
processed_sub_tensors = []
for sub_tensor in sub_tensors:
    sa = ScaledDotProductAttention(d_model=8, d_k=8, d_v=8, h=8)
    output = sa(sub_tensor[0],sub_tensor[0],sub_tensor[0])
    processed_sub_tensors.append(output)
stacked_tensor = torch.stack(processed_sub_tensors, dim=0)

