import torch
import torch.nn as nn

# 创建一个形状为4x64x64x62的张量
input_tensor = torch.randn(4, 64, 64, 62)

# 计算需要填充的数量
padding = (0, 0, 0, 0, 0, 0, 0, 2)  # 对第四维进行填充，左右各填充0个单位，上下各填充2个单位

# 创建一个ZeroPad3d层，对第四维进行填充
zero_pad = nn.(padding)

# 对输入张量进行填充操作
padded_tensor = zero_pad(input_tensor)

# 输出填充后的张量的形状
print(padded_tensor.shape)  # 输出形状为4x64x64x64