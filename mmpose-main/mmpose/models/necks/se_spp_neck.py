# coding=utf-8

import math
import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmpose.registry import MODELS
from fightingcv_attention.attention.SEAttention import SEAttention
@MODELS.register_module()
class SE_SPPLayer(BaseModule):

    def __init__(self, num_levels,channel, reduction,pool_type='max_pool'):
        super(SE_SPPLayer, self).__init__()
        self.se=SEAttention(channel=channel, reduction=reduction)
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        x=x[0]
        x=self.se(x)
        num, c, h, w = x.size()  # num:样本数量 c:通道数 h:高 w:宽
        for i in range(self.num_levels):
            level = i + 1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (
            math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)

            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
                #orint(x_flatten.shape)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
                #orint(x_flatten.shape)
        return tuple([x_flatten])