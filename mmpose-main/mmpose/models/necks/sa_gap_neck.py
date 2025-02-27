# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from fightingcv_attention.attention.SelfAttention  import ScaledDotProductAttention
from mmpose.registry import MODELS


@MODELS.register_module()
class SA_GP(nn.Module):
    def __init__(self,cd_model, d_k, d_v, h):
        super().__init__()
        self.sa=ScaledDotProductAttention(d_model=cd_model, d_k=d_k, d_v=d_v, h=h)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        pass
    def sa_long(self,inputs):
        sub_tensors = torch.split(inputs, 1, dim=0)  # 沿着第一个维度分割成16个张量
        processed_sub_tensors = []
        for sub_tensor in sub_tensors:
            output = self.sa(sub_tensor[0], sub_tensor[0], sub_tensor[0])
            processed_sub_tensors.append(output)
        stacked_tensor = torch.stack(processed_sub_tensors, dim=0)
        return stacked_tensor
    def forward(self, inputs):
        """Forward function."""

        if isinstance(inputs, tuple):

            inputs = tuple([self.sa_long(x) for x in inputs])
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, list):
            inputs = [self.sa(x) for x in inputs]
            outs = [self.gap(x) for x in inputs]
            outs = [out.view(x.size(0), -1) for out, x in zip(outs, inputs)]
        elif isinstance(inputs, torch.Tensor):
            inputs=self.sa(inputs)
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
