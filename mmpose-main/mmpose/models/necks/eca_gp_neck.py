# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from fightingcv_attention.attention.ECAAttention  import ECAAttention
from mmpose.registry import MODELS


@MODELS.register_module()
class ECA_GP(nn.Module):
    def __init__(self):
        super().__init__()
        self.eca=ECAAttention(kernel_size=3)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs):
        """Forward function."""

        if isinstance(inputs, tuple):
            inputs = tuple([self.eca(x) for x in inputs])
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, list):
            inputs = [self.eca(x) for x in inputs]
            outs = [self.gap(x) for x in inputs]
            outs = [out.view(x.size(0), -1) for out, x in zip(outs, inputs)]
        elif isinstance(inputs, torch.Tensor):
            inputs=self.eca(inputs)
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
