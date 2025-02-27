# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from fightingcv_attention.attention.CBAM import CBAMBlock
from mmpose.registry import MODELS


@MODELS.register_module()
class CBAM_GP(nn.Module):
    def __init__(self,channel,reduction,kernel_size):
        super().__init__()
        self.cbam=CBAMBlock(channel=channel, reduction=reduction,kernel_size=kernel_size)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs):
        """Forward function."""

        if isinstance(inputs, tuple):
            inputs = tuple([self.cbam(x) for x in inputs])
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, list):
            inputs = [self.cbam(x) for x in inputs]
            outs = [self.gap(x) for x in inputs]
            outs = [out.view(x.size(0), -1) for out, x in zip(outs, inputs)]
        elif isinstance(inputs, torch.Tensor):
            inputs=self.cbam(inputs)
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
