# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from fightingcv_attention.attention.PolarizedSelfAttention import ParallelPolarizedSelfAttention,SequentialPolarizedSelfAttention
from mmpose.registry import MODELS


@MODELS.register_module()
class POLA_GAP(nn.Module):
    def __init__(self,channel):
        super().__init__()
        self.psa=SequentialPolarizedSelfAttention(channel=channel)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs):
        """Forward function."""

        if isinstance(inputs, tuple):
            inputs = tuple([self.psa(x) for x in inputs])
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, list):
            inputs = [self.psa(x) for x in inputs]
            outs = [self.gap(x) for x in inputs]
            outs = [out.view(x.size(0), -1) for out, x in zip(outs, inputs)]
        elif isinstance(inputs, torch.Tensor):
            inputs=self.psa(inputs)
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
