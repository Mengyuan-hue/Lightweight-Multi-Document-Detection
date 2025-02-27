# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from fightingcv_attention.attention.EMSA import EMSA
from mmpose.registry import MODELS


@MODELS.register_module()
class EMAS_GP(nn.Module):
    def __init__(self,d_model, d_k, d_v, h, H, W, ratio, apply_transform):
        super().__init__()
        self.emsa=EMSA(d_model=d_model, d_k=d_k, d_v=d_v, h=h,H=H,W=W,ratio=ratio,apply_transform=apply_transform)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs):
        """Forward function."""

        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, list):
            outs = [self.gap(x) for x in inputs]
            outs = [out.view(x.size(0), -1) for out, x in zip(outs, inputs)]
        elif isinstance(inputs, torch.Tensor):
            emsa_output=self.emsa(inputs)
            outs = self.gap(emsa_output)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
