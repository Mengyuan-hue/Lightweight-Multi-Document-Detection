# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from fightingcv_attention.attention.SKAttention  import SKAttention
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmpose.registry import MODELS
@MODELS.register_module()
class SPPBottleneck(BaseModule):

    """Spatial pyramid pooling layer used in YOLOv3-SPP.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (tuple[int]): Sequential of kernel sizes of pooling
            layers. Default: (5, 9, 13).
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(3, 5, 7),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvModule(
            conv2_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        print("shuru",len(x),len(x[0]),len(x[0][0]),len(x[0][0][0]))
        x = self.conv1(x[0])
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.cat(
                [x] + [pooling(x) for pooling in self.poolings], dim=1)
        print("池化后",len(x),len(x[0]),len(x[0][0]),len(x[0][0][0]))
        x = self.conv2(x)
        print("shuchu后",len(x),type(x),len(x[0]),len(x[0][0]),len(x[0][0][0]))
        x = self.gap(x)

        return tuple([x])