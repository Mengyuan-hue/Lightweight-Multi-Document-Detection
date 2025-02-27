import torch
# from light_cnns import mbv1
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmpose.registry import MODELS

from .base_backbone import BaseBackbone
def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class DepthSepConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False,
                 channel_multiplier=1.0, pw_kernel_size=1):
        super(DepthSepConv, self).__init__()

        self.conv_dw = nn.Conv2d(
            int(in_channels * channel_multiplier), int(in_channels * channel_multiplier), kernel_size,
            stride=stride, groups=int(in_channels * channel_multiplier), dilation=dilation, padding=padding)

        self.conv_pw = nn.Conv2d(int(in_channels * channel_multiplier), out_channels, pw_kernel_size, padding=padding,
                                 bias=bias)

        self.relu = nn.ReLU(inplace=True)

    @property
    def in_channels(self):
        return self.conv_dw.in_channels

    @property
    def out_channels(self):
        return self.conv_pw.out_channels

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        x = self.relu(x)
        return x

@MODELS.register_module()
class MobileNetV1(BaseBackbone):
    def __init__(self,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 norm_eval=False,
                 with_cp=False,
                 ):
        super(MobileNetV1, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.model = nn.Sequential(
            DepthSepConv(32, 64, stride=1),
            DepthSepConv(64, 128, stride=2),
            DepthSepConv(128, 128, stride=1),
            DepthSepConv(128, 256, stride=2),
            DepthSepConv(256, 256, stride=1),
            DepthSepConv(256, 512, stride=2),
            DepthSepConv(512, 512, stride=1),
            DepthSepConv(512, 512, stride=1),
            DepthSepConv(512, 512, stride=1),
            DepthSepConv(512, 512, stride=1),
            DepthSepConv(512, 512, stride=1),
            DepthSepConv(512, 1024, stride=2),
            DepthSepConv(1024, 1024, stride=1),
        )
    def forward(self, x):
        print("************",x.shape)
        x = self.conv1(x)
        print("************",x.shape)
        x = self.model(x)
        print("************",x.shape)
        x=[x]
        return tuple(x)

def mbv1(**kwargs):
    return MobileNetV1(**kwargs)