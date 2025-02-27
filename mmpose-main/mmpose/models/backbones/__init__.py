# Copyright (c) OpenMMLab. All rights reserved.
from .alexnet import AlexNet
from .cpm import CPM
from .csp_darknet import CSPDarknet
from .cspnext import CSPNeXt
from .dstformer import DSTFormer
from .hourglass import HourglassNet
from .hourglass_ae import HourglassAENet
from .hrformer import HRFormer
from .hrnet import HRNet
from .litehrnet import LiteHRNet
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .mspn import MSPN
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .rsn import RSN
from .scnet import SCNet
from .seresnet import SEResNet
from .seresnext import SEResNeXt
from .shufflenet_v1 import ShuffleNetV1
from .shufflenet_v2 import ShuffleNetV2
from .swin import SwinTransformer
from .tcn import TCN
from .v2v_net import V2VNet
from .vgg import VGG
from .vipnas_mbv3 import ViPNAS_MobileNetV3
from .vipnas_resnet import ViPNAS_ResNet
from .mobilenet_v2_SGE import MobileNetV2_SGE
from .mobilenet_v2_SA import MobileNetV2_SA
from .mobilenet_V2_spp import MobileNetV2_SPP
from .mobilenet_V1 import MobileNetV1
from .ghostNet import GhostNet
from .Res_Add_20 import ResNet_Add
from .mobileNetV2_SGE_4 import MobileNetV2_SGE_4
from .mobileNetV2_SGE_2 import MobileNetV2_SGE_2
from .mobileNet_V1_SGE_1 import MobileNetV2_SGE_1
__all__ = [
    'AlexNet', 'HourglassNet', 'HourglassAENet', 'HRNet', 'MobileNetV2',
    'MobileNetV3', 'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SCNet',
    'SEResNet', 'SEResNeXt', 'ShuffleNetV1', 'ShuffleNetV2', 'CPM', 'RSN',
    'MSPN', 'ResNeSt', 'VGG', 'TCN', 'ViPNAS_ResNet', 'ViPNAS_MobileNetV3',
    'LiteHRNet', 'V2VNet', 'HRFormer', 'PyramidVisionTransformer',
    'PyramidVisionTransformerV2', 'SwinTransformer', 'DSTFormer', 'CSPDarknet',
    'CSPNeXt','MobileNetV2_SGE','MobileNetV2_SA','MobileNetV2_SPP','MobileNetV1',
    'GhostNet','ResNet_Add','MobileNetV2_SGE_4','MobileNetV2_SGE_2','MobileNetV2_SGE_1'
]
