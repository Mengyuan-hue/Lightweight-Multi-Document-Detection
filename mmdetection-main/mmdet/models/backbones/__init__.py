# Copyright (c) OpenMMLab. All rights reserved.
from .csp_darknet import CSPDarknet
from .cspnext import CSPNeXt
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .efficientnet import EfficientNet
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .trident_resnet import TridentResNet
from .csp_darknet_thin_aspp import CSPDarknet_Thin_Aspp
from .csp_darknet_thin import CSPDarknet_Thin
from .csp_darknet_cross_thin_aspp import CSPDarknet_Cross_Thin_Aspp
from .csp_darknet_thin_cross import CSPDarknet_Cross_Thin
from .csp_darknet_thin_aspp1234 import CSPDarknet_Thin_Aspp1234
from .csp_darktnet_thin_cross_aspp1234 import CSPDarknet_Cross_Thin_ASPP1234
__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'PyramidVisionTransformer',
    'PyramidVisionTransformerV2', 'EfficientNet', 'CSPNeXt','CSPDarknet_Thin','CSPDarknet_Thin_Aspp',
    'CSPDarknet_Cross_Thin_Aspp','CSPDarknet_Cross_Thin','CSPDarknet_Thin_Aspp1234','CSPDarknet_Cross_Thin_ASPP1234'
]
