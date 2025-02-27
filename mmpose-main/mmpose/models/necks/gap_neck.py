# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmpose.registry import MODELS


@MODELS.register_module()
class GlobalAveragePooling(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.
    """

    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs):
        """Forward function."""
        # print("mobileV2Netbackbone 输入为256*256*3时的输出为：",len(inputs)," ",len(inputs[0])," ",len(inputs[0][0]))
        if isinstance(inputs, tuple):
            # print("mobileV2Netbackbone使用的是：tuple" )
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, list):
            # print("mobileV2Netbackbone使用的是：list" )
            outs = [self.gap(x) for x in inputs]
            outs = [out.view(x.size(0), -1) for out, x in zip(outs, inputs)]
        elif isinstance(inputs, torch.Tensor):
            # print("mobileV2Netbackbone使用的是：Tensor" )
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')

        # print("out的shape：", len(outs), type(outs)," ", len(outs[0]),type(outs[0]) ," ", len(outs[0][0]),type(outs[0][0]) , " ", len(outs[0][0][0]),type(outs[0][0][0]) ," ", len(outs[0][0][0][0]))
        print("neck",type(outs),len(outs),type(outs[0]),outs[0].shape)
        return outs
