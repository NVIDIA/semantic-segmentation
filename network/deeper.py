"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import torch
from torch import nn
from network.mynn import Upsample2
from network.utils import ConvBnRelu, get_trunk, get_aspp


class DeeperS8(nn.Module):
    """
    Panoptic DeepLab-style semantic segmentation network
    stride8 only
    """
    def __init__(self, num_classes, trunk='wrn38', criterion=None):
        super(DeeperS8, self).__init__()

        self.criterion = criterion
        self.trunk, s2_ch, s4_ch, high_level_ch = get_trunk(trunk_name=trunk,
                                                            output_stride=8)
        self.aspp, aspp_out_ch = get_aspp(high_level_ch, bottleneck_ch=256,
                                          output_stride=8)

        self.convs2 = nn.Conv2d(s2_ch, 32, kernel_size=1, bias=False)
        self.convs4 = nn.Conv2d(s4_ch, 64, kernel_size=1, bias=False)
        self.conv_up1 = nn.Conv2d(aspp_out_ch, 256, kernel_size=1, bias=False)
        self.conv_up2 = ConvBnRelu(256 + 64, 256, kernel_size=5, padding=2)
        self.conv_up3 = ConvBnRelu(256 + 32, 256, kernel_size=5, padding=2)
        self.conv_up5 = nn.Conv2d(256, num_classes, kernel_size=1, bias=False)

    def forward(self, inputs, gts=None):
        assert 'images' in inputs
        x = inputs['images']

        s2_features, s4_features, final_features = self.trunk(x)
        s2_features = self.convs2(s2_features)
        s4_features = self.convs4(s4_features)
        aspp = self.aspp(final_features)
        x = self.conv_up1(aspp)
        x = Upsample2(x)
        x = torch.cat([x, s4_features], 1)
        x = self.conv_up2(x)
        x = Upsample2(x)
        x = torch.cat([x, s2_features], 1)
        x = self.conv_up3(x)
        x = self.conv_up5(x)
        x = Upsample2(x)

        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            return self.criterion(x, gts)
        return {'pred': x}


def DeeperW38(num_classes, criterion, s2s4=True):
    return DeeperS8(num_classes, criterion=criterion, trunk='wrn38')


def DeeperX71(num_classes, criterion, s2s4=True):
    return DeeperS8(num_classes, criterion=criterion, trunk='xception71')


def DeeperEffB4(num_classes, criterion, s2s4=True):
    return DeeperS8(num_classes, criterion=criterion, trunk='efficientnet_b4')
