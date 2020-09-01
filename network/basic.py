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
from torch import nn

from network.mynn import initialize_weights, Upsample
from network.mynn import scale_as
from network.utils import get_aspp, get_trunk, make_seg_head
from config import cfg


class Basic(nn.Module):
    """
    Basic segmentation network, no ASPP, no Mscale
    """
    def __init__(self, num_classes, trunk='hrnetv2', criterion=None):
        super(Basic, self).__init__()
        self.criterion = criterion
        self.backbone, _, _, high_level_ch = get_trunk(
            trunk_name=trunk, output_stride=8)
        self.seg_head = make_seg_head(in_ch=high_level_ch,
                                      out_ch=num_classes)
        initialize_weights(self.seg_head)

    def forward(self, inputs):
        x = inputs['images']
        _, _, final_features = self.backbone(x)
        pred = self.seg_head(final_features)
        pred = scale_as(pred, x)

        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            loss = self.criterion(pred, gts)
            return loss
        else:
            output_dict = {'pred': pred}
            return output_dict


class ASPP(nn.Module):
    """
    ASPP-based Segmentation network
    """
    def __init__(self, num_classes, trunk='hrnetv2', criterion=None):
        super(ASPP, self).__init__()
        self.criterion = criterion
        self.backbone, _, _, high_level_ch = get_trunk(trunk)
        self.aspp, aspp_out_ch = get_aspp(high_level_ch,
                                          bottleneck_ch=cfg.MODEL.ASPP_BOT_CH,
                                          output_stride=8)
        self.bot_aspp = nn.Conv2d(aspp_out_ch, 256, kernel_size=1, bias=False)
        self.final = make_seg_head(in_ch=256,
                                   out_ch=num_classes)

        initialize_weights(self.final, self.bot_aspp, self.aspp)

    def forward(self, inputs):
        x = inputs['images']
        x_size = x.size()

        _, _, final_features = self.backbone(x)
        aspp = self.aspp(final_features)
        aspp = self.bot_aspp(aspp)
        pred = self.final(aspp)
        pred = Upsample(pred, x_size[2:])

        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            loss = self.criterion(pred, gts)
            return loss
        else:
            output_dict = {'pred': pred}
            return output_dict


def HRNet(num_classes, criterion, s2s4=None):
    return Basic(num_classes=num_classes, criterion=criterion,
                 trunk='hrnetv2')


def HRNet_ASP(num_classes, criterion, s2s4=None):
    return ASPP(num_classes=num_classes, criterion=criterion,
                trunk='hrnetv2')
