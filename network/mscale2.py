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


This is an alternative implementation of mscale, where we feed pairs of 
features from both lower and higher resolution images into the attention head.
"""
import torch
from torch import nn

from network.mynn import initialize_weights, Norm2d, Upsample
from network.mynn import ResizeX, scale_as
from network.utils import get_aspp, get_trunk
from network.utils import make_seg_head, make_attn_head
from config import cfg


class MscaleBase(nn.Module):
    """
    Multi-scale attention segmentation model base class
    """
    def __init__(self):
        super(MscaleBase, self).__init__()
        self.criterion = None

    def _fwd(self, x):
        pass

    def nscale_forward(self, inputs, scales):
        """
        Hierarchical attention, primarily used for getting best inference
        results.

        We use attention at multiple scales, giving priority to the lower
        resolutions. For example, if we have 4 scales {0.5, 1.0, 1.5, 2.0},
        then evaluation is done as follows:

              p_joint = attn_1.5 * p_1.5 + (1 - attn_1.5) * down(p_2.0)
              p_joint = attn_1.0 * p_1.0 + (1 - attn_1.0) * down(p_joint)
              p_joint = up(attn_0.5 * p_0.5) * (1 - up(attn_0.5)) * p_joint

        The target scale is always 1.0, and 1.0 is expected to be part of the
        list of scales. When predictions are done at greater than 1.0 scale,
        the predictions are downsampled before combining with the next lower
        scale.

        Inputs:
          scales - a list of scales to evaluate
          inputs - dict containing 'images', the input, and 'gts', the ground
                   truth mask

        Output:
          If training, return loss, else return prediction + attention
        """
        x_1x = inputs['images']

        assert 1.0 in scales, 'expected 1.0 to be the target scale'
        # Lower resolution provides attention for higher rez predictions,
        # so we evaluate in order: high to low
        scales = sorted(scales, reverse=True)
        pred = None
        last_feats = None

        for idx, s in enumerate(scales):
            x = ResizeX(x_1x, s)
            p, feats = self._fwd(x)

            # Generate attention prediction
            if idx > 0:
                assert last_feats is not None
                # downscale feats
                last_feats = scale_as(last_feats, feats)
                cat_feats = torch.cat([feats, last_feats], 1)
                attn = self.scale_attn(cat_feats)
                attn = scale_as(attn, p)

            if pred is None:
                # This is the top scale prediction
                pred = p
            elif s >= 1.0:
                # downscale previous
                pred = scale_as(pred, p)
                pred = attn * p + (1 - attn) * pred
            else:
                # upscale current
                p = attn * p
                p = scale_as(p, pred)
                attn = scale_as(attn, pred)
                pred = p + (1 - attn) * pred

            last_feats = feats

        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            loss = self.criterion(pred, gts)
            return loss
        else:
            # FIXME: should add multi-scale values for pred and attn
            return {'pred': pred,
                    'attn_10x': attn}

    def two_scale_forward(self, inputs):
        assert 'images' in inputs

        x_1x = inputs['images']
        x_lo = ResizeX(x_1x, cfg.MODEL.MSCALE_LO_SCALE)

        p_lo, feats_lo = self._fwd(x_lo)
        p_1x, feats_hi = self._fwd(x_1x)

        feats_hi = scale_as(feats_hi, feats_lo)
        cat_feats = torch.cat([feats_lo, feats_hi], 1)
        logit_attn = self.scale_attn(cat_feats)
        logit_attn = scale_as(logit_attn, p_lo)

        p_lo = logit_attn * p_lo
        p_lo = scale_as(p_lo, p_1x)
        logit_attn = scale_as(logit_attn, p_1x)
        joint_pred = p_lo + (1 - logit_attn) * p_1x

        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            loss = self.criterion(joint_pred, gts)
            return loss
        else:
            # FIXME: should add multi-scale values for pred and attn
            return {'pred': joint_pred,
                    'attn_10x': logit_attn}

    def forward(self, inputs):
        if cfg.MODEL.N_SCALES and not self.training:
            return self.nscale_forward(inputs, cfg.MODEL.N_SCALES)

        return self.two_scale_forward(inputs)


class MscaleV3Plus(MscaleBase):
    """
    DeepLabV3Plus-based mscale segmentation model
    """
    def __init__(self, num_classes, trunk='wrn38', criterion=None):
        super(MscaleV3Plus, self).__init__()
        self.criterion = criterion
        self.backbone, s2_ch, _s4_ch, high_level_ch = get_trunk(trunk)
        self.aspp, aspp_out_ch = get_aspp(high_level_ch,
                                          bottleneck_ch=256,
                                          output_stride=8)
        self.bot_fine = nn.Conv2d(s2_ch, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(aspp_out_ch, 256, kernel_size=1, bias=False)

        # Semantic segmentation prediction head
        self.final = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        # Scale-attention prediction head
        scale_in_ch = 2 * (256 + 48)

        self.scale_attn = nn.Sequential(
            nn.Conv2d(scale_in_ch, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, bias=False),
            nn.Sigmoid())

        if cfg.OPTIONS.INIT_DECODER:
            initialize_weights(self.bot_fine)
            initialize_weights(self.bot_aspp)
            initialize_weights(self.scale_attn)
            initialize_weights(self.final)
        else:
            initialize_weights(self.final)

    def _fwd(self, x):
        x_size = x.size()
        s2_features, _, final_features = self.backbone(x)
        aspp = self.aspp(final_features)

        conv_aspp = self.bot_aspp(aspp)
        conv_s2 = self.bot_fine(s2_features)
        conv_aspp = Upsample(conv_aspp, s2_features.size()[2:])
        cat_s4 = [conv_s2, conv_aspp]
        cat_s4 = torch.cat(cat_s4, 1)

        final = self.final(cat_s4)
        out = Upsample(final, x_size[2:])

        return out, cat_s4


def DeepV3R50(num_classes, criterion):
    return MscaleV3Plus(num_classes, trunk='resnet-50', criterion=criterion)


class Basic(MscaleBase):
    """
    """
    def __init__(self, num_classes, trunk='hrnetv2', criterion=None):
        super(Basic, self).__init__()
        self.criterion = criterion
        self.backbone, _, _, high_level_ch = get_trunk(
            trunk_name=trunk, output_stride=8)

        self.cls_head = make_seg_head(in_ch=high_level_ch, bot_ch=256,
                                      out_ch=num_classes)
        self.scale_attn = make_attn_head(in_ch=high_level_ch * 2, bot_ch=256,
                                         out_ch=1)

    def two_scale_forward(self, inputs):
        assert 'images' in inputs

        x_1x = inputs['images']
        x_lo = ResizeX(x_1x, cfg.MODEL.MSCALE_LO_SCALE)

        p_lo, feats_lo = self._fwd(x_lo)
        p_1x, feats_hi = self._fwd(x_1x)

        feats_lo = scale_as(feats_lo, feats_hi)
        cat_feats = torch.cat([feats_lo, feats_hi], 1)
        logit_attn = self.scale_attn(cat_feats)
        logit_attn_lo = scale_as(logit_attn, p_lo)
        logit_attn_1x = scale_as(logit_attn, p_1x)

        p_lo = logit_attn_lo * p_lo
        p_lo = scale_as(p_lo, p_1x)
        joint_pred = p_lo + (1 - logit_attn_1x) * p_1x

        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            loss = self.criterion(joint_pred, gts)
            return loss
        else:
            return joint_pred, logit_attn_1x

    def _fwd(self, x, aspp_lo=None, aspp_attn=None, scale_float=None):
        _, _, final_features = self.backbone(x)
        pred = self.cls_head(final_features)
        pred = scale_as(pred, x)

        return pred, final_features


def HRNet(num_classes, criterion, s2s4=None):
    return Basic(num_classes=num_classes, criterion=criterion,
                 trunk='hrnetv2')
