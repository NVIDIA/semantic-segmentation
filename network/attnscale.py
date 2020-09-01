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

from network.mynn import initialize_weights, Norm2d, Upsample
from network.mynn import ResizeX, scale_as
from network.utils import get_aspp, get_trunk
from config import cfg


class ASDV3P(nn.Module):
    """
    DeepLabV3+ with Attention-to-scale style attention

    Attn head:
    conv 3x3 512 ch
    relu
    conv 1x1 3   ch -> 1.0, 0.75, 0.5

    train with 3 output scales: 0.5, 1.0, 2.0
    min/max scale aug set to [0.5, 1.0]
    """
    def __init__(self, num_classes, trunk='wrn38', criterion=None,
                 use_dpc=False, fuse_aspp=False, attn_2b=False, bn_head=False):
        super(ASDV3P, self).__init__()
        self.criterion = criterion
        self.fuse_aspp = fuse_aspp
        self.attn_2b = attn_2b
        self.backbone, s2_ch, _s4_ch, high_level_ch = get_trunk(trunk)
        self.aspp, aspp_out_ch = get_aspp(high_level_ch,
                                          bottleneck_ch=256,
                                          output_stride=8,
                                          dpc=use_dpc)
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
        assert cfg.MODEL.N_SCALES is not None
        self.scales = sorted(cfg.MODEL.N_SCALES)

        num_scales = len(self.scales)
        if cfg.MODEL.ATTNSCALE_BN_HEAD or bn_head:
            self.scale_attn = nn.Sequential(
                nn.Conv2d(num_scales * (256 + 48), 256, kernel_size=3,
                          padding=1, bias=False),
                Norm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                Norm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_scales, kernel_size=1, bias=False))
        else:
            self.scale_attn = nn.Sequential(
                nn.Conv2d(num_scales * (256 + 48), 512, kernel_size=3,
                          padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, num_scales, kernel_size=1, padding=1,
                          bias=False))

        if cfg.OPTIONS.INIT_DECODER:
            initialize_weights(self.bot_fine)
            initialize_weights(self.bot_aspp)
            initialize_weights(self.scale_attn)
            initialize_weights(self.final)
        else:
            initialize_weights(self.final)

    def _fwd(self, x, aspp_lo=None, aspp_attn=None):
        """
        Run the network, and return final feature and logit predictions
        """
        x_size = x.size()
        s2_features, _, final_features = self.backbone(x)
        aspp = self.aspp(final_features)

        if self.fuse_aspp and \
           aspp_lo is not None and aspp_attn is not None:
            aspp_attn = scale_as(aspp_attn, aspp)
            aspp_lo = scale_as(aspp_lo, aspp)
            aspp = aspp_attn * aspp_lo + (1 - aspp_attn) * aspp

        conv_aspp = self.bot_aspp(aspp)
        conv_s2 = self.bot_fine(s2_features)
        conv_aspp = Upsample(conv_aspp, s2_features.size()[2:])
        cat_s4 = [conv_s2, conv_aspp]
        cat_s4 = torch.cat(cat_s4, 1)
        final = self.final(cat_s4)
        out = Upsample(final, x_size[2:])

        return out, cat_s4

    def _forward_fused(self, inputs):
        """
        Combine multiple scales of predictions together with attention
        predicted jointly off of multi-scale features.
        """
        x_1x = inputs['images']

        # run 1x scale
        assert 1.0 in self.scales, 'expected one of scales to be 1.0'
        ps = {}
        ps[1.0], feats_1x = self._fwd(x_1x)
        concat_feats = [feats_1x]

        # run all other scales
        for scale in self.scales:
            if scale == 1.0:
                continue
            resized_x = ResizeX(x_1x, scale)
            p, feats = self._fwd(resized_x)
            ps[scale] = scale_as(p, x_1x)
            feats = scale_as(feats, feats_1x)
            concat_feats.append(feats)

        concat_feats = torch.cat(concat_feats, 1)
        attn_tensor = self.scale_attn(concat_feats)

        output = None
        for idx, scale in enumerate(self.scales):
            attn = attn_tensor[:, idx:idx+1, :, :]
            attn_1x_scale = scale_as(attn, x_1x)
            if output is None:
                # logx.msg(f'ps[scale] shape {ps[scale].shape} '
                #         f'attn shape {attn_1x_scale.shape}')
                output = ps[scale] * attn_1x_scale
            else:
                output += ps[scale] * attn_1x_scale

        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            loss = self.criterion(output, gts)

            if cfg.LOSS.SUPERVISED_MSCALE_WT:
                for scale in self.scales:
                    loss_scale = self.criterion(ps[scale], gts, do_rmi=False)
                    loss += cfg.LOSS.SUPERVISED_MSCALE_WT * loss_scale
            return loss
        else:
            return output, attn

    def forward(self, inputs):
        # FIXME: could add other assets for visualization
        return {'pred': self._forward_fused(inputs)}


def DeepV3R50(num_classes, criterion):
    return ASDV3P(num_classes, trunk='resnet-50', criterion=criterion)


# Batch-norm head
def DeepV3R50B(num_classes, criterion):
    return ASDV3P(num_classes, trunk='resnet-50', criterion=criterion,
                  bn_head=True)


def DeepV3W38(num_classes, criterion):
    return ASDV3P(num_classes, trunk='wrn38', criterion=criterion)


class ASDV3P_Paired(nn.Module):
    """
    DeepLabV3+ with Attention-to-scale style attention

    Attn head:
    conv 3x3 512 ch
    relu
    conv 1x1 3   ch -> 1.0, 0.75, 0.5

    train with 3 output scales: 0.5, 1.0, 2.0
    min/max scale aug set to [0.5, 1.0]
    """
    def __init__(self, num_classes, trunk='wrn38', criterion=None,
                 use_dpc=False, fuse_aspp=False, attn_2b=False, bn_head=False):
        super(ASDV3P_Paired, self).__init__()
        self.criterion = criterion
        self.fuse_aspp = fuse_aspp
        self.attn_2b = attn_2b
        self.backbone, s2_ch, _s4_ch, high_level_ch = get_trunk(trunk)
        self.aspp, aspp_out_ch = get_aspp(high_level_ch,
                                          bottleneck_ch=256,
                                          output_stride=8,
                                          dpc=use_dpc)
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
        assert cfg.MODEL.N_SCALES is not None
        self.trn_scales = (0.5, 1.0)
        self.inf_scales = sorted(cfg.MODEL.N_SCALES)

        num_scales = 2
        if cfg.MODEL.ATTNSCALE_BN_HEAD or bn_head:
            self.scale_attn = nn.Sequential(
                nn.Conv2d(num_scales * (256 + 48), 256, kernel_size=3,
                          padding=1, bias=False),
                Norm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                Norm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_scales, kernel_size=1, bias=False),
                nn.Sigmoid())
        else:
            self.scale_attn = nn.Sequential(
                nn.Conv2d(num_scales * (256 + 48), 512, kernel_size=3,
                          padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, num_scales, kernel_size=1, padding=1,
                          bias=False))

        if cfg.OPTIONS.INIT_DECODER:
            initialize_weights(self.bot_fine)
            initialize_weights(self.bot_aspp)
            initialize_weights(self.scale_attn)
            initialize_weights(self.final)
        else:
            initialize_weights(self.final)

    def _fwd(self, x, aspp_lo=None, aspp_attn=None):
        """
        Run the network, and return final feature and logit predictions
        """
        x_size = x.size()
        s2_features, _, final_features = self.backbone(x)
        aspp = self.aspp(final_features)

        if self.fuse_aspp and \
           aspp_lo is not None and aspp_attn is not None:
            aspp_attn = scale_as(aspp_attn, aspp)
            aspp_lo = scale_as(aspp_lo, aspp)
            aspp = aspp_attn * aspp_lo + (1 - aspp_attn) * aspp

        conv_aspp = self.bot_aspp(aspp)
        conv_s2 = self.bot_fine(s2_features)
        conv_aspp = Upsample(conv_aspp, s2_features.size()[2:])
        cat_s4 = [conv_s2, conv_aspp]
        cat_s4 = torch.cat(cat_s4, 1)
        final = self.final(cat_s4)
        out = Upsample(final, x_size[2:])

        return out, cat_s4

    def _forward_paired(self, inputs, scales):
        """
        Hierarchical form of attention where we only predict attention for
        pairs of scales at a time.

        At inference time we can combine many scales together.
        """
        x_1x = inputs['images']

        # run 1x scale
        assert 1.0 in scales, 'expected one of scales to be 1.0'
        ps = {}
        all_feats = {}
        ps[1.0], all_feats[1.0] = self._fwd(x_1x)

        # run all other scales
        for scale in scales:
            if scale == 1.0:
                continue
            resized_x = ResizeX(x_1x, scale)
            p, feats = self._fwd(resized_x)
            ps[scale] = scale_as(p, x_1x)
            all_feats[scale] = scale_as(feats, all_feats[1.0])

        # Generate all attention outputs
        output = None
        num_scales = len(scales)
        attn = {}
        for idx in range(num_scales - 1):
            lo_scale = scales[idx]
            hi_scale = scales[idx + 1]
            concat_feats = torch.cat([all_feats[lo_scale],
                                      all_feats[hi_scale]], 1)
            p_attn = self.scale_attn(concat_feats)
            attn[lo_scale] = scale_as(p_attn, x_1x)

        # Normalize attentions
        norm_attn = {}
        last_attn = None
        for idx in range(num_scales - 1):
            lo_scale = scales[idx]
            hi_scale = scales[idx + 1]
            attn_lo = attn[lo_scale][:, 0:1, :, :]
            attn_hi = attn[lo_scale][:, 1:2, :, :]
            if last_attn is None:
                norm_attn[lo_scale] = attn_lo
                norm_attn[hi_scale] = attn_hi
            else:
                normalize_this_attn = last_attn / (attn_lo + attn_hi)
                norm_attn[lo_scale] = attn_lo * normalize_this_attn
                norm_attn[hi_scale] = attn_hi * normalize_this_attn
            last_attn = attn_hi

        # Apply attentions
        for idx, scale in enumerate(scales):
            attn = norm_attn[scale]
            attn_1x_scale = scale_as(attn, x_1x)
            if output is None:
                output = ps[scale] * attn_1x_scale
            else:
                output += ps[scale] * attn_1x_scale

        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            loss = self.criterion(output, gts)
            return loss
        else:
            return output, attn

    def forward(self, inputs):
        if self.training:
            return self._forward_paired(inputs, self.trn_scales)
        else:
            return {'pred': self._forward_paired(inputs, self.inf_scales)}


# Batch-norm head with paired attention
def DeepV3R50BP(num_classes, criterion):
    return ASDV3P_Paired(num_classes, trunk='resnet-50', criterion=criterion,
                         bn_head=True)
