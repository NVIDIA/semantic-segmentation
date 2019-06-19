'''
Portions of this code are adapted from:
https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py
https://github.com/ClementPinard/FlowNetPytorch/blob/master/models/FlowNetS.py
'''
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import init
import os

from models.model_utils import conv2d, deconv2d

from flownet2_pytorch.models import FlowNet2
from flownet2_pytorch.networks.resample2d_package.resample2d import Resample2d


class SDCNet2D(nn.Module):
    def __init__(self, args):
        super(SDCNet2D,self).__init__()

        self.rgb_max = args.rgb_max
        self.sequence_length = args.sequence_length

        factor = 2
        input_channels = self.sequence_length * 3 + (self.sequence_length - 1) * 2

        self.conv1 = conv2d(input_channels, 64 // factor, kernel_size=7, stride=2)
        self.conv2 = conv2d(64 // factor, 128 // factor, kernel_size=5, stride=2)
        self.conv3 = conv2d(128 // factor, 256 // factor, kernel_size=5, stride=2)
        self.conv3_1 = conv2d(256 // factor, 256 // factor)
        self.conv4 = conv2d(256 // factor, 512 // factor, stride=2)
        self.conv4_1 = conv2d(512 // factor, 512 // factor)
        self.conv5 = conv2d(512 // factor, 512 // factor, stride=2)
        self.conv5_1 = conv2d(512 // factor, 512 // factor)
        self.conv6 = conv2d(512 // factor, 1024 // factor, stride=2)
        self.conv6_1 = conv2d(1024 // factor, 1024 // factor)

        self.deconv5 = deconv2d(1024 // factor, 512 // factor)
        self.deconv4 = deconv2d(1024 // factor, 256 // factor)
        self.deconv3 = deconv2d(768 // factor, 128 // factor)
        self.deconv2 = deconv2d(384 // factor, 64 // factor)
        self.deconv1 = deconv2d(192 // factor, 32 // factor)
        self.deconv0 = deconv2d(96 // factor, 16 // factor)

        self.final_flow = nn.Conv2d(input_channels + 16 // factor, 2,
                                    kernel_size=3, stride=1, padding=1, bias=True)


        # init parameters, when doing convtranspose3d, do bilinear init
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose3d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

        self.flownet2 = FlowNet2(args, batchNorm=False)
        assert os.path.exists(args.flownet2_checkpoint), "flownet2 checkpoint must be provided."
        flownet2_checkpoint = torch.load(args.flownet2_checkpoint)
        self.flownet2.load_state_dict(flownet2_checkpoint['state_dict'], strict=False)

        for param in self.flownet2.parameters():
            param.requires_grad = False

        self.warp_nn = Resample2d(bilinear=False)
        self.warp_bilinear = Resample2d(bilinear=True)

        self.L1Loss = nn.L1Loss()

        flow_mean = torch.FloatTensor([-0.94427323, -1.23077035]).view(1, 2, 1, 1)
        flow_std = torch.FloatTensor([13.77204132, 7.47463894]).view(1, 2, 1, 1)
        rgb_mean = torch.FloatTensor([106.7747911, 96.13649598, 76.61428884]).view(1, 3, 1, 1)

        self.register_buffer('flow_mean', flow_mean)
        self.register_buffer('flow_std', flow_std)
        self.register_buffer('rgb_mean', rgb_mean)

        self.ignore_keys = ['flownet2']
        return

    def interframe_optical_flow(self, input_images):

        #FIXME: flownet2 implementation expects RGB images, 
        # while image formats for both SDCNet and DeepLabV3 expects BGR.
        # input_images = [torch.flip(input_image, dims=[1]) for input_image in input_images]

        # Create image pairs for flownet, then merge batch and frame dimension
        # so theres only a single call to flownet2 is done.
        flownet2_inputs = torch.stack(
            [torch.cat([input_images[i + 1].unsqueeze(2), input_images[i].unsqueeze(2)], dim=2) for i in
             range(0, self.sequence_length - 1)], dim=0).contiguous()

        batch_size, channel_count, height, width = input_images[0].shape
        flownet2_inputs_flattened = flownet2_inputs.view(-1,channel_count, 2, height, width)
        flownet2_outputs = [self.flownet2(flownet2_input) for flownet2_input in
                            torch.chunk(flownet2_inputs_flattened, self.sequence_length - 1)]

        #FIXME: flipback images to BGR, 
        # input_images = [torch.flip(input_image, dims=[1]) for input_image in input_images]

        return flownet2_outputs

    def network_output(self, input_images, input_flows):

        # Normalize input flows
        input_flows = [(input_flow - self.flow_mean) / (3 * self.flow_std) for
                       input_flow in input_flows]

        # Normalize input via flownet2-type normalisation
        concated_images = torch.cat([image.unsqueeze(2) for image in input_images], dim=2).contiguous()
        rgb_mean = concated_images.view(concated_images.size()[:2] + (-1,)).mean(dim=-1).view(
            concated_images.size()[:2] + 2 * (1,))
        input_images = [(input_image - rgb_mean) / self.rgb_max for input_image in input_images]
        bsize, channels, height, width = input_flows[0].shape

        # Atypical concatenation of input images along channels (done for compatibility with pre-trained models)
        # for two rgb images, concated channels would appear as (r1r2g1g2b1b2),
        # instaed of typical (r1g1b1r2g2b2) that can be obtained by torch.cat(..,dim=1)
        input_images = torch.cat([input_image.unsqueeze(2) for input_image in input_images], dim=2)
        input_images = input_images.contiguous().view(bsize, -1, height, width)

        # same atypical concatenation done for input flows.
        input_flows = torch.cat([input_flow.unsqueeze(2) for input_flow in input_flows], dim=2)
        input_flows = input_flows.contiguous().view(bsize, -1, height, width)

        # Network input
        images_and_flows = torch.cat((input_flows, input_images), dim=1)

        out_conv1 = self.conv1(images_and_flows)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        out_deconv5 = self.deconv5(out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5), 1)

        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4), 1)

        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3), 1)

        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)

        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)

        out_deconv0 = self.deconv0(concat1)

        concat0 = torch.cat((images_and_flows, out_deconv0), 1)
        output_flow = self.final_flow(concat0)

        flow_prediction = 3 * self.flow_std * output_flow + self.flow_mean

        return flow_prediction

    def prepare_inputs(self, input_dict):
        images = input_dict['image']  # expects a list

        input_images = images[:-1]

        target_image = images[-1]

        last_image = (input_images[-1]).clone()

        return input_images, last_image, target_image

    def forward(self, input_dict, label_image=None):

        input_images, last_image, target_image = self.prepare_inputs(input_dict)

        input_flows = self.interframe_optical_flow(input_images)

        flow_prediction = self.network_output(input_images, input_flows)

        image_prediction = self.warp_bilinear(last_image, flow_prediction)

        if label_image is not None:
            label_prediction = self.warp_nn(label_image, flow_prediction)

        # calculate losses
        losses = {}

        losses['color'] = self.L1Loss(image_prediction/self.rgb_max, target_image/self.rgb_max)

        losses['color_gradient'] = self.L1Loss(torch.abs(image_prediction[...,1:] - image_prediction[...,:-1]), \
                                               torch.abs(target_image[...,1:] - target_image[...,:-1])) + \
                                   self.L1Loss(torch.abs(image_prediction[..., 1:,:] - image_prediction[..., :-1,:]), \
                                               torch.abs(target_image[..., 1:,:] - target_image[..., :-1,:]))

        losses['flow_smoothness'] = self.L1Loss(flow_prediction[...,1:], flow_prediction[...,:-1]) + \
                                    self.L1Loss(flow_prediction[..., 1:,:], flow_prediction[..., :-1,:])

        losses['tot'] = 0.7 * losses['color'] + 0.2 * losses['color_gradient'] + 0.1 * losses['flow_smoothness']

        if label_image is not None:
            image_prediction = label_prediction

        return losses, image_prediction, target_image


class SDCNet2DRecon(SDCNet2D):
    def __init__(self, args):
        args.sequence_length += 1
        super(SDCNet2DRecon, self).__init__(args)

    def prepare_inputs(self, input_dict):
        images = input_dict['image']  # expects a list

        input_images = images

        target_image = images[-1]

        last_image = (input_images[-2]).clone()

        return input_images, last_image, target_image