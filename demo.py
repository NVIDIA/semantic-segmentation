import os
import sys
import argparse
from PIL import Image
import numpy as np
import cv2

import torch
from torch.backends import cudnn
import torchvision.transforms as transforms

import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg

parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--demo-image', type=str, default='', help='path to demo image', required=True)
parser.add_argument('--snapshot', type=str, default='./pretrained_models/cityscapes_best.pth', help='pre-trained checkpoint', required=True)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus', help='network architecture used for inference')
parser.add_argument('--save-dir', type=str, default='./save', help='path to save your results')
args = parser.parse_args()
assert_and_infer_cfg(args, train_mode=False)
cudnn.benchmark = False
torch.cuda.empty_cache()

# get net
args.dataset_cls = cityscapes
net = network.get_net(args, criterion=None)
net = torch.nn.DataParallel(net).cuda()
print('Net built.')
net, _ = restore_snapshot(net, optimizer=None, snapshot=args.snapshot, restore_optimizer_bool=False)
net.eval()
print('Net restored.')

# get data
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])
img = Image.open(args.demo_image).convert('RGB')
img_tensor = img_transform(img)

# predict
with torch.no_grad():
    img = img_tensor.unsqueeze(0).cuda()
    pred = net(img)
    print('Inference done.')

pred = pred.cpu().numpy().squeeze()
pred = np.argmax(pred, axis=0)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

colorized = args.dataset_cls.colorize_mask(pred)
colorized.save(os.path.join(args.save_dir, 'color_mask.png'))

label_out = np.zeros_like(pred)
for label_id, train_id in args.dataset_cls.id_to_trainid.items():
    label_out[np.where(pred == train_id)] = label_id
    cv2.imwrite(os.path.join(args.save_dir, 'pred_mask.png'), label_out)
print('Results saved.')
