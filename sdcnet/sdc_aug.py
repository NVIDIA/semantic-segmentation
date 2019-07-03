import os 
import sys
import argparse
import cv2
import numpy as np
from PIL import Image
import shutil

import torch
import torch.nn as nn
from torch.autograd import Variable

from models.sdc_net2d import *

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default='', type=str, metavar='PATH', help='path to trained video reconstruction checkpoint')
parser.add_argument('--flownet2_checkpoint', default='', type=str, metavar='PATH', help='path to flownet-2 best checkpoint')
parser.add_argument('--source_dir', default='', type=str, help='directory for data (default: Cityscapes root directory)')
parser.add_argument('--target_dir', default='', type=str, help='directory to save augmented data')
parser.add_argument('--sequence_length', default=2, type=int, metavar="SEQUENCE_LENGTH",
                    help='number of interpolated frames (default : 2)')
parser.add_argument("--rgb_max", type=float, default = 255.)
parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--propagate', type=int, default=3, help='propagate how many steps')
parser.add_argument('--vis', action='store_true', default=False, help='augment color encoded segmentation map')

def get_model():
	model = SDCNet2DRecon(args)
	checkpoint = torch.load(args.pretrained)
	args.start_epoch = 0 if 'epoch' not in checkpoint else checkpoint['epoch']
	state_dict = checkpoint if 'state_dict' not in checkpoint else checkpoint['state_dict']
	model.load_state_dict(state_dict, strict=False)
	print("Loaded checkpoint '{}' (at epoch {})".format(args.pretrained, args.start_epoch))
	return model

def get_data(img1_dir, img2_dir, img3_dir, gt2_color_dir, gt2_labelid_dir):

	img1_rgb = cv2.imread(img1_dir)
	img2_rgb = cv2.imread(img2_dir)
	img3_rgb = cv2.imread(img3_dir)
	gt2_rgb = cv2.imread(gt2_color_dir)
	gt2_labelid = cv2.imread(gt2_labelid_dir, 0)

	img1_rgb = img1_rgb.transpose((2,0,1))
	img2_rgb = img2_rgb.transpose((2,0,1))
	img3_rgb = img3_rgb.transpose((2,0,1))
	gt2_rgb = gt2_rgb.transpose((2,0,1))
	gt2_labelid = np.expand_dims(gt2_labelid, axis=0)

	img1_rgb = np.expand_dims(img1_rgb, axis=0)
	img2_rgb = np.expand_dims(img2_rgb, axis=0)
	img3_rgb = np.expand_dims(img3_rgb, axis=0)
	gt2_rgb = np.expand_dims(gt2_rgb, axis=0)
	gt2_labelid = np.expand_dims(gt2_labelid, axis=0)

	img1_rgb = torch.from_numpy(img1_rgb.astype(np.float32))
	img2_rgb = torch.from_numpy(img2_rgb.astype(np.float32))
	img3_rgb = torch.from_numpy(img3_rgb.astype(np.float32))
	gt2_rgb = torch.from_numpy(gt2_rgb.astype(np.float32))
	gt2_labelid = torch.from_numpy(gt2_labelid.astype(np.float32))

	return img1_rgb, img2_rgb, img3_rgb, gt2_rgb, gt2_labelid

def one_step_augmentation(model, mask_prefix, colormask_prefix, rgb_prefix, sequence_prefix, split, mode, reverse):

	split_dir = os.path.join(args.source_dir, rgb_prefix, split)
	scenes = os.listdir(split_dir)
	scenes.sort()
	for scene in scenes:
		print("Augmenting %s for mode %s" % (scene, mode))
		scene_dir = os.path.join(split_dir, scene)
		frames = os.listdir(scene_dir)
		frames.sort()
		
		for frame in frames:
			seq_info = frame.split("_")
			seq_id2 = seq_info[-2]
			
			seq_id1 = "%06d" % (int(seq_id2) - 1)
			seq_id3 = "%06d" % (int(seq_id2) + 1)
			im1_name = seq_info[0] + "_" + seq_info[1] + "_" + seq_id1 + "_" + seq_info[-1]
			im3_name = seq_info[0] + "_" + seq_info[1] + "_" + seq_id3 + "_" + seq_info[-1]
			source_im1 = os.path.join(args.source_dir, sequence_prefix, split, scene, im1_name)
			source_im2 = os.path.join(scene_dir, frame)
			source_im3 = os.path.join(args.source_dir, sequence_prefix, split, scene, im3_name)
			color_gt_name = seq_info[0] + "_" + seq_info[1] + "_" + seq_id2 + "_gtFine_color.png"
			labelid_gt_name = seq_info[0] + "_" + seq_info[1] + "_" + seq_id2 + "_gtFine_labelIds.png"
			source_color = os.path.join(args.source_dir, mask_prefix, split, scene, color_gt_name)
			source_labelid = os.path.join(args.source_dir, mask_prefix, split, scene, labelid_gt_name)
			
			if not os.path.isfile(source_im1):
				print("%s does not exist" % (source_im1))
				sys.exit()
			if not os.path.isfile(source_im3):
				print("%s does not exist" % (source_im3))
				sys.exit()

			if not reverse:
				img1_rgb, img2_rgb, img3_rgb, gt2_rgb, gt2_labelid = get_data(source_im1, source_im2, source_im3, source_color, source_labelid)
			else:
				img1_rgb, img2_rgb, img3_rgb, gt2_rgb, gt2_labelid = get_data(source_im3, source_im2, source_im1, source_color, source_labelid)

			img1_rgb = Variable(img1_rgb).contiguous().cuda()
			img2_rgb = Variable(img2_rgb).contiguous().cuda()
			img3_rgb = Variable(img3_rgb).contiguous().cuda()
			gt2_rgb = Variable(gt2_rgb).contiguous().cuda()
			gt2_labelid = Variable(gt2_labelid).contiguous().cuda()
			input_dict = {}
			input_dict['image'] = [img1_rgb, img2_rgb, img3_rgb]

			if mode == "rgb_image":
				_, pred3_rgb, _ = model(input_dict)
				pred3_rgb_img = ( pred3_rgb.data.cpu().numpy().squeeze().transpose(1,2,0) ).astype(np.uint8)
				
				if not os.path.exists(os.path.join(args.target_dir, rgb_prefix, split, scene)):
					os.makedirs(os.path.join(args.target_dir, rgb_prefix, split, scene))
				target_im1 = os.path.join(args.target_dir, rgb_prefix, split, scene, im1_name)
				target_im2 = os.path.join(args.target_dir, rgb_prefix, split, scene, frame)
				target_im3 = os.path.join(args.target_dir, rgb_prefix, split, scene, im3_name)
				
				if not reverse:
					shutil.copyfile(source_im2, target_im2)

				if not reverse:
					cv2.imwrite(target_im3, pred3_rgb_img)
				else:
					cv2.imwrite(target_im1, pred3_rgb_img)

			elif mode == "color_segmap":
				_, pred3_colormap, _ = model(input_dict, label_image=gt2_rgb)
				pred3_colormap_img = ( pred3_colormap.data.cpu().numpy().squeeze().transpose(1,2,0) ).astype(np.uint8)

				if not os.path.exists(os.path.join(args.target_dir, colormask_prefix, split, scene)):
					os.makedirs(os.path.join(args.target_dir, colormask_prefix, split, scene))

				target_color = os.path.join(args.target_dir, colormask_prefix, split, scene, color_gt_name)
				
				if not reverse:
					shutil.copyfile(source_color, target_color)
				
				if not reverse:
					color_gt_name3 = seq_info[0] + "_" + seq_info[1] + "_" + seq_id3 + "_gtFine_color.png"
					target_color3 = os.path.join(args.target_dir, colormask_prefix, split, scene, color_gt_name3)
					cv2.imwrite(target_color3, pred3_colormap_img)
				else:
					color_gt_name1 = seq_info[0] + "_" + seq_info[1] + "_" + seq_id1 + "_gtFine_color.png"
					target_color1 = os.path.join(args.target_dir, colormask_prefix, split, scene, color_gt_name1)
					cv2.imwrite(target_color1, pred3_colormap_img)

			elif mode == "labelid":
				_, pred3_labelid, _ = model(input_dict, label_image=gt2_labelid)
				pred3_labelid_img = pred3_labelid.data.cpu().numpy().squeeze().astype(np.uint8)

				if not os.path.exists(os.path.join(args.target_dir, mask_prefix, split, scene)):
					os.makedirs(os.path.join(args.target_dir, mask_prefix, split, scene))
				target_labelid = os.path.join(args.target_dir, mask_prefix, split, scene, labelid_gt_name)

				if not reverse:
					shutil.copyfile(source_labelid, target_labelid)

				if not reverse:
					labelid_gt_name3 = seq_info[0] + "_" + seq_info[1] + "_" + seq_id3 + "_gtFine_labelIds.png"
					target_labelid3 = os.path.join(args.target_dir, mask_prefix, split, scene, labelid_gt_name3)
					cv2.imwrite(target_labelid3, pred3_labelid_img)
				else:
					labelid_gt_name1 = seq_info[0] + "_" + seq_info[1] + "_" + seq_id1 + "_gtFine_labelIds.png"
					target_labelid1 = os.path.join(args.target_dir, mask_prefix, split, scene, labelid_gt_name1)
					cv2.imwrite(target_labelid1, pred3_labelid_img)
			else:
				print("Mode %s is not supported." % (mode))
				sys.exit()

def multi_step_augmentation(model, mask_prefix, colormask_prefix, rgb_prefix, sequence_prefix, split, mode, reverse, propagate):

	if not os.path.exists(args.target_dir):
		print("Perform --propagate 1 first. The augmentation is in an auto-regressive manner. ")

	split_dir = os.path.join(args.source_dir, rgb_prefix, split)
	scenes = os.listdir(split_dir)
	scenes.sort()

	for scene in scenes:
		print("Augmenting %s for mode %s" % (scene, mode))
		scene_dir = os.path.join(split_dir, scene)
		frames = os.listdir(scene_dir)
		frames.sort()
		
		for frame in frames:
			seq_info = frame.split("_")
			seq_id2 = seq_info[-2]
			seq_id1 = "%06d" % (int(seq_id2) - propagate)
			seq_id3 = "%06d" % (int(seq_id2) + propagate)

			middle_seq_id1 = "%06d" % (int(seq_id2) - propagate + 1)
			middle_seq_id3 = "%06d" % (int(seq_id2) + propagate - 1)

			if not reverse:

				im1_name = seq_info[0] + "_" + seq_info[1] + "_" + seq_id2 + "_" + seq_info[-1]
				im2_name = seq_info[0] + "_" + seq_info[1] + "_" + middle_seq_id3 + "_" + seq_info[-1]
				im3_name = seq_info[0] + "_" + seq_info[1] + "_" + seq_id3 + "_" + seq_info[-1]

				source_im1 = os.path.join(args.source_dir, sequence_prefix, split, scene, im1_name)
				source_im2 = os.path.join(args.source_dir, sequence_prefix, split, scene, im2_name)
				source_im3 = os.path.join(args.source_dir, sequence_prefix, split, scene, im3_name)

				color_gt_name = seq_info[0] + "_" + seq_info[1] + "_" + middle_seq_id3 + "_gtFine_color.png"
				labelid_gt_name = seq_info[0] + "_" + seq_info[1] + "_" + middle_seq_id3 + "_gtFine_labelIds.png"
				source_color = os.path.join(args.target_dir, colormask_prefix, split, scene, color_gt_name)
				source_labelid = os.path.join(args.target_dir, mask_prefix, split, scene, labelid_gt_name)
				middle_im2 = os.path.join(args.target_dir, rgb_prefix, split, scene, im2_name)

			else:

				im1_name = seq_info[0] + "_" + seq_info[1] + "_" + seq_id2 + "_" + seq_info[-1]
				im2_name = seq_info[0] + "_" + seq_info[1] + "_" + middle_seq_id1 + "_" + seq_info[-1]
				im3_name = seq_info[0] + "_" + seq_info[1] + "_" + seq_id1 + "_" + seq_info[-1]

				source_im1 = os.path.join(args.source_dir, sequence_prefix, split, scene, im1_name)
				source_im2 = os.path.join(args.source_dir, sequence_prefix, split, scene, im2_name)
				source_im3 = os.path.join(args.source_dir, sequence_prefix, split, scene, im3_name)

				color_gt_name = seq_info[0] + "_" + seq_info[1] + "_" + middle_seq_id1 + "_gtFine_color.png"
				labelid_gt_name = seq_info[0] + "_" + seq_info[1] + "_" + middle_seq_id1 + "_gtFine_labelIds.png"
				source_color = os.path.join(args.target_dir, colormask_prefix, split, scene, color_gt_name)
				source_labelid = os.path.join(args.target_dir, mask_prefix, split, scene, labelid_gt_name)
				middle_im2 = os.path.join(args.target_dir, rgb_prefix, split, scene, im2_name)

			if not os.path.isfile(source_color):
				print("%s does not exist" % (source_color))
				sys.exit()
			if not os.path.isfile(source_labelid):
				print("%s does not exist" % (source_labelid))
				sys.exit()

			img1_rgb, img2_rgb, img3_rgb, gt2_rgb, gt2_labelid = get_data(source_im1, source_im2, source_im3, source_color, source_labelid)
			
			img1_rgb = Variable(img1_rgb).contiguous().cuda()
			img2_rgb = Variable(img2_rgb).contiguous().cuda()
			img3_rgb = Variable(img3_rgb).contiguous().cuda()
			gt2_rgb = Variable(gt2_rgb).contiguous().cuda()
			gt2_labelid = Variable(gt2_labelid).contiguous().cuda()
			input_dict = {}
			input_dict['image'] = [img1_rgb, img2_rgb, img3_rgb]

			if mode == "rgb_image":
				middle_im2_rgb = cv2.imread(middle_im2)
				middle_im2_rgb = middle_im2_rgb.transpose((2,0,1))
				middle_im2_rgb = np.expand_dims(middle_im2_rgb, axis=0)
				middle_im2_rgb = torch.from_numpy(middle_im2_rgb.astype(np.float32))
				middle_im2_rgb = Variable(middle_im2_rgb).contiguous().cuda()

				_, pred3_rgb, _ = model(input_dict, label_image=middle_im2_rgb)
				pred3_rgb_img = ( pred3_rgb.data.cpu().numpy().squeeze().transpose(1,2,0) ).astype(np.uint8)

				target_im3 = os.path.join(args.target_dir, rgb_prefix, split, scene, im3_name)
				cv2.imwrite(target_im3, pred3_rgb_img)

			elif mode == "color_segmap":

				_, pred3_colormap, _ = model(input_dict, label_image=gt2_rgb)
				pred3_colormap_img = ( pred3_colormap.data.cpu().numpy().squeeze().transpose(1,2,0) ).astype(np.uint8)

				if not reverse:
					color_gt_name3 = seq_info[0] + "_" + seq_info[1] + "_" + seq_id3 + "_gtFine_color.png"
					target_color3 = os.path.join(args.target_dir, colormask_prefix, split, scene, color_gt_name3)
					cv2.imwrite(target_color3, pred3_colormap_img)
				else:
					color_gt_name1 = seq_info[0] + "_" + seq_info[1] + "_" + seq_id1 + "_gtFine_color.png"
					target_color1 = os.path.join(args.target_dir, colormask_prefix, split, scene, color_gt_name1)
					cv2.imwrite(target_color1, pred3_colormap_img)

			elif mode == "labelid":

				_, pred3_labelid, _ = model(input_dict, label_image=gt2_labelid)
				pred3_labelid_img = pred3_labelid.data.cpu().numpy().squeeze().astype(np.uint8)
					
				if not reverse:
					labelid_gt_name3 = seq_info[0] + "_" + seq_info[1] + "_" + seq_id3 + "_gtFine_labelIds.png"
					target_labelid3 = os.path.join(args.target_dir, mask_prefix, split, scene, labelid_gt_name3)
					cv2.imwrite(target_labelid3, pred3_labelid_img)
				else:
					labelid_gt_name1 = seq_info[0] + "_" + seq_info[1] + "_" + seq_id1 + "_gtFine_labelIds.png"
					target_labelid1 = os.path.join(args.target_dir, mask_prefix, split, scene, labelid_gt_name1)
					cv2.imwrite(target_labelid1, pred3_labelid_img)
			else:
				print("Mode %s is not supported." % (mode))
				sys.exit()

if __name__ == '__main__':
	global args
	args = parser.parse_args()

	# Load pre-trained video reconstruction model
	net = get_model()
	net.eval()
	net = net.cuda()

	# Config paths
	if not os.path.exists(args.target_dir):
		os.makedirs(args.target_dir)

	mask_prefix = "gtFine"
	colormask_prefix = "gtFineColor"
	rgb_prefix = "leftImg8bit"
	sequence_prefix = "leftImg8bit_sequence"
	split = "train"

	# Generate augmented dataset
	# create +-1 data
	one_step_augmentation(net, mask_prefix, colormask_prefix, rgb_prefix, sequence_prefix, split, 'rgb_image', False)
	one_step_augmentation(net, mask_prefix, colormask_prefix, rgb_prefix, sequence_prefix, split, 'labelid', False)
	one_step_augmentation(net, mask_prefix, colormask_prefix, rgb_prefix, sequence_prefix, split, 'rgb_image', True)
	one_step_augmentation(net, mask_prefix, colormask_prefix, rgb_prefix, sequence_prefix, split, 'labelid', True)

	if args.vis:
		one_step_augmentation(net, mask_prefix, colormask_prefix, rgb_prefix, sequence_prefix, split, 'color_segmap', False)
		one_step_augmentation(net, mask_prefix, colormask_prefix, rgb_prefix, sequence_prefix, split, 'color_segmap', True)
	
	for i in range(2, args.propagate+1):
		# create +-n data
		multi_step_augmentation(net, mask_prefix, colormask_prefix, rgb_prefix, sequence_prefix, split, 'rgb_image', False, i)
		multi_step_augmentation(net, mask_prefix, colormask_prefix, rgb_prefix, sequence_prefix, split, 'labelid', False, i)
		multi_step_augmentation(net, mask_prefix, colormask_prefix, rgb_prefix, sequence_prefix, split, 'rgb_image', True, i)
		multi_step_augmentation(net, mask_prefix, colormask_prefix, rgb_prefix, sequence_prefix, split, 'labelid', True, i)
		if args.vis:
			multi_step_augmentation(net, mask_prefix, colormask_prefix, rgb_prefix, sequence_prefix, split, 'color_segmap', False, i)
			multi_step_augmentation(net, mask_prefix, colormask_prefix, rgb_prefix, sequence_prefix, split, 'color_segmap', True, i)
		
	

