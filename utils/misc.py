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


Miscellanous Functions
"""
import cv2
import sys
import os
import torch
import numpy as np

import torchvision.transforms as standard_transforms
import torchvision.utils as vutils

from tabulate import tabulate
from PIL import Image

from config import cfg
from utils.results_page import ResultsPage
from runx.logx import logx


def fast_hist(pred, gtruth, num_classes):
    # mask indicates pixels we care about
    mask = (gtruth >= 0) & (gtruth < num_classes)

    # stretch ground truth labels by num_classes
    #   class 0  -> 0
    #   class 1  -> 19
    #   class 18 -> 342
    #
    # TP at 0 + 0, 1 + 1, 2 + 2 ...
    #
    # TP exist where value == num_classes*class_id + class_id
    # FP = row[class].sum() - TP
    # FN = col[class].sum() - TP
    hist = np.bincount(num_classes * gtruth[mask].astype(int) + pred[mask],
                       minlength=num_classes ** 2)
    hist = hist.reshape(num_classes, num_classes)
    return hist


def prep_experiment(args):
    """
    Make output directories, setup logging, Tensorboard, snapshot code.
    """
    args.ngpu = torch.cuda.device_count()
    args.best_record = {'mean_iu': -1, 'epoch': 0}


def calculate_iou(hist_data):
    acc = np.diag(hist_data).sum() / hist_data.sum()
    acc_cls = np.diag(hist_data) / hist_data.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    divisor = hist_data.sum(axis=1) + hist_data.sum(axis=0) - \
        np.diag(hist_data)
    iu = np.diag(hist_data) / divisor
    return iu, acc, acc_cls


def tensor_to_pil(img):
    inv_mean = [-mean / std for mean, std in zip(cfg.DATASET.MEAN,
                                                 cfg.DATASET.STD)]
    inv_std = [1 / std for std in cfg.DATASET.STD]
    inv_normalize = standard_transforms.Normalize(
        mean=inv_mean, std=inv_std
    )
    img = inv_normalize(img)
    img = img.cpu()
    img = standard_transforms.ToPILImage()(img).convert('RGB')
    return img


def eval_metrics(iou_acc, args, net, optim, val_loss, epoch, mf_score=None):
    """
    Modified IOU mechanism for on-the-fly IOU calculations ( prevents memory
    overflow for large dataset) Only applies to eval/eval.py
    """
    was_best = False

    iou_per_scale = {}
    iou_per_scale[1.0] = iou_acc
    if args.apex:
        iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
        torch.distributed.all_reduce(iou_acc_tensor,
                                     op=torch.distributed.ReduceOp.SUM)
        iou_per_scale[1.0] = iou_acc_tensor.cpu().numpy()
    scales = [1.0]

    # Only rank 0 should save models and calculate metrics
    if args.global_rank != 0:
        return

    hist = iou_per_scale[args.default_scale]
    iu, acc, acc_cls = calculate_iou(hist)
    iou_per_scale = {args.default_scale: iu}

    # calculate iou for other scales
    for scale in scales:
        if scale != args.default_scale:
            iou_per_scale[scale], _, _ = calculate_iou(iou_per_scale[scale])

    print_evaluate_results(hist, iu, epoch=epoch,
                           iou_per_scale=iou_per_scale,
                           log_multiscale_tb=args.log_msinf_to_tb)

    freq = hist.sum(axis=1) / hist.sum()
    mean_iu = np.nanmean(iu)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    metrics = {
        'loss': val_loss.avg,
        'mean_iu': mean_iu,
        'acc_cls': acc_cls,
        'acc': acc,
    }
    logx.metric('val', metrics, epoch)
    logx.msg('Mean: {:2.2f}'.format(mean_iu * 100))

    save_dict = {
        'epoch': epoch,
        'arch': args.arch,
        'num_classes': cfg.DATASET_INST.num_classes,
        'state_dict': net.state_dict(),
        'optimizer': optim.state_dict(),
        'mean_iu': mean_iu,
        'command': ' '.join(sys.argv[1:])
    }
    logx.save_model(save_dict, metric=mean_iu, epoch=epoch)
    torch.cuda.synchronize()

    if mean_iu > args.best_record['mean_iu']:
        was_best = True

        args.best_record['val_loss'] = val_loss.avg
        if mf_score is not None:
            args.best_record['mask_f1_score'] = mf_score.avg
        args.best_record['acc'] = acc
        args.best_record['acc_cls'] = acc_cls
        args.best_record['fwavacc'] = fwavacc
        args.best_record['mean_iu'] = mean_iu
        args.best_record['epoch'] = epoch

    logx.msg('-' * 107)
    if mf_score is None:
        fmt_str = ('{:5}: [epoch {}], [val loss {:0.5f}], [acc {:0.5f}], '
                   '[acc_cls {:.5f}], [mean_iu {:.5f}], [fwavacc {:0.5f}]')
        current_scores = fmt_str.format('this', epoch, val_loss.avg, acc,
                                        acc_cls, mean_iu, fwavacc)
        logx.msg(current_scores)
        best_scores = fmt_str.format(
            'best',
            args.best_record['epoch'], args.best_record['val_loss'],
            args.best_record['acc'], args.best_record['acc_cls'],
            args.best_record['mean_iu'], args.best_record['fwavacc'])
        logx.msg(best_scores)
    else:
        fmt_str = ('{:5}: [epoch {}], [val loss {:0.5f}], [mask f1 {:.5f} ] '
                   '[acc {:0.5f}], '
                   '[acc_cls {:.5f}], [mean_iu {:.5f}], [fwavacc {:0.5f}]')
        current_scores = fmt_str.format('this', epoch, val_loss.avg,
                                        mf_score.avg, acc,
                                        acc_cls, mean_iu, fwavacc)
        logx.msg(current_scores)
        best_scores = fmt_str.format(
            'best',
            args.best_record['epoch'], args.best_record['val_loss'],
            args.best_record['mask_f1_score'],
            args.best_record['acc'], args.best_record['acc_cls'],
            args.best_record['mean_iu'], args.best_record['fwavacc'])
        logx.msg(best_scores)
    logx.msg('-' * 107)

    return was_best


class ImageDumper():
    """
    Image dumping class
    
    You pass images/tensors from training pipeline into this object and it first
    converts them to images (doing transformations where necessary) and then
    writes the images out to disk.
    """
    def __init__(self, val_len, tensorboard=True, write_webpage=True,
                 webpage_fn='index.html', dump_all_images=False, dump_assets=False,
                 dump_err_prob=False, dump_num=10, dump_for_auto_labelling=False, 
                 dump_for_submission=False):
        """
        :val_len: num validation images
        :tensorboard: push summary to tensorboard
        :webpage: generate a summary html page
        :webpage_fn: name of webpage file
        :dump_all_images: dump all (validation) images, e.g. for video
        :dump_num: number of images to dump if not dumping all
        :dump_assets: dump attention maps
        """
        self.val_len = val_len
        self.tensorboard = tensorboard
        self.write_webpage = write_webpage
        self.webpage_fn = os.path.join(cfg.RESULT_DIR,
                                       'best_images', webpage_fn)
        self.dump_assets = dump_assets
        self.dump_for_auto_labelling = dump_for_auto_labelling
        self.dump_for_submission = dump_for_submission

        self.viz_frequency = max(1, val_len // dump_num)
        if dump_all_images:
            self.dump_frequency = 1
        else:
            self.dump_frequency = self.viz_frequency

        inv_mean = [-mean / std for mean, std in zip(cfg.DATASET.MEAN,
                                                   cfg.DATASET.STD)]
        inv_std = [1 / std for std in cfg.DATASET.STD]
        self.inv_normalize = standard_transforms.Normalize(
            mean=inv_mean, std=inv_std
        )

        if self.dump_for_submission:
            self.save_dir = os.path.join(cfg.RESULT_DIR, 'submit')
        elif self.dump_for_auto_labelling:
            self.save_dir = os.path.join(cfg.RESULT_DIR)
        else:
            self.save_dir = os.path.join(cfg.RESULT_DIR, 'best_images')

        os.makedirs(self.save_dir, exist_ok=True)

        self.imgs_to_tensorboard = []
        self.imgs_to_webpage = []

        if cfg.DATASET.NAME == 'cityscapes':
            # If all images of a dataset are identical, as in cityscapes,
            # there's no need to crop the images before tiling them into a
            # grid for displaying in tensorboard. Otherwise, need to center
            # crop the images
            self.visualize = standard_transforms.Compose([
                standard_transforms.Resize(384),
                standard_transforms.ToTensor()
            ])
        else:
            self.visualize = standard_transforms.Compose([
                standard_transforms.Resize(384),
                standard_transforms.CenterCrop((384, 384)),
                standard_transforms.ToTensor()
            ])

    def reset(self):
        self.imgs_to_tensorboard = []
        self.imgs_to_webpage = []

    def dump(self, dump_dict, val_idx):
        """
        dump a single batch of images

        :dump_dict: a dictionary containing elements to dump out
          'input_images': source image
          'gt_images': label
          'img_names': img_names
          'assets': dict with keys:
            'predictions': final prediction
            'pred_*': different scales of predictions
            'attn_*': different scales of attn
            'err_mask': err_mask
        """
        if self.dump_for_auto_labelling or  self.dump_for_submission:
            pass
        elif (val_idx % self.dump_frequency or cfg.GLOBAL_RANK != 0):
            return
        else:
            pass

        colorize_mask_fn = cfg.DATASET_INST.colorize_mask
        idx = 0  # only use first element of batch

        input_image = dump_dict['input_images'][idx]
        prob_image = dump_dict['assets']['prob_mask'][idx]
        gt_image = dump_dict['gt_images'][idx]
        prediction = dump_dict['assets']['predictions'][idx]
        del dump_dict['assets']['predictions']
        img_name = dump_dict['img_names'][idx]
        
        if self.dump_for_auto_labelling:
            # Dump Prob
            prob_fn = '{}_prob.png'.format(img_name)
            prob_fn = os.path.join(self.save_dir, prob_fn)
            cv2.imwrite(prob_fn, (prob_image.cpu().numpy()*255).astype(np.uint8))
            
        if self.dump_for_auto_labelling or self.dump_for_submission:
            # Dump Predictions
            prediction_cpu = np.array(prediction)
            label_out = np.zeros_like(prediction)
            submit_fn = '{}.png'.format(img_name)
            for label_id, train_id in   cfg.DATASET_INST.id_to_trainid.items():
                label_out[np.where(prediction_cpu == train_id)] = label_id
            cv2.imwrite(os.path.join(self.save_dir, submit_fn), label_out)
            return

        input_image = self.inv_normalize(input_image)
        input_image = input_image.cpu()
        input_image = standard_transforms.ToPILImage()(input_image)
        input_image = input_image.convert("RGB")
        input_image_fn = f'{img_name}_input.png'
        input_image.save(os.path.join(self.save_dir, input_image_fn))

        gt_fn = '{}_gt.png'.format(img_name)
        gt_pil = colorize_mask_fn(gt_image.cpu().numpy())
        gt_pil.save(os.path.join(self.save_dir, gt_fn))

        prediction_fn = '{}_prediction.png'.format(img_name)
        prediction_pil = colorize_mask_fn(prediction)
        prediction_pil.save(os.path.join(self.save_dir, prediction_fn))

        prediction_pil = prediction_pil.convert('RGB')
        composited = Image.blend(input_image, prediction_pil, 0.4)
        composited_fn = 'composited_{}.png'.format(img_name)
        composited_fn = os.path.join(self.save_dir, composited_fn)
        composited.save(composited_fn)

        # only visualize a limited number of images
        if val_idx % self.viz_frequency or cfg.GLOBAL_RANK != 0:
            return

        to_tensorboard = [
            self.visualize(input_image.convert('RGB')),
            self.visualize(gt_pil.convert('RGB')),
            self.visualize(prediction_pil.convert('RGB')),
        ]
        to_webpage = [
            (input_image_fn, 'input'),
            (gt_fn, 'gt'),
            (prediction_fn, 'prediction'),
        ]

        if self.dump_assets:
            assets = dump_dict['assets']
            for asset in assets:
                mask = assets[asset][idx]
                mask_fn = os.path.join(self.save_dir, f'{img_name}_{asset}.png')

                if 'pred_' in asset:
                    pred_pil = colorize_mask_fn(mask)
                    pred_pil.save(mask_fn)
                    continue

                if type(mask) == torch.Tensor:
                    mask = mask.squeeze().cpu().numpy()
                else:
                    mask = mask.squeeze()
                mask = (mask * 255)
                mask = mask.astype(np.uint8)
                mask_pil = Image.fromarray(mask)
                mask_pil = mask_pil.convert('RGB')
                mask_pil.save(mask_fn)
                to_tensorboard.append(self.visualize(mask_pil))
                to_webpage.append((mask_fn, asset))

        self.imgs_to_tensorboard.append(to_tensorboard)
        self.imgs_to_webpage.append(to_webpage)

    def write_summaries(self, was_best):
        """
        write out tensorboard
        write out html webpage summary

        only update tensorboard if was a best epoch
        always update webpage
        always save N images
        """
        if self.write_webpage:
            ip = ResultsPage('prediction examples', self.webpage_fn)
            for img_set in self.imgs_to_webpage:
                ip.add_table(img_set)
            ip.write_page()

        if self.tensorboard and was_best:

            if len(self.imgs_to_tensorboard):
                num_per_row = len(self.imgs_to_tensorboard[0])

                # flatten array:
                flattenned = []
                for a in self.imgs_to_tensorboard:
                    for b in a:
                        flattenned.append(b)

                imgs_to_tensorboard = torch.stack(flattenned, 0)
                imgs_to_tensorboard = vutils.make_grid(
                    imgs_to_tensorboard, nrow=num_per_row, padding=5)
                logx.add_image('imgs', imgs_to_tensorboard, cfg.EPOCH)


def print_evaluate_results(hist, iu, epoch=0, iou_per_scale=None,
                           log_multiscale_tb=False):
    """
    If single scale:
       just print results for default scale
    else
       print all scale results

    Inputs:
    hist = histogram for default scale
    iu = IOU for default scale
    iou_per_scale = iou for all scales
    """
    id2cat = cfg.DATASET_INST.trainid_to_name
    # id2cat = {i: i for i in range(cfg.DATASET.NUM_CLASSES)}

    iu_FP = hist.sum(axis=1) - np.diag(hist)
    iu_FN = hist.sum(axis=0) - np.diag(hist)
    iu_TP = np.diag(hist)

    logx.msg('IoU:')

    header = ['Id', 'label']
    header.extend(['iU_{}'.format(scale) for scale in iou_per_scale])
    header.extend(['TP', 'FP', 'FN', 'Precision', 'Recall'])

    tabulate_data = []

    for class_id in range(len(iu)):
        class_data = []
        class_data.append(class_id)
        class_name = "{}".format(id2cat[class_id]) if class_id in id2cat else ''
        class_data.append(class_name)
        for scale in iou_per_scale:
            class_data.append(iou_per_scale[scale][class_id] * 100)

        total_pixels = hist.sum()
        class_data.append(100 * iu_TP[class_id] / total_pixels)
        class_data.append(iu_FP[class_id] / iu_TP[class_id])
        class_data.append(iu_FN[class_id] / iu_TP[class_id])
        class_data.append(iu_TP[class_id] / (iu_TP[class_id] + iu_FP[class_id]))
        class_data.append(iu_TP[class_id] / (iu_TP[class_id] + iu_FN[class_id]))
        tabulate_data.append(class_data)

        if log_multiscale_tb:
            logx.add_scalar("xscale_%0.1f/%s" % (0.5, str(id2cat[class_id])),
                            float(iou_per_scale[0.5][class_id] * 100), epoch)
            logx.add_scalar("xscale_%0.1f/%s" % (1.0, str(id2cat[class_id])),
                            float(iou_per_scale[1.0][class_id] * 100), epoch)
            logx.add_scalar("xscale_%0.1f/%s" % (2.0, str(id2cat[class_id])),
                            float(iou_per_scale[2.0][class_id] * 100), epoch)

    print_str = str(tabulate((tabulate_data), headers=header, floatfmt='1.2f'))
    logx.msg(print_str)


def metrics_per_image(hist):
    """
    Calculate tp, fp, fn for one image
    """
    FP = hist.sum(axis=1) - np.diag(hist)
    FN = hist.sum(axis=0) - np.diag(hist)
    return FP, FN


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fmt_scale(prefix, scale):
    """
    format scale name

    :prefix: a string that is the beginning of the field name
    :scale: a scale value (0.25, 0.5, 1.0, 2.0)
    """

    scale_str = str(float(scale))
    scale_str.replace('.', '')
    return f'{prefix}_{scale_str}x'
