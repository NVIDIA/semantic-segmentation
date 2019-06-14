#!/usr/bin/env python
import argparse
import os
import numpy as np
import shutil
import torch
import torch.backends.cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter

import cv2

from tqdm import tqdm

### masks warning : RuntimeError: Set changed size during iteration #481
# https://github.com/tqdm/tqdm/issues/481
tqdm.monitor_interval = 0
###
import datasets
import models
from utility import tools

from skimage.measure import compare_psnr, compare_ssim
import math

#### Import apex's distributed module.
try:
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
from apex import amp
###

### Imagenet style model collection (https://github.com/pytorch/examples/blob/master/imagenet/main.py)
model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__") and callable(models.__dict__[name]))
###

"""

Fitsum A. Reda, Guilin Liu, Kevin J. Shih, Robert Kirby, Jon Barker, David Tarjan, Andrew Tao, and Bryan Catanzaro. 
"SDC-Net: Video prediction using spatially-displaced convolution.", in ECCV 2018, pp. 718-733. 2018.

"""

parser = argparse.ArgumentParser(description='A PyTorch Implementation of SDCNet2D')

parser.add_argument('--model', metavar='MODEL', default='SDCNet2D',
                    choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: SDCNet2D)')
parser.add_argument('-s', '--save', '--save_root', default='./', type=str, metavar='SAVE_PATH',
                    help='Path of the output folder. (default: current path)')
parser.add_argument('--torch_home', default='./.torch', type=str,
                    metavar='TORCH_HOME',
                    help='Path to store native torch downloads of vgg, etc.')
parser.add_argument('-n', '--name', default='sdctrain', type=str, metavar='RUN_NAME',
                    help='Name of folder for output model')

parser.add_argument('--dataset', default='FrameLoader', type=str, metavar='TRAINING_DATALOADER_CLASS',
                    help='Specify dataset class for loading (Default: FrameLoader)')

parser.add_argument('--resume', default='', type=str, metavar='CHECKPOINT_PATH',
                    help='path to checkpoint (default: none)')

parser.add_argument('--distributed_backend', default='nccl', type=str, metavar='DISTRIBUTED_BACKEND',
                    help='backend used for communication between processes.')

# Resources
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loader workers (default: 10)')
parser.add_argument('-g', '--gpus', type=int, default=-1,
                    help='number of GPUs to use')

# Learning rate parameters.
parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_scheduler', default='MultiStepLR', type=str,
                    metavar='LR_Scheduler', help='Scheduler for learning' +
                                                 ' rate (only ExponentialLR, MultiStepLR, PolyLR supported.')
parser.add_argument('--lr_gamma', default=0.1, type=float,
                    help='learning rate will be multipled by this gamma')
parser.add_argument('--lr_step', default=200, type=int,
                    help='stepsize of changing the learning rate')
parser.add_argument('--lr_milestones', type=int, nargs='+',
                    default=[250, 450], help="Spatial dimension to " +
                                             "crop training samples for training")

# Gradient.
parser.add_argument('--clip_gradients', default=-1.0, type=float,
                    help='If positive, clip the gradients by this value.')

# Optimization hyper-parameters
parser.add_argument('-b', '--batch_size', default=4, type=int, metavar='BATCH_SIZE',
                    help='mini-batch per gpu size (default : 4)')
parser.add_argument('--wd', '--weight_decay', default=0.001, type=float, metavar='WEIGHT_DECAY',
                    help='weight_decay (default = 0.001)')
parser.add_argument('--seed', default=1234, type=int, metavar="SEED",
                    help='seed for initializing training. ')
parser.add_argument('--optimizer', default='Adam', type=str, metavar='OPTIMIZER',
                    help='Specify optimizer from torch.optim (Default: Adam)')

parser.add_argument('--print_freq', default=100, type=int, metavar="PRINT_FREQ",
                    help='frequency of printing training status (default: 100)')

parser.add_argument('--save_freq', type=int, default=20, metavar="SAVE_FREQ",
                    help='frequency of saving intermediate models (default: 20)')

parser.add_argument('--epochs', default=500, type=int, metavar="EPOCHES",
                    help='number of total epochs to run (default: 500)')

# Training sequence, supports a single sequence for now
parser.add_argument('--train_file',  metavar="TRAINING_FILE",
                    help='training file')
parser.add_argument('--sequence_length', default=2, type=int, metavar="SEQUENCE_LENGTH",
                    help='number of interpolated frames (default : 7)')
parser.add_argument('--crop_size', type=int, nargs='+', default=[448, 448], metavar="CROP_SIZE",
                    help="Spatial dimension to crop training samples for training (default : [448, 448])")
parser.add_argument('--train_n_batches', default=-1, type=int, metavar="TRAIN_N_BATCHES",
                    help="Limit the number of minibatch iterations per epoch. Used for debugging purposes. \
                    (default : -1")

# FlowNet2 or mixed-precision training experiments
parser.add_argument('--fp16', action='store_true', help="Enable mixed precision training \
                    using AMP: https://www.github.com/nvidia/apex (default : False) ")
parser.add_argument('--flownet2_checkpoint', required=True,
                    type=str, metavar='FLOWNET2_CHECKPOINT',
                    help='flownet-2 checkpoint: https://github.com/NVIDIA/flownet2-pytorch#converted-caffe-pre-trained-models')
parser.add_argument("--start_index", type=int, default=0, metavar="START_INDEX",
                    help="Index to start loading input data (default : 0)")

# Validation sequence, supports a single sequence for now
parser.add_argument('--val_file', metavar="VALIDATION_FILE",
                    help='validation file (default : None)')
parser.add_argument('--val_batch_size', type=int, default=1,
                    help="Batch size to use for validation.")
parser.add_argument('--val_n_batches', default=-1, type=int,
                    help="Limit the number of minibatch iterations per epoch. Used for debugging purposes.")
parser.add_argument('--video_fps', type=int, default=30,
                    help="Render predicted video with a specified frame rate")
parser.add_argument('--val_freq', default=500000, type=int,
                    help='frequency of running validation')
parser.add_argument('--stride', default=64, type=int,
                    help='The factor for which padded validation image sizes should be evenly divisible. (default: 64)')
parser.add_argument('--initial_eval', action='store_true', default=False,
                    help='Perform initial evaluation before training.')

# Misc: undersample large sequences (--step_size), compute flow after downscale (--flow_scale)
parser.add_argument("--sample_rate", type=int, default=1,
                    help="step size in looping through datasets")
parser.add_argument('--start_epoch', type=int, default=-1,
                    help="Set epoch number during resuming")
parser.add_argument('--skip_aug', action='store_true', help='Skips expensive geometric or photometric augmentations.')

parser.add_argument('--rgb_max', type=float, default=255, help="maximum expected value of rgb colors")

parser.add_argument('--local_rank', default=None, type=int,
                    help='Torch Distributed')

parser.add_argument('--write_images', action='store_true',
                    help='write to folder \'args.save/args.name\' prediction and ground-truth images.')

parser.add_argument('--write_video', action='store_true', help='save video to \'args.save/args.name.mp4\'.')

parser.add_argument('--eval', action='store_true', help='Run model in inference or evaluation mode.')


def parse_and_set_args(block):
    args = parser.parse_args()

    if args.resume != '':
        block.log('setting initial eval to true since checkpoint is provided')
        args.initial_eval = True

    torch.backends.cudnn.benchmark = True
    block.log('Enabling torch.backends.cudnn.benchmark')

    args.rank = int(os.getenv('RANK', 0))
    if args.local_rank:
        args.rank = args.local_rank
    args.world_size = int(os.getenv("WORLD_SIZE", 1))

    block.log("Creating save directory: {}".format(
        os.path.join(args.save, args.name)))
    args.save_root = os.path.join(args.save, args.name)
    os.makedirs(args.save_root, exist_ok=True)

    os.makedirs(args.torch_home, exist_ok=True)
    os.environ['TORCH_HOME'] = args.torch_home

    defaults, input_arguments = {}, {}
    for key in vars(args):
        defaults[key] = parser.get_default(key)

    for argument, value in sorted(vars(args).items()):
        if value != defaults[argument] and argument in vars(parser.parse_args()).keys():
            input_arguments['--' + str(argument)] = value
            block.log('{}: {}'.format(argument, value))

    args.network_class = tools.module_to_dict(models)[args.model]
    args.optimizer_class = tools.module_to_dict(torch.optim)[args.optimizer]
    args.dataset_class = tools.module_to_dict(datasets)[args.dataset]

    if args.eval:
        args.train_file = args.val_file

    return args

def initialilze_distributed(args):
    # Manually set the device ids.
    torch.cuda.set_device(args.rank % torch.cuda.device_count())
    # Call the init process
    if args.world_size > 1:
        init_method = 'env://'
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.world_size, rank=args.rank,
            init_method=init_method)

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_train_and_valid_data_loaders(block, args):

    # training dataloader
    tkwargs = {'batch_size': args.batch_size,
               'num_workers': args.workers,
               'pin_memory': True, 'drop_last': True}

    train_dataset = args.dataset_class(args,
                                       root=args.train_file, is_training=True)

    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler,
        shuffle=(train_sampler is None), **tkwargs)

    block.log('Number of Training Images: {}:{}'.format(
        len(train_loader.dataset), len(train_loader)))

    # validation dataloader
    vkwargs = {'batch_size': args.val_batch_size,
               'num_workers': args.workers,
               'pin_memory': False, 'drop_last': True}
    val_dataset = args.dataset_class(args, root=args.val_file)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, **vkwargs)

    block.log('Number of Validation Images: {}:{}'.format(
        len(val_loader.dataset), len(val_loader)))

    return train_loader, train_sampler, val_loader

def load_model(model, optimizer, block, args):
    # trained weights
    checkpoint = torch.load(args.resume, map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'], strict=False)
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if 'epoch' in checkpoint:
        args.start_epoch = max(0, checkpoint['epoch'])

    block.log("Successfully loaded checkpoint (at epoch {})".format(
        checkpoint['epoch']))

def build_and_initialize_model_and_optimizer(block, args):

    model = args.network_class(args)
    block.log('Number of parameters: {}'.format(
        sum([p.data.nelement()
             if p.requires_grad else 0 for p in model.parameters()])))

    block.log('Initializing CUDA')
    assert torch.cuda.is_available(), 'only GPUs support at the moment'
    model.cuda(torch.cuda.current_device())

    optimizer = args.optimizer_class(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    block.log("Attempting to Load checkpoint '{}'".format(args.resume))
    if args.resume and os.path.isfile(args.resume):
        load_model(model, optimizer, block, args)
    elif args.resume:
        block.log("No checkpoint found at '{}'".format(args.resume))
        exit(1)
    else:
        block.log("Random initialization, checkpoint not provided.")
        args.start_epoch = 0

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # Run multi-process when it is needed.
    if args.world_size > 1:
        model = DDP(model)

    return model, optimizer

def get_learning_rate_scheduler(optimizer, block, args):
    if args.lr_scheduler == 'ExponentialLR':
        block.log('using exponential decay learning rate with ' +
                  '{} decay rate'.format(args.lr_gamma))
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                              args.lr_gamma)
    elif args.lr_scheduler == 'MultiStepLR':
        block.log('using multi-step learning rate with {} gamma' +
                  ' and {} milestones.'.format(args.lr_gamma,
                                               args.lr_milestones))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)

    elif args.lr_scheduler == 'PolyLR':
        lambda_map = lambda epoc: math.pow(1 - epoc / args.epochs, 0.8)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_map)

    else:
        raise NameError('unknown {} learning rate scheduler'.format(
            args.lr_scheduler))
    return lr_scheduler

def forward_only(inputs_gpu, model):
    # Forward pass.
    losses, outputs, targets = model(inputs_gpu)

    # Loss.
    for k in losses:
        losses[k] = losses[k].mean(dim=0)
    loss = losses['tot']

    return loss, outputs, targets

def calc_linf_grad_norm(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    return max(p.grad.data.abs().max() for p in parameters)

def train_step(batch_cpu, model, optimizer, block, args, print_linf_grad=False):
    # Move data to GPU.
    inputs = {k: [b.cuda() for b in batch_cpu[k]] for k in batch_cpu}

    # Forward pass.
    loss, outputs, targets = forward_only(inputs, model)

    # Backward and SGP steps.
    optimizer.zero_grad()
    if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    # Calculate and print norm infinity of the gradients.
    if print_linf_grad:
        block.log('graldients Linf: {:0.3f}'.format(calc_linf_grad_norm(
            model.parameters())))

    # Clip gradients by value.
    if args.clip_gradients > 0:
        torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_gradients)

    optimizer.step()

    return loss, outputs, targets

def evaluate(model, val_loader, block, args, epoch):

    gHeight, gWidth = val_loader.dataset[0]['ishape']

    if args.rank == 0 and args.write_video:
        _pipe = tools.create_pipe(os.path.join(args.save_root, '__epoch_%03d.mp4' % (epoch)),
                                  gWidth, gHeight, frame_rate=args.video_fps)

    with torch.no_grad():

        loss_values = tools.AverageMeter('loss_values')
        avg_metrics = np.zeros((0, 3), dtype=float)

        # Set the model in evaluation mode.
        model.eval()
        num_batches = len(val_loader) if args.val_n_batches < 0 else args.val_n_batches

        for i, batch in enumerate(tqdm(val_loader, total=num_batches)):

            # Target images need to be only on CPU.
            target_images = batch['image'][-1]
            input_filenames = batch['input_files'][-1]

            inputs = {k: [b.cuda() for b in batch[k]] for k in batch if k == 'image'}

            loss, output_images, _ = forward_only(inputs, model)

            for b in range(args.val_batch_size):

                pred_image = ( output_images[b].data.cpu().numpy().transpose(1,2,0) ).astype(np.uint8)
                gt_image = ( target_images[b].data.cpu().numpy().transpose(1,2,0) ).astype(np.uint8)

                pred_image = pred_image[:gHeight, :gWidth, :]
                gt_image = gt_image[:gHeight, :gWidth, :]

                psnr = compare_psnr(pred_image, gt_image)
                ssim = compare_ssim(pred_image, gt_image, multichannel=True, gaussian_weights=True)
                err = pred_image.astype(np.float32) - gt_image.astype(np.float32)
                ie = np.mean(np.sqrt(np.sum(err * err, axis=2)))

                avg_metrics = np.vstack((avg_metrics, np.array([psnr, ssim, ie])))

                loss_values.update(loss.data.item(), output_images.size(0))

                if args.rank == 0 and args.write_video:
                    _pipe.stdin.write( pred_image[...,::-1].tobytes() )

                if args.rank == 0 and args.write_images:
                    image_filename = os.path.basename( input_filenames[b] )
                    cv2.imwrite( os.path.join( args.save_root, image_filename ), pred_image )
                    # imageio.imwrite( os.path.join( args.save_root, image_filename ), pred_image )

            if (i + 1) >= num_batches:
                break

        avg_metrics = np.nanmean(avg_metrics, axis=0)
        result2print = 'PSNR: {:.2f}, SSIM: {:.3f}, IE: {:.2f}'.format(
            avg_metrics[0], avg_metrics[1], avg_metrics[2])
        v_psnr, v_ssim, v_ie = avg_metrics[0], avg_metrics[1], avg_metrics[2]
        block.log(result2print)

    # close stream
    if args.rank == 0 and args.write_video:
        _pipe.stdin.close()
        _pipe.wait()

    # rename video with psnr
    if args.rank == 0 and args.write_video and args.eval:
        shutil.move(os.path.join(args.save_root, '__epoch_%03d.mp4' % (epoch)),
                    os.path.join(args.save_root, '_%s_%03d_psnr_%1.2f.mp4' % (args.name, epoch, avg_metrics[0])))
    elif args.rank == 0 and args.write_video:
        shutil.move(os.path.join(args.save_root, '__epoch_%03d.mp4' % (epoch)),
                    os.path.join(args.save_root, '__epoch_%03d_psnr_%1.2f.mp4' % (epoch, avg_metrics[0])))


    # Move back the model to train mode.
    model.train()

    torch.cuda.empty_cache()
    block.log('max memory allocated (GB): {:.3f}: '.format(
        torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)))

    return v_psnr, v_ssim, v_ie, loss_values.avg

def write_summary(global_index, learning_rate, t_loss, t_loss_avg,
                  v_loss, v_psnr, v_ssim, v_ie, args, epoch, end_of_epoch):
    # Write to tensorboard.
    if args.rank == 0:
        args.logger.add_scalar("train/lr", learning_rate, global_index)
        args.logger.add_scalar("train/trainloss", t_loss, global_index)
        args.logger.add_scalar("train/trainlossavg", t_loss_avg, global_index)
        args.logger.add_scalar("val/valloss", v_loss, global_index)
        args.logger.add_scalar("val/PSNR", v_psnr, global_index)
        args.logger.add_scalar("val/SSIM", v_ssim, global_index)
        args.logger.add_scalar("val/RMS", v_ie, global_index)

def train_epoch(epoch, args, model, optimizer, lr_scheduler,
                train_sampler, train_loader, val_loader,
                v_psnr, v_ssim, v_ie, v_loss, block):

    # Arverage loss calculator.
    loss_values = tools.AverageMeter('')

    # Advance Learning rate.
    lr_scheduler.step(epoch=epoch)

    # This will ensure the data is shuffled each epoch.
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    # Get number of batches in one epoch.
    num_batches = len(train_loader) if args.train_n_batches < 0 \
        else args.train_n_batches

    global_index = 0
    t_loss_epoch = 0.0
    for i, batch in enumerate(train_loader):

        # Set global index.
        global_index = epoch * num_batches + i

        # Move one step.
        loss, outputs, _ = train_step(
            batch, model, optimizer, block, args,
            ((global_index + 1) % args.print_freq == 0))

        # Update the loss accumulator.
        loss_values.update(loss.data.item(), outputs.size(0))

        end_of_epoch = False
        if ((i + 1) == num_batches):
            end_of_epoch = True

        # Summary writer.
        if (global_index + 1) % args.print_freq == 0 or end_of_epoch:
            # Reduce the loss.
            if args.world_size > 1:
                t_loss_gpu = torch.Tensor([loss_values.val]).cuda()
                train_loss_avg_epoch = torch.Tensor([loss_values.sum]).cuda()
                torch.distributed.all_reduce(t_loss_gpu)
                torch.distributed.all_reduce(train_loss_avg_epoch)
                t_loss = t_loss_gpu.item() / args.world_size
                t_loss_epoch = train_loss_avg_epoch / (args.world_size * loss_values.count)
            else:
                t_loss = loss_values.val
                t_loss_epoch = loss_values.avg

            # Write to tensorboard.
            write_summary(global_index, lr_scheduler.get_lr()[0], t_loss, t_loss_epoch,
                          v_loss, v_psnr, v_ssim, v_ie, args, epoch, end_of_epoch)

            # And reset the loss accumulator.
            loss_values.reset()

            # Print some output.
            dict2print = {'iter': global_index,
                          'epoch': str(epoch) + '/' + str(args.epochs),
                          'batch': str(i + 1) + '/' + str(num_batches)}
            str2print = ' '.join(key + " : " + str(dict2print[key])
                                 for key in dict2print)
            str2print += ' trainLoss:' + ' %1.3f' % (t_loss)
            str2print += ' trainLossAvg:' + ' %1.3f' % (t_loss_epoch)
            str2print += ' valLoss' + ' %1.3f' % (v_loss)
            str2print += ' valPSNR' + ' %1.3f' % (v_psnr)
            str2print += ' lr:' + ' %1.6f' % (lr_scheduler.get_lr()[0])
            block.log(str2print)

        # Break the training loop if we have reached the maximum number of batches.
        if (i + 1) >= num_batches:
            break

    return global_index

def save_model(model, optimizer, epoch, global_index, max_psnr, block, args):
    # Write on rank zero only
    if args.rank == 0:
        if args.world_size > 1:
            model_ = model.module
        else:
            model_ = model
        state_dict = model_.state_dict()
        tmp_keys = state_dict.copy()
        for k in state_dict:
            [tmp_keys.pop(k) if (k in tmp_keys and ikey in k)
             else None for ikey in model_.ignore_keys]
        state_dict = tmp_keys.copy()
        # save checkpoint
        model_optim_state = {'epoch': epoch,
                             'arch': args.model,
                             'state_dict': state_dict,
                             'optimizer': optimizer.state_dict(),
                             }
        model_name = os.path.join(
            args.save_root, '_ckpt_epoch_%03d_iter_%07d_psnr_%1.2f.pt.tar' % (
                epoch, global_index, max_psnr))
        torch.save(model_optim_state, model_name)
        block.log('saved model {}'.format(model_name))

        return model_name

def train(model, optimizer, lr_scheduler, train_loader,
          train_sampler, val_loader, block, args):

    # Set the model to train mode.
    model.train()

    # Keep track of maximum PSNR.
    max_psnr = -1

    # Perform an initial evaluation.
    if args.eval:
        block.log('Running Inference on Model.')
        _ = evaluate(model, val_loader, block, args, args.start_epoch + 1)
        return 0

    elif args.initial_eval:
        block.log('Initial evaluation.')
        v_psnr, v_ssim, v_ie, v_loss = evaluate(model, val_loader, block, args, args.start_epoch + 1)

    else:
        v_psnr, v_ssim, v_ie, v_loss = 20.0, 0.5, 15.0, 0.0

    for epoch in range(args.start_epoch, args.epochs):

        # Train for an epoch.
        global_index = train_epoch(epoch, args, model, optimizer, lr_scheduler,
                                   train_sampler, train_loader, val_loader,
                                   v_psnr, v_ssim, v_ie, v_loss, block)

        if (epoch + 1) % args.save_freq == 0:
            v_psnr, v_ssim, v_ie, v_loss = evaluate(model, val_loader, block, args, epoch + 1)
            if v_psnr > max_psnr:
                max_psnr = v_psnr
                save_model(model, optimizer, epoch + 1, global_index,
                           max_psnr, block, args)

        print("Completed epoch and checks", flush=True)

    return 0

def main():
    # Parse the args.
    with tools.TimerBlock("\nParsing Arguments") as block:
        args = parse_and_set_args(block)

    # Initialize torch.distributed.
    with tools.TimerBlock("Initializing Distributed"):
        initialilze_distributed(args)

    # Set all random seed for reproducability.
    with tools.TimerBlock("Setting Random Seed"):
        set_random_seed(args.seed)

    # Train and validation data loaders.
    with tools.TimerBlock("Building {} Dataset".format(args.dataset)) as block:
        train_loader, train_sampler, val_loader = get_train_and_valid_data_loaders(block, args)

    # Build the model and optimizer.
    with tools.TimerBlock("Building {} Model and {} Optimizer".format(
            args.model, args.optimizer_class.__name__)) as block:
        model, optimizer = build_and_initialize_model_and_optimizer(block, args)

    # Learning rate scheduler.
    with tools.TimerBlock("Building {} Learning Rate Scheduler".format(
            args.optimizer)) as block:
        lr_scheduler = get_learning_rate_scheduler(optimizer, block, args)

    # Set the tf writer on rank 0.
    with tools.TimerBlock("Creating Tensorboard Writers"):
        if args.rank == 0 and not args.eval:
            try:
                args.logger = SummaryWriter(logdir=args.save_root)
            except:
                args.logger = SummaryWriter(log_dir=args.save_root)

    # Start training
    with tools.TimerBlock("Training Model") as block:
        train(model, optimizer, lr_scheduler, train_loader,
              train_sampler, val_loader, block, args)

    return 0

if __name__ == '__main__':
    main()
