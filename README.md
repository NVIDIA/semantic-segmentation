# Improving Semantic Segmentation via Video Prediction and Label Relaxation
### [Project](https://nv-adlr.github.io/publication/2018-Segmentation) | [Paper](https://arxiv.org/pdf/1812.01593.pdf) | [YouTube](https://www.youtube.com/watch?v=aEbXjGZDZSQ)  | [Cityscapes Score](https://www.cityscapes-dataset.com/anonymous-results/?id=555fc2b66c6e00b953c72b98b100e396c37274e0788e871a85f1b7b4f4fa130e) | [Kitti Score](http://www.cvlibs.net/datasets/kitti/eval_semseg_detail.php?benchmark=semantics2015&result=83cac7efbd41b1f2fc095f9bc1168bc548b48885) <br>
PyTorch implementation of our CVPR2019 paper (oral) on achieving state-of-the-art semantic segmentation results using Deeplabv3-Plus like architecture with a WideResNet38 trunk. We present a video prediction-based methodology to scale up training sets by synthesizing new training samples and propose a novel label relaxation technique to make training objectives robust to label noise. <br>

[Improving Semantic Segmentation via Video Propagation and Label Relaxation](https://nv-adlr.github.io/publication/2018-Segmentation) <br />
Yi Zhu<sup>1*</sup>, Karan Sapra<sup>2*</sup>, [Fitsum A. Reda](https://scholar.google.com/citations?user=quZ_qLYAAAAJ&hl=en)<sup>2</sup>, Kevin J. Shih<sup>2</sup>, Shawn Newsam<sup>1</sup>, Andrew Tao<sup>2</sup>, [Bryan Catanzaro](http://catanzaro.name/)<sup>2</sup>  
<sup>1</sup>UC Merced, <sup>2</sup>NVIDIA Corporation  <br />
In CVPR 2019 (* equal contributions).

[SDCNet: Video Prediction using Spatially Displaced Convolution](https://nv-adlr.github.io/publication/2018-SDCNet)  
[Fitsum A. Reda](https://scholar.google.com/citations?user=quZ_qLYAAAAJ&hl=en), Guilin Liu, Kevin J. Shih, Robert Kirby, Jon Barker, David Tarjan, Andrew Tao, [Bryan Catanzaro](http://catanzaro.name/)<br />
NVIDIA Corporation <br />
In ECCV 2018. 

![alt text](images/method.png)

## Installation 

    # Get Semantic Segmentation source code
    git clone --recursive https://github.com/NVIDIA/semantic-segmentation.git
    cd semantic-segmentation

    # Build Docker Image
    docker build -t nvidia-segmgentation -f Dockerfile .

If you prefer not to use docker, you can manually install the following requirements: 

* An NVIDIA GPU and CUDA 9.0 or higher. Some operations only have gpu implementation.
* PyTorch (>= 0.5.1)
* Python 3
* numpy
* sklearn
* h5py
* scikit-image
* pillow
* piexif
* cffi
* tqdm
* dominate
* tensorboardX
* opencv-python
* nose
* ninja


Multiple GPU training and mixed precision training are supported, and the code provides examples for training and inference. For more help, type <br/>
      
    python3 train.py --help



## Network architectures

Our repo now supports DeepLabV3+ architecture with different backbones, including `WideResNet38`, `SEResNeXt(50, 101)` and `ResNet(50,101)`. 

  
## Pre-trained models
We've included pre-trained models. Download checkpoints to a folder `pretrained_models`. 

* [pretrained_models/cityscapes_best.pth](https://drive.google.com/file/d/1P4kPaMY-SmQ3yPJQTJ7xMGAB_Su-1zTl/view?usp=sharing)[1071MB, WideResNet38 backbone]
* [pretrained_models/camvid_best.pth](https://drive.google.com/file/d/1OzUCbFdXulB2P80Qxm7C3iNTeTP0Mvb_/view?usp=sharing)[1071MB, WideResNet38 backbone]
* [pretrained_models/kitti_best.pth](https://drive.google.com/file/d/1OrTcqH_I3PHFiMlTTZJgBy8l_pladwtg/view?usp=sharing)[1071MB, WideResNet38 backbone]
* [pretrained_models/sdc_cityscapes_vrec.pth.tar](https://drive.google.com/file/d/1OxnJo2tFEQs3vuY01ibPFjn3cRCo2yWt/view?usp=sharing)[38MB]
* [pretrained_models/FlowNet2_checkpoint.pth.tar](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing)[620MB]

ImageNet Weights
* [pretrained_models/wider_resnet38.pth.tar](https://drive.google.com/file/d/1OfKQPQXbXGbWAQJj2R82x6qyz6f-1U6t/view?usp=sharing)[833MB]

Other Weights
* [pretrained_models/cityscapes_cv0_seresnext50_nosdcaug.pth](https://drive.google.com/file/d/1aGdA1WAKKkU2y-87wSOE1prwrIzs_L-h/view?usp=sharing)[324MB]
* [pretrained_models/cityscapes_cv0_wideresnet38_nosdcaug.pth](https://drive.google.com/file/d/1CKB7gpcPLgDLA7LuFJc46rYcNzF3aWzH/view?usp=sharing)[1.1GB]

## Data Loaders

Dataloaders for Cityscapes, Mapillary, Camvid and Kitti are available in [datasets](./datasets). Details of preparing each dataset can be found at [PREPARE_DATASETS.md](https://github.com/NVIDIA/semantic-segmentation/blob/master/PREPARE_DATASETS.md) <br />


## Semantic segmentation demo for a single image

If you want to try our trained model on any driving scene images, simply use

```
CUDA_VISIBLE_DEVICES=0 python demo.py --demo-image YOUR_IMG --snapshot ./pretrained_models/cityscapes_best.pth --save-dir YOUR_SAVE_DIR
```
This snapshot is trained on Cityscapes dataset, with `DeepLabV3+` architecture and `WideResNet38` backbone. The predicted segmentation masks will be saved to `YOUR_SAVE_DIR`. Check it out. 

## Semantic segmentation demo for a folder of images

If you want to try our trained model on a folder of driving scene images, simply use

```
CUDA_VISIBLE_DEVICES=0 python demo_folder.py --demo-folder YOUR_FOLDER --snapshot ./pretrained_models/cityscapes_best.pth --save-dir YOUR_SAVE_DIR
```
This snapshot is trained on Cityscapes dataset, with `DeepLabV3+` architecture and `WideResNet38` backbone. The predicted segmentation masks will be saved to `YOUR_SAVE_DIR`. Check it out. 
 
## A quick start with light SEResNeXt50 backbone

Note that, in this section, we use the standard train/val split in Cityscapes to train our model, which is `cv 0`. 

If you have less than 8 GPUs in your machine, please change `--nproc_per_node=8` to the number of GPUs you have in all the .sh files under folder `scripts`.

### Pre-Training on Mapillary 
First, you can pre-train a DeepLabV3+ model with `SEResNeXt50` trunk on Mapillary dataset. Set `__C.DATASET.MAPILLARY_DIR` in `config.py` to where you store the Mapillary data. 

```
./scripts/train_mapillary_SEResNeXt50.sh
```

When you first run training on a new dataset with flag `--class_uniform_pct` on, it will take some time to preprocess the dataset. Depending on your machine, the preprocessing can take half an hour or more. Once it finishes, you will have a json file in your root folder, e.g., `mapillary_tile1024.json`. You can read more details about `class uniform sampling` in our paper, the idea is to make sure that all classes are approximately uniformly chosen during training.

### Fine-tuning on Cityscapes 
Once you have the Mapillary pre-trained model (training mIoU should be 50+), you can start fine-tuning the model on Cityscapes dataset. Set `__C.DATASET.CITYSCAPES_DIR` in `config.py` to where you store the Cityscapes data. Your training mIoU in the end should be 80+. 
```
./scripts/train_cityscapes_SEResNeXt50.sh
```

### Inference

Our inference code supports two ways of evaluation: pooling and sliding based eval. The pooling based eval is faster than sliding based eval but provides slightly lower numbers. We use `sliding` as default. 
 ```
 ./scripts/eval_cityscapes_SEResNeXt50.sh <weight_file_location> <result_save_location>
 ```

In the `result_save_location` you set, you will find several folders: `rgb`, `pred`, `compose` and `diff`. `rgb` contains the color-encode predicted segmentation masks. `pred` contains what you need to submit to the evaluation server. `compose` contains the overlapped images of original video frame and the color-encode predicted segmentation masks. `diff` contains the difference between our prediction and the ground truth. 

Right now, our inference code only supports Cityscapes dataset.  


## Reproducing our results with heavy WideResNet38 backbone

Note that, in this section, we use an alternative train/val split in Cityscapes to train our model, which is `cv 2`. You can find the difference between `cv 0` and `cv 2` in the supplementary material section in our arXiv paper. 

### Pre-Training on Mapillary 
```
./scripts/train_mapillary_WideResNet38.sh
```

### Fine-tuning on Cityscapes 
```
./scripts/train_cityscapes_WideResNet38.sh
```

### Inference
```
./scripts/eval_cityscapes_WideResNet38.sh <weight_file_location> <result_save_location>
```

For submitting to Cityscapes benchmark, we change it to multi-scale setting. 
 ```
 ./scripts/submit_cityscapes_WideResNet38.sh <weight_file_location> <result_save_location>
 ```

Now you can zip the `pred` folder and upload to Cityscapes leaderboard. For the test submission, there is nothing in the `diff` folder because we don't have ground truth. 

At this point, you can already achieve top performance on Cityscapes benchmark (83+ mIoU). In order to further boost the segmentation performance, we can use the augmented dataset to help model's generalization capibility. 

### Label Propagation using Video Prediction 
First, you need to donwload the Cityscapes sequence dataset. Note that the sequence dataset is very large (a 325GB .zip file). Then we can use video prediction model to propagate GT segmentation masks to adjacent video frames, so that we can have more annotated image-label pairs during training. 

```
cd ./sdcnet

bash flownet2_pytorch/install.sh

./_aug.sh
```

By default, we predict five past frames and five future frames, which effectively enlarge the dataset 10 times. If you prefer to propagate less or more time steps, you can change the `--propagate` accordingly. Enjoy the augmented dataset. 


## Results on Cityscapes

![alt text](images/vis.png)

## Training IOU using fp16

Training results for WideResNet38 and SEResNeXt50 trained in fp16 on DGX-1 (8-GPU V100). fp16 can significantly speed up experiments without losing much accuracy. 

<table class="tg">
  <tr>
    <th class="tg-t2cw">Model Name</th>
    <th class="tg-t2cw">Mean IOU</th>
    <th class="tg-t2cw">Training Time</th>
  </tr>
  <tr>
    <td class="tg-rg0h">DeepWV3Plus(no sdc-aug)</td>
    <td class="tg-rg0h">81.4</td>
    <td class="tg-rg0h">~14 hrs</td>
  </tr>
  <tr>
    <td class="tg-rg0h">DeepSRNX50V3PlusD_m1(no sdc-aug)</td>
    <td class="tg-rg0h">80.0</td>
    <td class="tg-rg0h">~9 hrs</td>
  </tr>
</table>


## Reference 

If you find this implementation useful in your work, please acknowledge it appropriately and cite the paper or code accordingly:

```
@inproceedings{semantic_cvpr19,
  author       = {Yi Zhu*, Karan Sapra*, Fitsum A. Reda, Kevin J. Shih, Shawn Newsam, Andrew Tao, Bryan Catanzaro},
  title        = {Improving Semantic Segmentation via Video Propagation and Label Relaxation},
  booktitle    = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month        = {June},
  year         = {2019},
  url          = {https://nv-adlr.github.io/publication/2018-Segmentation}
}
* indicates equal contribution

@inproceedings{reda2018sdc,
  title={SDC-Net: Video prediction using spatially-displaced convolution},
  author={Reda, Fitsum A and Liu, Guilin and Shih, Kevin J and Kirby, Robert and Barker, Jon and Tarjan, David and Tao, Andrew and Catanzaro, Bryan},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={718--733},
  year={2018}
}
```
We encourage people to contribute to our code base and provide suggestions, point any issues, or solution using merge request, and we hope this repo is useful.  

## Acknowledgments

Parts of the code were heavily derived from [pytorch-semantic-segmentation](https://github.com/ZijunDeng/pytorch-semantic-segmentation), [inplace-abn](https://github.com/mapillary/inplace_abn), [Pytorch](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py), [ClementPinard/FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch), [NVIDIA/flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch) and [Cadene](https://github.com/Cadene/pretrained-models.pytorch).
 
Our initial models used SyncBN from [Synchronized Batch Norm](https://github.com/zhanghang1989/PyTorch-Encoding) but since then have been ported to [Apex SyncBN](https://github.com/NVIDIA/apex) developed by Jie Jiang.

We would also like to thank Ming-Yu Liu and Peter Kontschieder.
 
## Coding style
* 4 spaces for indentation rather than tabs
* 100 character line length
* PEP8 formatting
