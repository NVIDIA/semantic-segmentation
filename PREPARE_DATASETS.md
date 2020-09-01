## Mapillary Vistas Dataset

First of all, please request the research edition dataset from [here](https://www.mapillary.com/dataset/vistas/). The downloaded file is named as `mapillary-vistas-dataset_public_v1.1.zip`.

Then simply unzip the file by
```shell
unzip mapillary-vistas-dataset_public_v1.1.zip
```

The folder structure will look like:
```
Mapillary
├── config.json
├── demo.py
├── Mapillary Vistas Research Edition License.pdf
├── README
├── requirements.txt
├── training
│   ├── images
│   ├── instances
│   ├── labels
│   ├── panoptic
├── validation
│   ├── images
│   ├── instances
│   ├── labels
│   ├── panoptic
├── testing
│   ├── images
│   ├── instances
│   ├── labels
│   ├── panoptic
```
Note that, the `instances`, `labels` and `panoptic` folders inside `testing` are empty. 

Suppose you store your dataset at `~/username/data/Mapillary`, please update the dataset path in `config.py`,
```
__C.DATASET.MAPILLARY_DIR = '~/username/data/Mapillary'
```

## Cityscapes Dataset

### Download Dataset
First of all, please request the dataset from [here](https://www.cityscapes-dataset.com/). You need multiple files.
```
- leftImg8bit_trainvaltest.zip
- gtFine_trainvaltest.zip
- leftImg8bit_trainextra.zip
- gtCoarse.zip
- refinement_final_v0.zip [link] (https://drive.google.com/file/d/1DtPo-WP-hjaOwsbj6ZxTtOo_7R_4TKRG/)   # This file is only needed for autolabelled training for recreating SOTA
```

If you prefer to use command lines (e.g., `wget`) to download the dataset,
```
# First step, obtain your login credentials.
Please register an account at https://www.cityscapes-dataset.com/login/.

# Second step, log into cityscapes system, suppose you already have a USERNAME and a PASSWORD.
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=USERNAME&password=PASSWORD&submit=Login' https://www.cityscapes-dataset.com/login/

# Third step, download the zip files you need.
wget -c -t 0 --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3

# The corresponding packageID is listed below,
1  -> gtFine_trainvaltest.zip (241MB)            md5sum: 4237c19de34c8a376e9ba46b495d6f66
2  -> gtCoarse.zip (1.3GB)                       md5sum: 1c7b95c84b1d36cc59a9194d8e5b989f
3  -> leftImg8bit_trainvaltest.zip (11GB)        md5sum: 0a6e97e94b616a514066c9e2adb0c97f
4  -> leftImg8bit_trainextra.zip (44GB)          md5sum: 9167a331a158ce3e8989e166c95d56d4
5  -> refinement_final_v0.zip (5GB)              md5sum: 82aa6698ef7358457894c7cc924534fb
```

### Prepare Folder Structure

Now unzip those files, the desired folder structure will look like,
```
Cityscapes
├── leftImg8bit_trainvaltest
│   ├── leftImg8bit
│   │   ├── train
│   │   │   ├── aachen
│   │   │   │   ├── aachen_000000_000019_leftImg8bit.png
│   │   │   │   ├── aachen_000001_000019_leftImg8bit.png
│   │   │   │   ├── ...
│   │   │   ├── bochum
│   │   │   ├── ...
│   │   ├── val
│   │   ├── test
├── gtFine_trainvaltest
│   ├── gtFine
│   │   ├── train
│   │   │   ├── aachen
│   │   │   │   ├── aachen_000000_000019_gtFine_color.png
│   │   │   │   ├── aachen_000000_000019_gtFine_instanceIds.png
│   │   │   │   ├── aachen_000000_000019_gtFine_labelIds.png
│   │   │   │   ├── aachen_000000_000019_gtFine_polygons.json
│   │   │   │   ├── ...
│   │   │   ├── bochum
│   │   │   ├── ...
│   │   ├── val
│   │   ├── test
├── leftImg8bit_trainextra
│   ├── leftImg8bit
│   │   ├── train_extra
│   │   │   ├── augsburg
│   │   │   ├── bad-honnef
│   │   │   ├── ...
├── gtCoarse
│   ├── gtCoarse
│   │   ├── train
│   │   ├── train_extra
│   │   ├── val
├── autolabelled
│   ├── train_extra
│   │   ├── augsburg
│   │   ├── bad-honnef
│   │   ├── ...
```



