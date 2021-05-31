# Black-Box Dissector: Towards Erasing-based Hard-Label Model Stealing Attack

Our codes are based on the source code of Knockoff-Nets kindly provided by the authors.

## Installation

### Environment

Can be set up as:

```bash
$ pip install -r requirements.txt
```

Prepare the folder as follows:

```
├── attackset
├── data
└── models
  ├── attack
  └── victim
```
  
## Datasets

You will need five datasets to perform all experiment in the paper, all extracted into the 'data/' directory.

* Victim datasets:
  * CIFAR10 (In `data/cifar10/cifar-10-batches-py/`)
  * MNIST (In `data/mnist/MNIST/`)
  * Caltech256 (Images in `data/256_ObjectCategories/<classname>/*.jpg`)
  * CUB-200-2011 (Images in `data/CUB_200_2011/images/<classname>/*.jpg`)
* Attack datasets:
  * ImageNet ILSVRC 2012 (Images in `data/ILSVRC2012/training_imgs/<classname>/*.jpg`)

The structure of image files looks like:

```
data
└── ILSVRC2012
  └── training_imgs
    └── n01440764
      ├── n01440764_18.jpeg
      ├── n01440764_36.jpeg
      └── ...
    └── ...
└── 256_ObjectCategories
  └── 001.ak47
    ├── 001_0001.jpg
    ├── 001_0002.jpg
    └── ...
  └── ...
└── cifar10
  └──cifar-10-batches-py
    ├── batches.meta
    ├── data_batch_1
    └── ...
└── CUBS_200_2011
  └── images
    └── 001.Black_footed_Albatross
      ├── Black_Footed_Albatross_0001_796111.jpg
      ├── Black_Footed_Albatross_0002_55.jpg
      └── ...
    └── ...
└── mnist
  └── MNIST
    ├── processed
    └── raw
```

## Victim Model

### Train Victim Models

```bash
# Format:
$ python blackbox_model/victim/train.py DS_NAME ARCH -d DEV_ID \
         -o models/victim/VIC_DIR -e EPOCHS --pretrained
# where DS_NAME = {CUBS200, Caltech256, CIFAR10, MNIST}, ARCH = resnet34

# More details:
$ python blackbox_model/victim/train.py --help

# Example (CUBS-200-2011):
$ python blackbox_model/victim/train.py CUBS200 resnet34 -d 1 \
        -o models/victim/cubs200-resnet34 -e 100 --log-interval 25 \
        --pretrained imagenet
```

## Attack Datasets

In order to simulate the online APIs, we first obtain the output value of the victim model for all images and save it.
Note: This does not mean the attacker has the predicted value of all pictures, the attacker can only access a certain number according to the budget.

### Build Attack Datasets

```bash
# Format:
$ python build_transferset.py --victim_dataset VD_NAME \
         --victim_model_dir models/victim/VIC_DIR --out_dir attackset/O_DIR/ \
         --queryset DS_NAME --batch_size BS -d DEV_ID
# where VD_NAME = {CUBS200, Caltech256, CIFAR10, MNIST}, DS_NAME = ImageNet1k

# More details:
$ python build_transferset.py --help

# Example (CUBS-200-2011):
$ python build_transferset.py --victim_dataset CUBS200 \
         --victim_model_dir models/victim/cubs200-resnet34 --out_dir attackset/cubs200-resnet34/ \
         --queryset ImageNet1k --batch_size 128 -d 1
```

## BD Attack

### Run BD Attack

```bash
# Format:
$ CUDA_VISIBLE_DEVICES = DEV_ID python attack.py --test_dataset DS_NAME --dataset_path attackset/O_DIR/ \
                                --batch_size BS --modelname ARCH --cuda BOOL --train_epochs EPOCHS \
                                --initial_budget BUDGET --blackbox_dir models/victim/VIC_DIR --save_dir models/attack/ATT_DIR \
                                --tmp_dir data/TMP_DIR --sampling_strategy STRATEGY --lr LR --weight_decay WD --momentum MOMENTUM \
                                --sh SH --num_workers NW --pretrained PRETRAINED --step_size SS
# where DS_NAME = {CUBS200, Caltech256, CIFAR10, MNIST}, modelname = {resnet34, resnet18, resnet50, vgg16, densenet}, STRATEGY = {radom, kcenter}, pretrained = {None, imagenet}

# More details:
$ python attack.py --help

# Example (CUBS-200-2011):
$ CUDA_VISIBLE_DEVICES = 1 python attack.py --test_dataset CUBS200 --dataset_path attackset/cubs200-resnet34/attackset.pickle \
                           --batch_size 128 --modelname resnet34 --cuda True --train_epochs 200 \
                           --initial_budget 100 --blackbox_dir models/victim/cubs200-resnet34 --save_dir models/attack/cubs200-resnet34/ \
                           --tmp_dir data/tmp_CUBS200/ --sampling_strategy random --lr 0.02 --weight_decay 5e-4 --momentum 0.9 \
                           --sh 0.1 --num_workers 16 --pretrained None --step_size 60
```
