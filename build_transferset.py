import os.path as osp
import os
import pickle
import json
from datetime import datetime
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import blackbox_model.models.zoo as zoo

from blackbox_model import datasets
# import blackbox_model.datasets as datasets
import blackbox_model.utils.transforms as transform_utils
import blackbox_model.utils.model as model_utils
import blackbox_model.utils.utils as blackbox_utils
from blackbox_model.victim.blackbox import Blackbox
import blackbox_model.config as cfg
from my_transform import PatchTransform, SaturationTrasform, EdgeTransform, BrightnessTransform

import glob
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

cifar_transform = transforms.Compose([
                  transforms.Resize(256),
                  transforms.CenterCrop(224),
                  transforms.Resize(32),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                       std=(0.2023, 0.1994, 0.2010))
                  ])

mnist_transform = transforms.Compose([
                  transforms.Resize(256),
                  transforms.CenterCrop(224),
                  transforms.Resize(28),
                  transforms.Grayscale(num_output_channels=1),
                  transforms.ToTensor(),
                  transforms.Normalize((0.1307,), (0.3081,))
                  ])

svhn_transform = transforms.Compose([
                  transforms.Resize(256),
                  transforms.CenterCrop(224),
                  transforms.Resize(32),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                       std=(0.2023, 0.1994, 0.2010))
                  ])

class Adversary(object):
    def __init__(self, blackbox, queryset, label_only=False, batch_size=64):
        self.blackbox = blackbox
        self.queryset = queryset
        self.label_only = label_only

        self.n_queryset = len(self.queryset)
        self.batch_size = batch_size
        self.idx_set = set()

        self.transferset = []  # List of tuples [(img_path, output_probs)]

        self._restart()

    def _restart(self):
        # np.random.seed(cfg.DEFAULT_SEED)
        # torch.manual_seed(cfg.DEFAULT_SEED)
        # torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        self.idx_set = set(range(len(self.queryset)))
        self.transferset = []

    def get_transferset(self):
        start_B = 0
        end_B = self.n_queryset - 1
        budget = self.n_queryset
        with tqdm(total=budget) as pbar:
            for t, B in enumerate(range(start_B, end_B, self.batch_size)):
                idxs = list(self.idx_set)[B:B+self.batch_size]
                x_t = torch.stack([self.queryset[i][0] for i in idxs]).to(self.blackbox.device)
                # x_t = torch.stack([self.queryset[i][0] for i in idxs]).cuda()
                y_t = self.blackbox(x_t).cpu()

                if hasattr(self.queryset, 'samples'):
                    # Any DatasetFolder (or subclass) has this attribute
                    # Saving image paths are space-efficient
                    img_t = [self.queryset.samples[i][0] for i in idxs]  # Image paths
                else:
                    # Otherwise, store the image itself
                    # But, we need to store the non-transformed version
                    img_t = [self.queryset.data[i] for i in idxs]
                    if isinstance(self.queryset.data[0], torch.Tensor):
                        img_t = [x.numpy() for x in img_t]

                for i in range(x_t.size(0)):
                    img_t_i = img_t[i].squeeze() if isinstance(img_t[i], np.ndarray) else img_t[i]
                    if self.label_only:
                        # print('=> Using argmax labels (instead of posterior probabilities)')
                        y_i = y_t[i].cpu().squeeze()
                        argmax_k = y_i.argmax()
                        y_i_1hot = torch.zeros_like(y_i)
                        y_i_1hot[argmax_k] = 1.
                        self.transferset.append((img_t_i, y_i_1hot))
                    else:
                        self.transferset.append((img_t_i, y_t[i].cpu().squeeze()))

                pbar.update(x_t.size(0))

        return self.transferset


def main():
    parser = argparse.ArgumentParser(description='Construct transfer set')
    parser.add_argument('--victim_dataset', type=str,
                        help='The name of victim model training dataset', default='CIFAR10')
    parser.add_argument('--victim_model_dir', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"', default='./models/victim/cifar10-resnet34')
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store transfer set', required=True)
    # parser.add_argument('--budget', metavar='N', type=int, help='Size of transfer set to construct', required=True)
    parser.add_argument('--queryset', type=str, help='Adversary\'s dataset (P_A(X))', default='ImageNet1k')
    parser.add_argument('--batch_size', type=int, help='Batch size of queries', default=256)
    parser.add_argument('--label_only', type=bool, help='Whether to return on labels', default=False)
    # ----------- Other params
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    args = parser.parse_args()
    params = vars(args)
    out_path = params['out_dir']
    blackbox_utils.create_dir(out_path)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    blackbox_dir = params['victim_model_dir']
    blackbox = Blackbox.from_modeldir(blackbox_dir, device)
    # blackbox = zoo.get_net(modelname='resnet34', modeltype='cifar', pretrained=None, num_classes=10)
    # blackbox.load_state_dict(torch.load('./checkpoint/pgd_adversarial_training.pth')['net'])
    # blackbox = blackbox.to(device)
    queryset_name = params['queryset']
    valid_datasets = datasets.__dict__.keys()
    if queryset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    if params['victim_dataset'] == 'CIFAR10':
        transform = cifar_transform
    elif params['victim_dataset'] == 'SVHN':
        transform = svhn_transform
    elif params['victim_dataset'] == 'MNIST':
        transform = mnist_transform
    elif params['victim_dataset'] == 'SVHN':
        transform = svhn_transform
    else:
        transform = datasets.modelfamily_to_transforms['imagenet']['test']
    queryset = datasets.__dict__[queryset_name](train=True, transform=transform, test_frac=0)
    batch_size = params['batch_size']
    transfer_out_path = osp.join(out_path, 'transferset.pickle')
    adversary = Adversary(blackbox, queryset, label_only=args.label_only, batch_size=batch_size)

    print('=> constructing transfer set...')
    transferset = adversary.get_transferset()
    with open(transfer_out_path, 'wb') as wf:
        pickle.dump(transferset, wf)
    print('=> transfer set ({} samples) written to: {}'.format(len(transferset), transfer_out_path))

if __name__ == '__main__':
    main()
