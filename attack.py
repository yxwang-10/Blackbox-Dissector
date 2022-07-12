import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.utils.data.sampler  as sampler
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader
from torch.autograd import Variable, Function

import numpy as np
import argparse
import pickle
import random
import os
import tqdm
import time
import copy
# import glob
# import pdb
# import shutil
import cv2

import blackbox_model.models.zoo as zoo
from blackbox_model.victim.blackbox import Blackbox
import blackbox_model.datasets as datasets
from PIL import Image

# import wandb
import warnings
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from scipy.stats import entropy
from my_transform import RandomErasing, PrioriErasing, PrioriPatchErasing
from sampler import Kcenter_sampler, Entropy_sampler

warnings.filterwarnings('ignore')
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument("--test_dataset", type=str, default='CIFAR10')
parser.add_argument("--dataset_path", type=str, default='./label_only/cifar/transferset.pickle')
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--modelname", type=str, default="resnet34")
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--train_epochs", type=int, default=200)
parser.add_argument("--initial_budget", type=int, default=100)
parser.add_argument("--blackbox_dir", type=str, default='models/victim/cifar10-resnet34')
parser.add_argument("--save_dir", type=str, default='./label_only/cifar/our/')
parser.add_argument("--tmp_dir", type=str, default='./images/cifar_tmp_our/')
parser.add_argument("--sampling_strategy", type=str, default='random')
parser.add_argument("--lr", type=float, default=0.02)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--sh", type=float, default=0.1)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument("--step_size", type=int, default=60)
parser.add_argument("--erase_rate", type=float, default=0.25)

args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

if not os.path.exists(args.tmp_dir):
    os.mkdir(args.tmp_dir)

cifar_train_transform = transforms.Compose([
                        transforms.Resize(32),
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                             std=(0.2023, 0.1994, 0.2010)),
                        ])

mnist_train_transform = transforms.Compose([
                        transforms.Resize(28),
                        transforms.Grayscale(num_output_channels=1),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ])

imagenet_train_transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                           ])

def init_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_pseudo_label(task_model, img, num=10):
    tmp_e_trans = RandomErasing(probability=1, sh=args.sh, mean=[0, 0, 0])
    e_imgs = []
    for j in range(num):
        e_imgs.extend(tmp_e_trans(img.clone()).unsqueeze(0))
    e_imgs = torch.stack(e_imgs, 0)
    with torch.no_grad():
        pre = (F.softmax(task_model(e_imgs.cuda()), dim=1).cpu().sum(0).unsqueeze(0))/num
    return pre

def get_soft_targets(task_model, imgs):
    e_trans = RandomErasing(probability=1, sh=args.sh, mean=[0,0,0])
    e_label = []
    task_model.eval()

    for i in range(imgs.size(0)):
        img = imgs[i]
        e_imgs = []
        for j in range(10):
            e_imgs.extend(e_trans(img.clone()).unsqueeze(0))
        e_imgs = torch.stack(e_imgs, 0)
        with torch.no_grad():
            e_label.extend((F.softmax(task_model(e_imgs.cuda()), dim=1).cpu().sum(0).unsqueeze(0))/10)
    e_label = torch.stack(e_label, 0)
    task_model.train()

    return e_label

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == None:
                continue
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            elif "view" in name.lower():
                x = x.view(x.size(0),-1)
            elif "classifier" in name.lower():
                x = x.view(x.size(0),-1)
                x = module(x)
            else:
                x = module(x)
        
        return target_activations, x

class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

class TransferSetImagePaths(ImageFolder):
    """TransferSet Dataset, for when images are stored as *paths*"""

    def __init__(self, samples, transform=None, target_transform=None):
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform

class NewImageNet(Dataset):
    def __init__(self, samples, transform=None, target_transform=None):
        self.imagenet = TransferSetImagePaths(samples=samples, transform=transform)

    def __len__(self):
        return len(self.imagenet)
    
    def __getitem__(self, index):
        
        if isinstance(index, np.float64):
            index = index.astype(np.int64)
    
        data, target = self.imagenet[index]

        return data, target, index

def preprocess_img(sample, tmp_dir, labeled=True):
    new_sample = []
    t = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    if labeled:
        for path, one_hot in tqdm.tqdm(sample):
            img = Image.open(path).convert("RGB")
            img = t(img)
            new_path = tmp_dir + path.split('/')[-1]
            torchvision.utils.save_image(img.clone(), new_path)
            new_sample.append((new_path, one_hot))
    else:
        for path in tqdm.tqdm(sample):
            img = Image.open(path).convert("RGB")
            img = t(img)
            new_path = tmp_dir + path.split('/')[-1]
            torchvision.utils.save_image(img.clone(), new_path)
            new_sample.append(new_path)

    return new_sample

def selected_with_kcenter(k_model, labeled_indices, all_indices, dataset_name, split, all_sample):
    kcenter_sampler = Kcenter_sampler()
    unlabeled_indices = np.setdiff1d(list(all_indices), labeled_indices)

    if dataset_name == 'CIFAR10':
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    elif dataset_name == 'SVHN':
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    elif dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.Resize(28), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize(mean=[0.1307,], std=[0.3081,])])
    else:
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    all_dataset = NewImageNet(samples=all_sample, transform=transform)
    unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
    labeled_sampler = data.sampler.SubsetRandomSampler(labeled_indices)
    unlabeled_dataloader = data.DataLoader(all_dataset, sampler=unlabeled_sampler, batch_size=128, drop_last=False, num_workers=8)
    labeled_dataloader = data.DataLoader(all_dataset, sampler=labeled_sampler, batch_size=128, drop_last=False, num_workers=8)
    sampled_indices = kcenter_sampler.sampler(k_model.cuda(), unlabeled_dataloader, True, split, labeled_dataloader, False)

    return sampled_indices

def get_pseudo_label_with_kcenter(task_model, labeled_indices, all_indices, dataset_name, split, all_sample, tmp_dir):
    new_sample = []
    kcenter_sampler = Kcenter_sampler()
    unlabeled_indices = np.setdiff1d(list(all_indices), labeled_indices)

    if dataset_name == 'CIFAR10':
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    elif dataset_name == 'SVHN':
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    elif dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.Resize(28), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize(mean=[0.1307,], std=[0.3081,])])
    else:
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    t = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

    all_dataset = NewImageNet(samples=all_sample, transform=transform)
    unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
    labeled_sampler = data.sampler.SubsetRandomSampler(labeled_indices)
    unlabeled_dataloader = data.DataLoader(all_dataset, sampler=unlabeled_sampler, batch_size=128, drop_last=False, num_workers=8)
    labeled_dataloader = data.DataLoader(all_dataset, sampler=labeled_sampler, batch_size=128, drop_last=False, num_workers=8)
    sampled_indices = kcenter_sampler.sampler(task_model.cuda(), unlabeled_dataloader, True, split, labeled_dataloader, False)

    for i in tqdm.tqdm(sampled_indices):
        path, _ = all_sample[i]
        img = Image.open(path).convert("RGB")
        img = t(img)
        new_path = tmp_dir + path.split('/')[-1]
        torchvision.utils.save_image(img.clone(), new_path)

        img = Image.open(path).convert("RGB")
        img = transform(img)
        pseudo_label = get_pseudo_label(task_model, img.clone())
        new_sample.append((new_path, pseudo_label.squeeze()))

    return new_sample, sampled_indices

def get_pseudo_label_with_entropy(task_model, labeled_indices, all_indices, dataset_name, split, all_sample, tmp_dir):
    new_sample = []
    entropy_sampler = Entropy_sampler
    unlabeled_indices = np.setdiff1d(list(all_indices), labeled_indices)

    if dataset_name == 'CIFAR10':
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    elif dataset_name == 'SVHN':
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    elif dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.Resize(28), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize(mean=[0.1307,], std=[0.3081,])])
    else:
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    t = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

    all_dataset = NewImageNet(samples=all_sample, transform=transform)
    unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
    labeled_sampler = data.sampler.SubsetRandomSampler(labeled_indices)
    unlabeled_dataloader = data.DataLoader(all_dataset, sampler=unlabeled_sampler, batch_size=128, drop_last=False, num_workers=10)
    labeled_dataloader = data.DataLoader(all_dataset, sampler=labeled_sampler, batch_size=128, drop_last=False, num_workers=10)
    sampled_indices = entropy_sampler.sampler(task_model.cuda(), unlabeled_dataloader, True, split)

    for i in tqdm.tqdm(sampled_indices):
        path, _ = all_sample[i]
        img = Image.open(path).convert("RGB")
        img = t(img)
        new_path = tmp_dir + path.split('/')[-1]
        torchvision.utils.save_image(img.clone(), new_path)

        img = Image.open(path).convert("RGB")
        img = transform(img)
        pseudo_label = get_pseudo_label(task_model, img.clone())
        new_sample.append((new_path, pseudo_label.squeeze()))

    return new_sample, sampled_indices

def get_pseudo_label_with_random(task_model, labeled_indices, all_indices, dataset_name, split, all_sample, tmp_dir):
    new_sample = []
    unlabeled_indices = np.setdiff1d(list(all_indices), labeled_indices)

    if dataset_name == 'CIFAR10':
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    elif dataset_name == 'SVHN':
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    elif dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.Resize(28), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize(mean=[0.1307,], std=[0.3081,])])
    else:
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    t = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

    all_dataset = NewImageNet(samples=all_sample, transform=transform)
    sampled_indices = random.sample(list(unlabeled_indices), split)

    for i in tqdm.tqdm(sampled_indices):
        path, _ = all_sample[i]
        img = Image.open(path).convert("RGB")
        img = t(img)
        new_path = tmp_dir + path.split('/')[-1]
        torchvision.utils.save_image(img.clone(), new_path)

        img = Image.open(path).convert("RGB")
        img = transform(img)
        pseudo_label = get_pseudo_label(task_model, img.clone())
        new_sample.append((new_path, pseudo_label.squeeze()))

    return new_sample, sampled_indices

def erase_and_save(task_model, indices, all_sample, save_dir, dataset_name, split, budget, num=10):
    
    e_dir = []
    e_pres = []
    if dataset_name == 'CIFAR10':
        e_trans = PrioriPatchErasing(probability=1, sh=0.1, mean=[0.4914, 0.4822, 0.4465])
        norm = transforms.Compose([transforms.ToPILImage(), transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
        mask_trans = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
        if args.modelname == 'resnet34':
            grad_cam = GradCam(model=task_model, feature_module=task_model.layer3, target_layer_names=["4"], use_cuda=True)
        elif args.modelname == 'resnet18':
            grad_cam = GradCam(model=task_model, feature_module=task_model.layer3, target_layer_names=["2"], use_cuda=True)
        elif args.modelname == 'resnet50':
            grad_cam = GradCam(model=task_model, feature_module=task_model.layer3, target_layer_names=["8"], use_cuda=True)
        elif args.modelname == 'densenet':
            grad_cam = GradCam(model=task_model, feature_module=task_model.dense3, target_layer_names=["2"], use_cuda=True)
        elif args.modelname == 'vgg16':
            grad_cam = GradCam(model=task_model, feature_module=task_model.features, target_layer_names=["28"], use_cuda=True)
    elif dataset_name == 'SVHN':
        e_trans = PrioriPatchErasing(probability=1, sh=0.1, mean=[0.4914, 0.4822, 0.4465])
        norm = transforms.Compose([transforms.ToPILImage(), transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
        mask_trans = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
        grad_cam = GradCam(model=task_model, feature_module=task_model.layer3, target_layer_names=["4"], use_cuda=True)
    elif dataset_name == 'MNIST':
        e_trans = PrioriPatchErasing(probability=1, sh=0.1, mean=[0.1307, 0.1307, 0.1307])
        norm = transforms.Compose([transforms.ToPILImage(), transforms.Resize(28), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mask_trans = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.Resize(28), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize(mean=[0.1307,], std=[0.3081,])])
        grad_cam = GradCam(model=task_model, feature_module=task_model.layer1, target_layer_names=["2"], use_cuda=True)
    else:
        e_trans = PrioriPatchErasing(probability=1, sh=0.1, mean=[0.485, 0.456, 0.406])
        norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        mask_trans = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        grad_cam = GradCam(model=task_model, feature_module=task_model.layer4, target_layer_names=["2"], use_cuda=True)
    t = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

    for i in tqdm.tqdm(indices):
        path, one_hot = all_sample[i]
        ID = str((path.split('/')[-1]).split('.')[0])
        img = Image.open(path).convert("RGB")
        img = t(img)
        cam_input = Image.open(path).convert("RGB")
        cam_input = mask_trans(cam_input).unsqueeze(0)
        mask = grad_cam(cam_input)
        tmp_e_img = []
        tmp_e_pres = []
        for j in range(num):
            e = e_trans(img.clone(), mask)
            tmp_e_img.append(e)
        for e in tmp_e_img:
            with torch.no_grad():
                task_model.eval()
                pred = task_model(norm(e.clone()).unsqueeze(0).cuda())
                pred = F.softmax(pred, dim=1).cpu().data
                tmp_e_pres.append(pred)
        tmp_score = [soft_cross_entropy(p.unsqueeze(0), one_hot) for p in tmp_e_pres]
        tmp_score = torch.from_numpy(np.array(tmp_score)).view(-1)
        _, t_indices = tmp_score.max(0)
        save_path = save_dir + '{}_e_{}.JPEG'.format(ID, split)
        e_dir.append(save_path)
        e_pres.append(tmp_e_pres[t_indices])
        torchvision.utils.save_image(tmp_e_img[t_indices].clone(), save_path)
            
    scores = [p.max() for p in e_pres]
    scores = torch.from_numpy(np.array(scores))
    scores = scores.view(-1)
    _, querry_indices = torch.topk(scores, budget)
    tmp = []
    selected_indices = []
    for i in querry_indices:
        tmp.append(e_dir[i])
        selected_indices.append(indices[i])
    
    return tmp, selected_indices

def query(blackbox_model, path, dataset_name):
    
    new = []
    if dataset_name == 'CIFAR10':
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    elif dataset_name == 'SVHN':
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    elif dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.Resize(28), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize(mean=[0.1307,], std=[0.3081,])])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    for p in path:
        img = Image.open(p).convert("RGB")
        with torch.no_grad():
            e_true = blackbox(transform(img).cuda().unsqueeze(0)).cpu().squeeze()
        argmax_k = e_true.argmax()
        y_i_1hot = torch.zeros_like(e_true)
        y_i_1hot[argmax_k] = 1.
        new.append((p,y_i_1hot))
    
    return new

def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights.unsqueeze(-1).float(), 1))
        # return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
    else:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))

def pseudo_loss(outputs, one_hot_targets, soft_targets):

    alpha = 0.5
    T = 2
    loss_CE = soft_cross_entropy(outputs, one_hot_targets)
    loss_p = soft_cross_entropy(outputs/T, F.softmax(soft_targets/T, dim=1)) * (T*T)
    loss = (1. - alpha)*loss_CE + alpha*loss_p

    return loss, loss_CE, loss_p

class Solver:
    def __init__(self, args, test_dataloader):
        self.args = args
        self.test_dataloader = test_dataloader
        self.ce_loss = soft_cross_entropy
        self.loss_pseudo = pseudo_loss
        self.mse_loss = nn.MSELoss(reduce=True, size_average=True)

    def train(self, task_dataloader, task_model, split):

        criterion = self.ce_loss
        optim_task_model = optim.SGD(task_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        task_scheduler = optim.lr_scheduler.StepLR(optim_task_model, step_size=self.args.step_size, gamma=0.1)
        task_model.train()        
        best_acc = 0
        print('=> start train task model with split:{}'.format(split))
        best_train_acc, train_acc = -1., -1.
        
        for iter_count in range(1, self.args.train_epochs + 1):
        # for iter_count in range(1):
            task_scheduler.step(iter_count)
            train_loss, train_acc = self.train_step(split, task_model, task_dataloader, criterion, optim_task_model, iter_count, 'cuda', log_interval=10, param=self.args)
            best_train_acc = max(best_train_acc, train_acc)
            acc = self.test(task_model, iter_count)
            if acc > best_acc:
                 best_acc = acc
                 best_model = copy.deepcopy(task_model.cpu())
            print('best test acc: ', best_acc)
            torch.cuda.empty_cache()
        
        final_accuracy = self.test(best_model, self.args.train_epochs + 1, True)
        savepath = os.path.join(self.args.save_dir, 'checkpoint_model_budget_{}_acc_{}.pth'.format(split, final_accuracy))
        torch.save(best_model.state_dict(), savepath)
        
        return final_accuracy, best_model
             
    def test(self, task_model, epoch, silence=False):
        task_model = task_model.cuda()
        task_model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for imgs, labels in self.test_dataloader:
                if self.args.cuda:
                    imgs = imgs.cuda()
                preds = task_model(imgs)
                preds = torch.argmax(preds, dim=1).cpu().numpy()
                correct += accuracy_score(labels, preds, normalize=False)
                total += imgs.size(0)
        acc = correct / total * 100
        if not silence:
            print('[Test]  Epoch: {}\tAccuracy: {:.1f} ({}/{})'.format(epoch, acc, correct, total))
        return acc

    def train_step(self, split, model, train_loader, criterion, optimizer, epoch, device, log_interval=10, param=None):
        model = model.cuda()
        model.train()
        train_loss = 0.
        correct = 0
        total = 0
        train_loss_batch = 0
        epoch_size = split
        t_start = time.time()
    
        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            
            inputs, targets = inputs.to(device), targets.to(device)
    
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    
            _, predicted = outputs.max(1)
            total += targets.size(0)
            if len(targets.size()) == 2:
                # Labels could be a posterior probability distribution. Use argmax as a proxy.
                target_probs, target_labels = targets.max(1)
            else:
                target_labels = targets
            correct += predicted.eq(target_labels).sum().item()
    
            prog = total / epoch_size
            exact_epoch = epoch + prog - 1
            acc = 100. * correct / total
            train_loss_batch = train_loss / total
    
            if (batch_idx + 1) % log_interval == 0:
                print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.1f} ({}/{})'.format(
                    exact_epoch, batch_idx * len(inputs), split, 100. * batch_idx / len(train_loader),
                    loss.item(), acc, correct, total))
    
        t_end = time.time()
        t_epoch = int(t_end - t_start)
        acc = 100. * correct / total
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
        return train_loss_batch, acc


def similar_evaluation(task_model, black_box, test_dataloader):
    total = 0
    result = 0
    task_model.eval()
    task_model = task_model.cuda()
    for images, _ in test_dataloader:
        images = images.cuda()
        with torch.no_grad():
            task_preds = task_model(images).cpu().data.numpy().argmax(1)
            black_box_preds = black_box(images).cpu().data.numpy().argmax(1)
        result += np.sum(task_preds == black_box_preds)
        total += images.size(0)

    return (result / total) * 100

init_seed()
labeled_sample = []
pseudo_path = []

dataset_name = args.test_dataset
modelfamily = datasets.dataset_to_modelfamily[dataset_name]
transform = datasets.modelfamily_to_transforms[modelfamily]['test']
dataset = datasets.__dict__[dataset_name]
testset = dataset(train=False, transform=transform)
test_dataloader = data.DataLoader(testset, drop_last=False, batch_size=args.batch_size, shuffle=True, num_workers=16)
solver = Solver(args, test_dataloader)
blackbox = Blackbox.from_modeldir(args.blackbox_dir, 'cuda')

if args.test_dataset == 'CIFAR10':
    transform = cifar_train_transform
elif args.test_dataset == 'SVHN':
    transform = cifar_train_transform
elif args.test_dataset == 'MNIST':
    transform = mnist_train_transform
else:
    transform = imagenet_train_transform

with open(args.dataset_path, 'rb') as rf:
    sample = pickle.load(rf)
num_classes = sample[0][1].size(0)
all_indices = set(np.arange(len(sample)))
labeled_indices = random.sample(list(all_indices), args.initial_budget)
new = []
for i in labeled_indices:
    new.append(torch.tensor(i))
labeled_indices = new
for i in labeled_indices:
    labeled_sample.append(sample[i])
labeled_sample = preprocess_img(labeled_sample, args.tmp_dir, labeled=True)
erase_indices = []

splits = [100, 200, 500, 800, 1000, 2000, 5000, 10000, 20000, 30000]
budgets = [  100, 300, 300, 200, 1000, 3000, 5000, 10000, 10000, 0]

acc_path = args.save_dir + 'acc.log'
sample_path = args.save_dir + 'sample.pickle'
indices_path = args.save_dir + 'indices.pickle'

for split, budget in zip(splits, budgets):

    # First training, using labeled data.
    task_model = zoo.get_net(modelname=args.modelname, modeltype=modelfamily, pretrained=None, num_classes=num_classes)
    labeled_dataset = NewImageNet(samples=labeled_sample,transform=transform)
    labeled_dataloader = data.DataLoader(labeled_dataset, batch_size=128, drop_last=False, num_workers=args.num_workers, shuffle=True)
    acc, task_model = solver.train(labeled_dataloader, task_model, split)
    similar = similar_evaluation(task_model, blackbox, test_dataloader)
    print("acc:{}, similarty:{}".format(acc, similar))
    with open(acc_path, 'a') as af:
        af.write(str(split) + ' ' + 'similar:' + str(similar) + '\n')
        af.write(str(split) + ' ' + 'acc:' + str(acc) + '\n')
    k_model = copy.deepcopy(task_model)
    
    
    # Getting pseudo label.
    pseudo_sample, pseudo_indices = get_pseudo_label_with_random(task_model, labeled_indices, all_indices, args.test_dataset, split, sample, args.tmp_dir)
    torch.cuda.empty_cache()

    # Second training, using labeled data and pseudo labeled data.
    task_model = zoo.get_net(modelname=args.modelname, modeltype=modelfamily, pretrained=None, num_classes=num_classes)
    train_dataset = NewImageNet(samples=labeled_sample+pseudo_sample,transform=transform)
    train_dataloader = data.DataLoader(train_dataset, batch_size=128, drop_last=False, num_workers=args.num_workers, shuffle=True)
    acc, task_model = solver.train(train_dataloader, task_model, split*2)
    similar = similar_evaluation(task_model, blackbox, test_dataloader)
    print("acc:{}, similarty:{}".format(acc, similar))
    with open(acc_path, 'a') as af:
        af.write(str(split) + ' ' + 'similar:' + str(similar) + '\n')
        af.write(str(split) + ' ' + 'acc:' + str(acc) + '\n')
    torch.cuda.empty_cache()

    # Select new samples, where the original pictures and the erased pictures each account for 50%.
    unerase_indices = np.setdiff1d(list(labeled_indices), erase_indices)
    if len(unerase_indices) >= int(budget*args.erase_rate):
        erase_path, selected_indices = erase_and_save(task_model, unerase_indices, sample, args.tmp_dir, args.test_dataset, split, int(budget*args.erase_rate))
        erase_indices += selected_indices
        erase_sample = query(blackbox, erase_path, args.test_dataset)
        labeled_sample += erase_sample
        if args.sampling_strategy == 'random':
            selected_indices = random.sample(list(np.setdiff1d(list(all_indices), labeled_indices)), int(budget*(1-args.erase_rate)))
        elif args.sampling_strategy =='kcenter':
            selected_indices = selected_with_kcenter(k_model, labeled_indices, all_indices, args.test_dataset, int(budget*(1-args.erase_rate)), sample)
        else:
            raise ValueError("Unrecognized strategy")
    else:
        erase_path, selected_indices = erase_and_save(task_model, unerase_indices, sample, args.tmp_dir, args.test_dataset, split, len(unerase_indices))
        erase_indices += selected_indices
        erase_sample = query(blackbox, erase_path, args.test_dataset)
        labeled_sample += erase_sample
        if args.sampling_strategy == 'random':
            selected_indices = random.sample(list(np.setdiff1d(list(all_indices), labeled_indices)), int(budget - len(unerase_indices)))
        elif args.sampling_strategy =='kcenter':
            selected_indices = selected_with_kcenter(k_model, labeled_indices, all_indices, args.test_dataset, int(budget - len(unerase_indices)), sample)
        else:
            raise ValueError("Unrecognized strategy")
    labeled_indices += list(selected_indices)
    tmp = []
    for i in selected_indices:
        tmp.append(sample[i])
    labeled_sample += preprocess_img(tmp, args.tmp_dir, labeled=True)

    with open(sample_path, 'wb') as wf:
        pickle.dump(labeled_sample, wf)
    with open(indices_path, 'wb') as wf:
        pickle.dump(labeled_indices, wf)
    
    del task_model
    torch.cuda.empty_cache()
