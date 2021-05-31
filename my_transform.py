import os
import os.path
import errno
import numpy as np
import sys
import cv2
from PIL import Image,  ImageFilter, ImageEnhance
import torchvision.transforms as transforms
import torchvision
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
from torchvision.datasets import ImageFolder
import torch
import random
import math

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


class PatchTransform(object):
    def __init__(self, k = 2):
        self.k = k

    def __call__(self, xtensor:torch.Tensor):
        '''
        X: torch.Tensor of shape(c, h, w)   h % self.k == 0
        :param xtensor:
        :return:
        '''
        patches = []
        c, h, w = xtensor.size()
        dh = h // self.k
        dw = w // self.k

        #print(dh, dw)
        sh = 0
        for i in range(h // dh):
            eh = sh + dh
            eh = min(eh, h)
            sw = 0
            for j in range(w // dw):
                ew = sw + dw
                ew = min(ew, w)
                patches.append(xtensor[:, sh:eh, sw:ew])

                #print(sh, eh, sw, ew)
                sw = ew
            sh = eh
            
        random.shuffle(patches)

        start = 0
        imgs = []
        for i in range(self.k):
            end = start + self.k
            imgs.append(torch.cat(patches[start:end], dim = 1))
            start = end
        img = torch.cat(imgs, dim = 2)
        return img


class SaturationTrasform(object):
    '''
    for each pixel v: v' = sign(2v - 1) * |2v - 1|^{2/p}  * 0.5 + 0.5
    then clip -> (0, 1)
    '''

    def __init__(self, saturation_level = 2.0):
        self.p = saturation_level

    def __call__(self, img):

        ones = torch.ones_like(img)
        #print(img.size(), torch.max(img), torch.min(img))
        ret_img = torch.sign(2 * img - ones) * torch.pow( torch.abs(2 * img - ones), 2.0/self.p)

        ret_img =  ret_img * 0.5 + ones * 0.5

        ret_img = torch.clamp(ret_img,0,1)

        return ret_img


class EdgeTransform(object):
    '''
    :param object:
    :return:
    '''
    def __init__(self,):
        pass
    def __call__(self, img):
        img = img.filter( ImageFilter.FIND_EDGES)
        return img

class BrightnessTransform(object):
    '''
    :param object:
    :return:
    '''
    def __init__(self, bright = 1.0):
        self.b = bright
    def __call__(self, img):
        img = ImageEnhance.Brightness(img).enhance(self.b)
        return img

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

class PrioriErasing(object):

    def __init__(self, s=0.1, mean=[0.4914, 0.4822, 0.4465]):

        self.mean = mean
        self.s = s
       
    def __call__(self, img, priori):
        
        size = img.shape[-1]
        
        priori = priori / priori.sum()
        p = np.reshape(priori, (-1,1)).squeeze()
        indices = np.random.choice(int(size*size), int(self.s*size*size), replace=False, p=p)
        mask = np.zeros(size*size)
        for i in indices:
            mask[i] = 1.0
        mask = np.reshape(mask, (size, size))
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        mean_mask = torch.ones_like(img)
        if len(self.mean) == 3:
            mean_mask[0] = self.mean[0]
            mean_mask[1] = self.mean[1]
            mean_mask[2] = self.mean[2]
        else:
            mean_mask[0] = self.mean[0]
        img = img * (1-mask) + mean_mask * mask

        return img

class PrioriPatchErasing(object):

    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    
    def __call__(self, img, mask):

        if random.uniform(0, 1) > self.probability:
            return img
        
        max_x, max_y = np.where(mask == np.max(mask))
        if len(max_x) > 0:
            max_x = max_x[0]
            max_y = max_y[0]
        else:
            max_x = random.choice(range(mask.shape[1]))
            max_y = random.choice(range(mask.shape[1]))
            
        if mask.shape[1] != img.size()[1]:
            max_x = int((max_x / mask.shape[1]) * img.size()[1])
            max_y = int((max_y / mask.shape[1]) * img.size()[1])

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                # x1 = random.randint(max_x - h, img.size()[1] - h)
                # y1 = random.randint(max_y - w, img.size()[2] - w)
                x1 = random.randint(max_x - h, min(img.size()[1] - h, max_x))
                y1 = random.randint(max_y - w, min(img.size()[2] - w, max_y))
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                
                return img

        return img