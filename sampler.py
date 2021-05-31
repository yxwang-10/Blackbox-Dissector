import torch
import torch.nn as nn
import random
import numpy as np
from scipy import spatial
from scipy.stats import entropy
# from sklearn.metrics import pairwise_distances
import torch.nn.functional as F

import tqdm
import math
import pdb

from my_transform import RandomErasing
e_trans = RandomErasing(probability=1, mean=[0,0,0])

"""
def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)
"""

class Random_sampler:
    def __init__(self, budget):
        self.budget = budget

    def sampler(self, data, budget):
        all_indices = []

        for _, _, indices in tqdm.tqdm(data):
            all_indices.extend(indices)

        sampler_indices = random.sample(all_indices, budget)

        return sampler_indices

class Entropy_sampler:
    def __init__(self, budget):
        self.budget = budget
        self.entropy_score = entropy

    def sampler(self, task_model, data, cuda, budget):
        print('=> start Entropy sampler for budget:{}'.format(budget))
        all_preds = []
        all_indices = []
        task_model.eval()

        for images, _, indices in tqdm.tqdm(data):

            if cuda:
                images = images.cuda()
            with torch.no_grad():
                preds = task_model(images)
                preds = F.softmax(preds, dim=1).cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        
        all_scores = [self.entropy_score(pred.numpy()) for pred in all_preds]
        all_scores = torch.from_numpy(np.array(all_scores))
        all_scores = all_scores.view(-1)

        _, querry_indices = torch.topk(all_scores, int(budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]

        return querry_pool_indices

class Kcenter_sampler:
    def __init__(self, metric='euclidean'):
        self.metric = metric
        self.min_distances = None
        # self.pairwise_distances = pairwise_distances
        # self.pairwise_distances = spatial.distance.cdist
        self.pairwise_distances = torch.cdist

    def initial_distances(self, cluster_centers, cluster_centers_features, features, reset_dist=False):
        batch_num = math.ceil(len(features)/10000)
        if reset_dist:
            self.min_distances = None
        for num, i in enumerate(range(batch_num)):
            dist = self.pairwise_distances(features[(num*10000):((num+1)*10000)].cuda(), cluster_centers_features.cuda(), 2)
            dist = np.asarray(dist.cpu())
            if num==0:
                self.min_distances = np.min(dist, axis=1).reshape(-1,1)
            else:
                self.min_distances = np.concatenate((self.min_distances, np.min(dist, axis=1).reshape(-1,1)))

    def update(self, new_cluster_centers_features, features):
        dist = self.pairwise_distances(features.cuda(), new_cluster_centers_features.unsqueeze(0).cuda(), 2)
        dist = np.asarray(dist.cpu())
        tmp_min_distances = np.min(dist, axis=1).reshape(-1,1)
        self.min_distances = np.minimum(self.min_distances, tmp_min_distances)
        
    def sampler(self, task_model, data, cuda, budget, current_data, initial=True, n_batches=1):
        task_model.eval()
        print('=> start k-Center sampler for budget:{}'.format(budget))
        point = []
        cluster_centers_features = []
        cluster_centers_indices = []
        features = []
        all_indices = []

        for images, _, indices in tqdm.tqdm(data):

            if cuda:
                images = images.cuda()
            with torch.no_grad():
                preds = task_model(images).cpu().data.numpy()
            features.extend(preds)
            all_indices.extend(indices)
            # break

        if not current_data == None:
            for images, _, indices in tqdm.tqdm(current_data):
                
                if cuda:
                    images = images.cuda()
                with torch.no_grad():
                    preds = task_model(images).cpu().data.numpy()
                cluster_centers_features.extend(preds)
                cluster_centers_indices.extend(indices)

        if not initial:
            self.initial_distances(cluster_centers_indices, torch.from_numpy(np.asarray(cluster_centers_features)), torch.from_numpy(np.asarray(features)))

        for _ in tqdm.tqdm(range(int(budget/n_batches))):
            if initial:
                ind = np.random.choice(np.arange(len(all_indices)))
                point.append(all_indices[ind])
                cluster_centers_indices.append(all_indices[ind])
                cluster_centers_features.append(features[ind])
                new_cluster_centers_features = features[ind]
                features.pop(ind)
                all_indices.pop(ind)
                self.initial_distances(cluster_centers_indices, torch.from_numpy(np.asarray(cluster_centers_features)), torch.from_numpy(np.asarray(features)))
                initial = False
            else:
                for _ in range(n_batches):
                    ind = np.argmax(self.min_distances)
                    point.append(all_indices[ind])
                    cluster_centers_indices.append(all_indices[ind])
                    cluster_centers_features.append(features[ind])
                    new_cluster_centers_features = features[ind]
                    features.pop(ind)
                    all_indices.pop(ind)
                    self.min_distances = np.delete(self.min_distances, ind, 0)
            
            self.update(torch.from_numpy(np.asarray(new_cluster_centers_features)), torch.from_numpy(np.asarray(features)))
        
        return point

class Importance_sampler:
    def __init__(self, budget):
        self.budget = budget

    def importance_score(self, v, high):
        var = np.var(v)
        MINVar = (1/len(v)) * (pow(((1/len(v)) - max(v)), 2) + (len(v) - 1) * pow(((1/len(v)) - (1-max(v)) / (len(v)-1)), 2))
        if high:
            return 1 - (MINVar * (max(v) / var))
        else:
            return MINVar * (max(v) / var)

    def sampler(self, task_model, data, cuda, budget, high=True):
        print('=> start Importance sampler for budget:{}'.format(budget))
        all_preds = []
        all_indices = []
        task_model.eval()

        for images, _, indices in tqdm.tqdm(data):

            if cuda:
                images = images.cuda()
            with torch.no_grad():
                preds = task_model(images)
                preds = F.softmax(preds, dim=1).cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_scores = [self.importance_score(pred.numpy(), high) for pred in all_preds]
        all_scores = torch.from_numpy(np.array(all_scores))
        all_scores = all_scores.view(-1)

        _, querry_indices = torch.topk(all_scores, int(budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]

        return querry_pool_indices

class Importance_erase_sampler:
    def __init__(self, budget):
        self.budget = budget

    def importance_score(self, v, high):
        var = np.var(v)
        MINVar = (1/len(v)) * (pow(((1/len(v)) - max(v)), 2) + (len(v) - 1) * pow(((1/len(v)) - (1-max(v)) / (len(v)-1)), 2))
        if high:
            return 1 - (MINVar * (max(v) / var))
        else:
            return MINVar * (max(v) / var)

    def sampler(self, task_model, data, cuda, budget, high=True):
        print('=> start Importance sampler for budget:{}'.format(budget))
        all_preds = []
        all_indices = []
        task_model.eval()

        for images, _, indices in tqdm.tqdm(data):

            if cuda:
                images = images.cuda()
            with torch.no_grad():
                for i in range(images.size(0)):
                    image = images[i]
                    e_images = []
                    for j in range(10):
                        e_images.extend(e_trans(image.clone()).unsqueeze(0))
                    e_images = torch.stack(e_images, 0)
                    all_preds.extend((F.softmax(task_model(e_images.cuda()), dim=1).cpu().sum(0).unsqueeze(0))/10)
                    # all_indices.extend(indices[i])
                    all_indices.append(indices[i])

        all_scores = [self.importance_score(pred.numpy(), high) for pred in all_preds]
        all_scores = torch.from_numpy(np.array(all_scores))
        all_scores = all_scores.view(-1)

        _, querry_indices = torch.topk(all_scores, int(budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]

        return querry_pool_indices

"""
class Model_sampler:
    def __init__(self, budget):
        self.budget = budget

    def sampler(self, small_model, data, cuda, budget):
        all_preds = []
        all_indices = []

        for images, _, indices in tqdm.tqdm(data):

            if cuda:
                images = images.cuda()
            with torch.no_grad():
                preds = nn.Sigmoid()(small_model(images)).cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)
        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)

        all_preds *= -1
        
        _, querry_indices = torch.topk(all_preds, int(budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]

        return querry_pool_indices

class Model_balance_sampler:
    def __init__(self, budget):
        self.budget = budget
    
    def sampler(self, task_model, small_model, data, cuda, budget, num_classes):
        n_sample = int(math.floor(budget/num_classes))
        all_indices = [[] for i in range(num_classes)]
        all_preds = [[] for i in range(num_classes)]
        list_sample = [i for i in range(num_classes)]
        random.shuffle(list_sample)

        for images, _, indices in tqdm.tqdm(data):

            if cuda:
                images = images.cuda()
            with torch.no_grad():
                preds = task_model(images)
                labels = torch.argmax(preds, dim=1)
                preds = nn.Sigmoid()(small_model(images)).cpu().data
            for i in range(len(labels)):
                all_indices[labels[i]].append(indices[i])
                all_preds[labels[i]].extend(preds[i])

        all_preds = [(torch.stack(all_preds[i]).view(-1)) * (-1) for i in range(num_classes)]
        small = []
        remain = budget
        for i in range(num_classes):
            if all_preds[i].shape[0] < int(n_sample + 5):
                small.append(i)
        list_sample = list(np.setdiff1d(list_sample, small))
        a = True
        for num, i in enumerate(small):
            a = False
            if num == 0:
                if n_sample > all_preds[i].shape[0]:
                    _, querry_indices = torch.topk(all_preds[i], all_preds[i].shape[0])
                    remain = remain - all_preds[i].shape[0]
                else:
                    _, querry_indices = torch.topk(all_preds[i], n_sample)
                    remain = remain - n_sample
                querry_pool_indices = np.asarray(all_indices[i])[querry_indices]
            else:
                if n_sample > all_preds[i].shape[0]:
                    _, querry_indices = torch.topk(all_preds[i], all_preds[i].shape[0])
                    remain = remain - all_preds[i].shape[0]
                else:
                    _, querry_indices = torch.topk(all_preds[i], n_sample)
                    remain = remain - n_sample
                querry_pool_indices = np.concatenate((querry_pool_indices, np.asarray(all_indices[i])[querry_indices]))
        n_sample = int(math.floor(remain/(num_classes - len(small))))
        for num, i in enumerate(list_sample):
            if num == (len(list_sample) - 1):
                _, querry_indices = torch.topk(all_preds[i], remain)
                querry_pool_indices = np.concatenate((querry_pool_indices, np.asarray(all_indices[i])[querry_indices]))
            else:
                if a and num==0:
                    _, querry_indices = torch.topk(all_preds[i], n_sample)
                    remain = remain - n_sample
                    querry_pool_indices = np.asarray(all_indices[i])[querry_indices]
                    a = False
                else:
                    _, querry_indices = torch.topk(all_preds[i], n_sample)
                    remain = remain - n_sample
                    querry_pool_indices = np.concatenate((querry_pool_indices, np.asarray(all_indices[i])[querry_indices]))

        return querry_pool_indices
"""