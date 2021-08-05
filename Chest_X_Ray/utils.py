import numpy as np
import pandas as pd
import torch
from math import log
import os
import torch.nn.functional as F 
import torch
from torch import nn,optim
from torchvision.datasets import ImageFolder
from torchvision import transforms as T,datasets,models
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, ConcatDataset, random_split, WeightedRandomSampler, Dataset
from collections import OrderedDict
from tqdm import tqdm
from sklearn import metrics
import seaborn as sns
import time
import matplotlib.pyplot as plt

torch.manual_seed(0)
torch.cuda.manual_seed(0)

def data_transforms(phase = None):
    
    if phase == 'train':
        data_T = T.Compose([T.Resize(255),
                            T.CenterCrop(224),
                            T.RandomHorizontalFlip(),
                            T.RandomRotation(10),
                            T.RandomGrayscale(),
                            T.RandomAffine(translate=(0.05,0.05), degrees=0),
                            T.ToTensor()
                           ])
    
    elif phase == 'test' or phase == 'val':

        data_T = T.Compose([T.Resize(255),
                            T.CenterCrop(224),
                            T.ToTensor()
                          ])
        
    return data_T

def get_class_distribution(dataset_obj):
    
    # Ref: https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a
    
    class_to_idx = {'NORMAL': 0, 'PNEUMONIA': 1}
    
    idx2class = {v: k for k, v in class_to_idx.items()}
    
    count_dict = {k:0 for k,v in class_to_idx.items()}
    
    for element in tqdm(dataset_obj):
        y_lbl = element[1]
        y_lbl = idx2class[y_lbl]
        count_dict[y_lbl] += 1
            
    return count_dict

def without_split_dataset():
    data_dir = 'chest_xray'

    TRAIN = 'train' # Contains training images
    TEST = 'test' # Contains test images
    VAL = 'val' # Contains validation images
    
    trainset = ImageFolder(os.path.join(data_dir, TRAIN), transform = data_transforms(TRAIN))
    validset = ImageFolder(os.path.join(data_dir, VAL), transform = data_transforms(VAL))
    testset = ImageFolder(os.path.join(data_dir, TEST), transform = data_transforms(TEST))
    
    torch.manual_seed(0);
    torch.cuda.manual_seed(0)
    
    target_list = torch.tensor(trainset.targets)
    target_list = target_list[torch.randperm(len(target_list))]
    
    class_count = [i for i in tqdm(get_class_distribution(trainset).values())]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
    
    class_weights_all = class_weights[target_list]
    
    weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
    )
    
    batch_size=16
    
    torch.manual_seed(0);
    torch.cuda.manual_seed(0)

    train_dl = DataLoader(trainset, batch_size=batch_size, sampler=weighted_sampler)
    val_dl = DataLoader(validset, batch_size=batch_size)
    test_dl = DataLoader(testset, batch_size=batch_size)
    
    return train_dl, val_dl, test_dl, trainset, validset, testset

def with_split_dataset():
    data_dir = 'chest_xray'

    TRAIN = 'train' # Contains training images
    TEST = 'test' # Contains test images
    VAL = 'val' # Contains validation images
    
    trainset = ImageFolder(os.path.join(data_dir, TRAIN), transform = data_transforms(TRAIN))
    validset = ImageFolder(os.path.join(data_dir, VAL), transform = data_transforms(VAL))
    testset = ImageFolder(os.path.join(data_dir, TEST), transform = data_transforms(TEST))
    
    l = []
    l.append(trainset)
    l.append(validset)
    dataset = ConcatDataset(l)
    
    train_size = round(len(dataset)*0.9) # 90%
    val_size = len(dataset) - train_size # 10%
    
    torch.manual_seed(0);
    torch.cuda.manual_seed(0)
    
    trainset, validset = random_split(dataset, [train_size, val_size])
    
    torch.manual_seed(0);
    torch.cuda.manual_seed(0)
    
    train_labels = []
    for i in tqdm(range(len(trainset))):
        train_labels.append(trainset[i][1])
    
    target_list = torch.tensor(train_labels)
    target_list = target_list[torch.randperm(len(target_list))]
    
    class_count = [i for i in tqdm(get_class_distribution(trainset).values())]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
    
    class_weights_all = class_weights[target_list]
    
    weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
    )
    
    batch_size=16

    torch.manual_seed(0);
    torch.cuda.manual_seed(0)
    
    train_dl = DataLoader(trainset, batch_size=batch_size, sampler=weighted_sampler)
    val_dl = DataLoader(validset, batch_size=batch_size)
    test_dl = DataLoader(testset, batch_size=batch_size)
    
    return train_dl, val_dl, test_dl, trainset, validset, testset

# Ref for Device Loading: https://jovian.ai/learn/deep-learning-with-pytorch-zero-to-gans/lesson/lesson-3-training-deep-neural-networks-on-a-gpu
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device) # yield will stop here, perform other steps, and the resumes to the next loop/batch

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


