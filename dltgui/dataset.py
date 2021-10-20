import cv2
import PIL.Image as Image
import numpy as np
import random
import os
import os.path as osp

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose

from .utils.seg_transforms import *


class DatasetBase(Dataset):
    def __init__(self, root, task, n_class, img_transform=Compose([]),\
        label_transform=Compose([]), co_transform=Compose([]), seed=0, phase='train'):
        self.root = root
        self.n_class = n_class
        self.img_dir = osp.join(self.root, "JPEGImages")
        self.label_dir = osp.join(self.root, "Annotations")
        self.train_list = osp.join(self.root, "ImageSets/train.txt")
        self.val_list = osp.join(self.root, "ImageSets/val.txt")
        self.test_list = osp.join(self.root, "ImageSets/test.txt")
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.co_transform = co_transform
        self.phase = phase
        self.task = task

    def read_img_list(self, filename):
        with open(filename) as f:
            img_list = f.readlines()
        return np.array(img_list)


class DatasetClassification(DatasetBase):
    def __init__(self, root, task, n_class, img_suffix='.jpg', img_transform=Compose([]),\
        label_transform=Compose([]), co_transform=Compose([]), seed=0, phase='train'):
        # print("DatasetClassification: ", root)
        DatasetBase.__init__(self, root, task, n_class,img_transform, label_transform,
            co_transform, seed, phase)
        self.img_suffix = img_suffix
        # print("phase: ", self.phase)
        if self.phase == 'train':
            self.img_list = self.read_img_list(self.train_list)
        elif self.phase == 'test':
            self.img_list = self.read_img_list(self.test_list)
        
        self.label_list = []
        if self.phase == 'train':
            with open(osp.join(self.root, "ImageSets/train_label.txt")) as f:
                self.label_list = f.readlines()
        elif self.phase == 'test':
            with open(osp.join(self.root, "ImageSets/test_label.txt")) as f:
                self.label_list = f.readlines()
    
    def __getitem__(self, index):
        filename = self.img_list[index].strip()
        image = Image.open(osp.join(self.img_dir, filename + self.img_suffix))
        image = self.img_transform(image)

        line = self.label_list[index]
        label = int(line[line.find(' ')+1:])
        label = self.label_transform(label)
        
        return image, label

    def __len__(self):
        return len(self.img_list)

class DatasetSegmentation(DatasetBase):
    def __init__(self, root, task, n_class, img_suffix='.png', img_transform=Compose([]),\
        label_transform=Compose([]), co_transform=Compose([]), seed=0, phase='train'):
        # print("DatasetClassification: ", root)
        DatasetBase.__init__(self, root, task, n_class,img_transform, label_transform,
            co_transform, seed, phase)
        self.img_suffix = img_suffix
        if self.phase == 'train':
            self.img_list = self.read_img_list(self.train_list)
        elif self.phase == 'test':
            self.img_list = self.read_img_list(self.test_list)
    
    def __getitem__(self, index):
        filename = self.img_list[index].strip()
        with open(os.path.join(self.img_dir, filename + self.img_suffix), 'rb') as f:
            image = Image.open(f).convert('RGB')
        input_size = image.size
        with open(os.path.join(self.label_dir, filename+'.bmp'), 'rb') as f:
            label = Image.open(f).convert('L')
            label = label.resize(input_size)

        image, label = self.co_transform((image, label))
        image = self.img_transform(image)
        label = self.label_transform(label)
        return np.array(image), np.array(label), filename

    def __len__(self):
        return len(self.img_list)

class ToTensorCLLabel(object):
    """
        Take a Classification Label and convert to Tensor
    """
    def __init__(self,tensor_type=torch.long):
        self.tensor_type = tensor_type

    def __call__(self,label):
        label = torch.tensor(label, dtype=self.tensor_type)
        return label

def build_classification_aug(img_size, means, stds, imgaugmentation=True,
              flip=True, rotation=True, zoom=True):
    imgtr, testtr, labeltr, cotr = [], [], [], []
    imgtr   = [transforms.Resize((img_size))]
    testtr  = [transforms.Resize((img_size))]
    labeltr = [ToTensorCLLabel()]

    if imgaugmentation == True:
        if flip == "True":
            imgtr.append(transforms.RandomHorizontalFlip())

        if rotation == "True":
            degree = random.randint(a=0, b=90)
            imgtr.append(transforms.RandomRotation(degree))

        if zoom == "True":
            padding = random.randint(a=0, b=30)
            imgtr.append(transforms.Pad(padding=padding))
    imgtr.append( transforms.Resize((img_size)) ) 
    imgtr.append( transforms.ToTensor())  
    imgtr.append( transforms.Normalize(means, stds) )

    testtr.append( transforms.Resize((img_size)) ) 
    testtr.append(transforms.ToTensor()) 
    testtr.append( transforms.Normalize(means, stds) )
    return imgtr, testtr, labeltr, cotr

def build_segmentation_aug(input_size, means, stds, imgaugmentation=True,
              flip=True, cropresize=True):
    imgtr, testtr, labeltr, cotr = [], [], [], []
    print("means: ", means, ", stds: ", stds)
    imgtr       = [transforms.ToTensor(), SegNormalizeOwn(means, stds)]
    testtr      = [transforms.ToTensor(), SegNormalizeOwn(means, stds)]
    labeltr     = [SegToTensorLabel()]
    cotr        = [SegResizedImage(size=input_size)]
    
    if imgaugmentation == True:
        if flip == "True":
            cotr.append(SegImageFlip())
        if cropresize == "True":
            cotr.append(SegRandomSizedCrop(input_size))

    return imgtr, testtr, labeltr, cotr

def build_dataset(task, params):
    if task == 'classification':
        return DatasetClassification(
            root = params['root'],
            task = params['task'],
            n_class = params['n_class'],
            img_suffix = params['img_suffix'],
            img_transform = Compose(params['imgtr']),
            label_transform = Compose(params['labeltr']),
            co_transform = Compose(params['cotr']),
            phase = params['phase']
        )
    elif task == 'segmentation':
        return DatasetSegmentation(
            root = params['root'],
            task = params['task'],
            n_class = params['n_class'],
            img_suffix = params['img_suffix'],
            img_transform = Compose(params['imgtr']),
            label_transform = Compose(params['labeltr']),
            co_transform = Compose(params['cotr']),
            phase = params['phase']
        )
    else:
        NotImplemented
