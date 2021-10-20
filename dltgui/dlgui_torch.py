# # # # # # # # # # # # # # # 
# Author: Mustafa Mert Tunalı
# ---------------------------
# ---------------------------
# Deep Learning Training GUI - Class Page
# ---------------------------
# ---------------------------
# # # # # # # # # # # # # # # 


# Libraries
from multiprocessing.context import BaseContext
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as display
import cv2
from PIL import Image
import subprocess
from multiprocessing import Process
import datetime
import pathlib
import os
cwd = os.getcwd()
os.chdir(cwd)
import os.path as osp
import Augmentor
import time
import random
import pickle

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torchvision.transforms import ToTensor,Compose
from torch.autograd import Variable
from torch.utils.data import DataLoader

# from .dataset import DatasetClassification
from .dataset import *
from .trainer import *
from .infer import *
from torch.utils.tensorboard import SummaryWriter
import torchvision

# makes the random numbers predictable.
np.random.seed(0)

# Tensorboard Function
def startTensorboard(logdir):
    # Start tensorboard with system call
    os.system("tensorboard --logdir {} --host=0.0.0.0".format(logdir))
    return

class dl_gui:
    
    "Version 1.0 This version, allows you to train image classification model easily"
    def __init__(self, task_type, project_name, input_size, dataset, pre_trained_model = 'VGG-11-bn', 
                cpu_gpu='',  number_of_classes = 5, batch_size = 16, epoch = 1, stage='train',
                fine_tune_epochs = 10, learning_rate=0.1, img_suffix='jpg'):

        self.task_type                      = task_type    
        self.project_name                   = project_name
        if stage == 'train':
            self.train_data_dir             = pathlib.Path(dataset)
        # self.split_dataset                  = split_dataset
        self.pre_trained_model              = pre_trained_model
        self.cpu_gpu                        = cpu_gpu
        self.noc                            = number_of_classes
        self.batch_size                     = batch_size
        self.epoch                          = epoch
        self.IMG_HEIGHT, self.IMG_WIDTH     = input_size[0], input_size[1]
        # self.CLASS_NAMES                    = np.array([item.name for item in self.data_dir.glob('*') if item.name != "LICENSE.txt"])
        self.fine_tune_epochs               = fine_tune_epochs
        self.learning_rate                  = learning_rate
        self.means                          = np.array([0., 0., 0.])
        self.stds                           = np.array([0., 0., 0.])
        self.img_suffix                     = "." + img_suffix
        if self.task_type == 'segmentation':
            self.pre_trained_model          = self.pre_trained_model.split(',')
        elif self.task_type == 'classification':
            self.pre_trained_model          = [self.pre_trained_model]
        if stage == 'test':
            self.test_data_dir              = pathlib.Path(dataset)
        if stage == 'train':
            self.dateset_prepare()
         
    def dateset_prepare(self):
        f_list = os.listdir(osp.join(self.train_data_dir, "JPEGImages"))
        num_imgs = 0
        for f_name in f_list:
            if f_name.endswith(self.img_suffix):
                img = cv2.imread(osp.join(self.train_data_dir, 'JPEGImages/{}'.format(f_name)))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                assert img.shape[-1] == 3, "input image should have 3 channels!"
                for i in range(3):
                    self.means[i] += img[:,:,i].mean()
                    self.stds[i]  += img[:,:,i].std()
                num_imgs += 1
        if num_imgs > 0:
            self.means = np.asarray(self.means) / (num_imgs * 255.)
            self.stds  = np.asarray(self.stds) / (num_imgs * 255.)

        mean_std_path = osp.join(self.train_data_dir, "ImageSets/mean_std_value.pkl")
        data = np.zeros((2, 3))
        data[0] = self.means
        data[1] = self.stds
        with open(mean_std_path, 'wb') as f:
            pickle.dump(data, f)
  
    def load_dataset(self, n_class=2, imgaugmentation = False, aug_ops = [], task = 'classification'):
        # print(task, self.means, self.stds, imgaugmentation, flip, rotation, zoom)
        if task == 'classification':
            flip, rotation,  zoom = aug_ops[0], aug_ops[1], aug_ops[2]
            imgtr, testtr, labeltr, cotr = build_classification_aug((self.IMG_WIDTH,self.IMG_HEIGHT),
                                                    self.means, self.stds, imgaugmentation,
                                                    flip, rotation, zoom)
        elif task == 'segmentation':
            flip, cropresize = aug_ops[0], aug_ops[1]
            imgtr, testtr, labeltr, cotr = build_segmentation_aug(
                                                    input_size = (self.IMG_WIDTH,self.IMG_HEIGHT),
                                                    means = self.means, 
                                                    stds = self.stds, 
                                                    imgaugmentation = imgaugmentation,
                                                    flip = flip, 
                                                    cropresize = cropresize)

    
        train_params = {}
        train_params['root']        = self.train_data_dir
        train_params['task']        = task
        train_params['n_class']     = n_class
        train_params['img_suffix']  = self.img_suffix
        train_params['imgtr']       = imgtr
        train_params['labeltr']     = labeltr
        train_params['cotr']        = cotr
        train_params['phase']       = 'train'

        train_set = build_dataset(task, train_params)
        self.train_data_gen = DataLoader(train_set, batch_size=self.batch_size,
                                        shuffle = True, num_workers = 0, drop_last = True)
        
        test_params = {}
        test_params['root']        = self.train_data_dir
        test_params['task']        = task
        test_params['n_class']     = n_class
        test_params['img_suffix']  = self.img_suffix
        test_params['imgtr']       = testtr
        test_params['labeltr']     = labeltr
        test_params['cotr']        = cotr
        test_params['phase']       = 'test'

        test_set = build_dataset(task, test_params)
        self.test_data_gen = DataLoader(test_set, batch_size = 1)
        self.STEPS_PER_EPOCH = np.ceil(len(self.train_data_gen) / self.batch_size)
        self.VALID_STEPS_PER_EPOCH = np.ceil(len(self.test_data_gen))


    def predict(self, img, model_dir, class_path, task):
        print("task: ", task)
        if not osp.isdir(img):
            print("img: ", img)
            return
        model_dir = osp.join('models/{}/{}'.format(task,model_dir))
        if not osp.exists(model_dir):
            print("model_dir: ", model_dir)
            return
        class_path = osp.join(self.test_data_dir, '{}'.format(class_path))
        if not osp.exists(class_path):
            print("class_path: ", class_path)
            return
        mead_std_file = osp.join(self.test_data_dir, "mean_std_value.pkl")
        if not osp.exists(mead_std_file):
            print("mead_std_file: ", mead_std_file)
            return
        with open(mead_std_file, 'rb') as f:
            data = pickle.load(f)
        means = data[0]
        stds = data[1]
        if task == 'classification':
            img_inputs, pred_results = infer_classification(noc = self.noc,
                                 img_path = img,
                                 model_path = model_dir,
                                 input_size = (self.IMG_WIDTH, self.IMG_HEIGHT),
                                 class_dict = class_path,
                                 means = means,
                                 stds = stds,
                                 model_names = self.pre_trained_model)
            show_heatmap = False
            return img_inputs, pred_results, show_heatmap
        elif task == 'segmentation':
            img_inputs, pred_results = infer_segmentation(noc = self.noc,
                                 img_path = img,
                                 model_path = model_dir,
                                 input_size = (self.IMG_WIDTH, self.IMG_HEIGHT),
                                 class_dict = class_path,
                                 means = means,
                                 stds = stds,
                                 model_names = self.pre_trained_model)
            show_heatmap = True
            return img_inputs, pred_results, show_heatmap
        # return "".join(map(str, classes[y_classes])), max_pred, show_heatmap, heat_map

      
    def train(self, task_type, optim_type, params, loss_type="Cross Entropy"):
        if task_type == "classification":
            train_classification(noc = self.noc,
                                 lr = self.learning_rate, 
                                 pre_trained_model = self.pre_trained_model,
                                 pretrained = True,
                                 cpu_gpu = self.cpu_gpu, 
                                 optim_type = optim_type, 
                                 train_data_gen = self.train_data_gen,
                                 test_data_gen = self.test_data_gen,
                                 save_path = "models/classification/",
                                 epoch_all = self.epoch,
                                 params = params)
            return 
        elif task_type == 'segmentation':
            train_segmentation(noc = self.noc,
                               lr = self.learning_rate, 
                               pre_trained_model = self.pre_trained_model,
                               pretrained = True,
                               loss_type = loss_type,
                               cpu_gpu = self.cpu_gpu, 
                               optim_type = optim_type, 
                               train_data_gen = self.train_data_gen,
                               test_data_gen = self.test_data_gen,
                               save_path = "models/segmentation/",
                               epoch_all = self.epoch,
                               params = params)
            return


    
def test():

    gui = dl_gui(project_name = 'seg_test', 
                 dataset = 'datasets/corrosion',
                 task_type = 'segmentation',
                 input_size = (513,513),
                 pre_trained_model = 'deeplabv3,resnet',
                 cpu_gpu = '/GPU:0',
                 number_of_classes = 2,
                 batch_size = 8,
                 epoch = 100,
                 learning_rate = 1e-3,
                 fine_tune_epochs = 10,
                 img_suffix = 'png',
                 stage = 'train')

    print("data_dir: ", gui.train_data_dir)
    """
    def load_dataset(self, n_class=2, imgaugmentation = False, flip = False, rotation = False,
             zoom = False, task = 'classification'):
    """
    flip, cropresize = 'False', 'False'
    gui.load_dataset(n_class = gui.noc, 
                    imgaugmentation = False, aug_ops = [flip, cropresize], 
                    task = 'segmentation')

    params = {}
    params['dataset'] = 'corrosion'
    params['cur_epoch'] = 0
    print("training start")
    gui.train(task_type='segmentation', optim_type='ADAM', 
            loss_type="Cross Entropy", params = params)

if __name__ == '__main__':
    test()
