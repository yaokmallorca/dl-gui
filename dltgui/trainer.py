import os
from multiprocessing import Process
from datetime import datetime
import threading
from typing import NamedTuple

from matplotlib.colors import same_color
import matplotlib.pyplot as plt
import numpy as np

# classification networks
from .models import vgg
from .models.deeplabv3 import deeplabv3
from .utils import metrics as seg_metrics

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision

# Tensorboard Function
def startTensorboard(logdir):
    # Start tensorboard with system call
    os.system("tensorboard --logdir {} --host=0.0.0.0".format(logdir))
    return

def load_network(noc, pre_trained_model, pretrained=True):
    if pre_trained_model[0] == "VGG-11-bn":
        return vgg.vgg11_bn(noc=noc, pretrained=pretrained)
    elif pre_trained_model[0] == 'deeplabv3':
        backbone = pre_trained_model[1]
        return deeplabv3.ResDeeplab(backbone=backbone, num_classes=noc)
    else:
        NotImplemented

def show_batches(batch_tensor, target_tensor):
        grid_imgs = torchvision.utils.make_grid(batch_tensor, nrow=5)
        plt.imshow(grid_imgs.permute(1,2,0))
        plt.savefig("batch_imgs.png")
        plt.clf()

        grid_imgs = torchvision.utils.make_grid(target_tensor*100, nrow=5)
        plt.imshow(grid_imgs.permute(1,2,0))
        plt.savefig("batch_labels.png")
        plt.clf()



def train_classification(noc, lr, pre_trained_model, pretrained, cpu_gpu, optim_type,
                         train_data_gen, test_data_gen, save_path, epoch_all, params):
    # load network module
    print(pre_trained_model[0])
    model = load_network(noc, pre_trained_model, pretrained)

    loss = nn.CrossEntropyLoss().cuda()
    best_acc = 0.

    if cpu_gpu == "/GPU:0":
        model.cuda()
    if optim_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optim_type == "ADAM":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, \
            model.parameters()), lr, [0.9, 0.999])
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Process(target=startTensorboard, args=("logs/{}".format(pre_trained_model),)).start()
    now = datetime.now()
    date_time = now.strftime("%Y-%d-%m-%H-%M-%S")
    Process(target=startTensorboard, args=("logs/{}/{}".format(pre_trained_model[0],date_time),)).start()
    writer = SummaryWriter("logs/{}/{}".format(pre_trained_model[0], date_time))
    # start training
    for epoch in range(1, epoch_all+1):
        model.train()
        losses = []
        for _, (data, target) in enumerate(train_data_gen):
            if cpu_gpu == "/GPU:0":
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            out = model(data)
            loss_values = loss(out, target)
            loss_values.backward()
            optimizer.step()
            print("Training epoch {}, lr: {:.6f}, loss: {:.6f}".format(str(epoch), lr, loss_values))
            losses.append(loss_values.detach().cpu().numpy())
            writer.add_scalar('Train/Loss', sum(losses)/len(losses), epoch)

        # validation
        model.eval()
        test_loss, correct = 0, 0
        for _, (data, target) in enumerate(test_data_gen):
            if cpu_gpu == "/GPU:0":
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += loss(output, target).data # sum up batch loss
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
        test_loss /= len(test_data_gen)
        test_acc = 100. * correct / len(test_data_gen)
        if test_acc > best_acc:
            best_acc = test_acc
            # save model weight
            snapshot = {
                'state_dict': model.state_dict(),
            }
            torch.save(snapshot, os.path.join(save_path,'{}_{}.pth.tar'.format(pre_trained_model[0],
                params['dataset'].split('/')[-1])))

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_data_gen),
            100. * correct / len(test_data_gen)))
        writer.add_scalar('Test/Loss', test_loss, epoch)
        writer.add_scalar('Test/Acc', 100. * correct / len(test_data_gen), epoch)
        params['cur_epoch'] = epoch

    
def train_segmentation(noc, lr, pre_trained_model, pretrained, cpu_gpu, optim_type, loss_type,
                         train_data_gen, test_data_gen, save_path, epoch_all, params):
    # load network module
    print("pretrained: ", pre_trained_model)
    model = load_network(noc, pre_trained_model, pretrained)

    print("trainer: ", loss_type)
    if loss_type == "Cross Entropy":
        loss = nn.NLLLoss(ignore_index=255)
        activation = nn.LogSoftmax(dim=1)

    best_miou = 0.
    if cpu_gpu == "/GPU:0":
        model.cuda()
        loss.cuda()
    if optim_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optim_type == "ADAM":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, \
            model.parameters()), lr, [0.9, 0.999])
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    now = datetime.now()
    date_time = now.strftime("%Y-%d-%m-%H-%M-%S")
    Process(target=startTensorboard, args=("logs/segmentation/{}_{}/{}".format(pre_trained_model[0],
            pre_trained_model[1], date_time),)).start()
    writer = SummaryWriter("logs/segmentation/{}_{}/{}".format(pre_trained_model[0], 
            pre_trained_model[1], date_time))
    # start training
    for epoch in range(1, epoch_all+1):
        model.train()
        losses = []
        for _, (data, target, _) in enumerate(train_data_gen):
            if cpu_gpu == "/GPU:0":
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            out = model(data)
            prob = activation(out)
            loss_values = loss(prob, target)
            loss_values.backward()
            optimizer.step()
            print("Training epoch {}, lr: {:.6f}, loss: {:.6f}".format(str(epoch), lr, loss_values))
            losses.append(loss_values.detach().cpu().numpy())
            writer.add_scalar('Train/Loss', sum(losses)/len(losses), epoch)

        # validation
        model.eval()
        test_loss, mious = 0, 0
        for _, (data, target, _) in enumerate(test_data_gen):
            if cpu_gpu == "/GPU:0":
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            prob = activation(output)
            test_loss += loss(prob, target).data # sum up batch loss

            soft_pred = nn.Softmax2d()(output)
            org_soft = soft_pred.data.cpu().numpy()[0]
            org_hard = np.argmax(org_soft, axis=0).astype(np.uint8)

            label_np = target.cpu()[0].numpy()
            miou, cls_iu, acc = seg_metrics.scores(label_np, org_hard, n_class=noc)
            # print("miou: ", miou, " cls_iu: ", cls_iu, " acc: ", acc)
            mious += miou
        test_loss /= len(test_data_gen)
        test_miou =  mious / float(len(test_data_gen))
        if test_miou > best_miou:
            best_miou = test_miou
            # save model weight
            snapshot = {
                'state_dict': model.state_dict(),
            }
            torch.save(snapshot, os.path.join(save_path,'{}_{}_{}.pth.tar'.format(pre_trained_model[0], 
                    pre_trained_model[1], params['dataset'].split('/')[-1])))

        print('\nTest set: Average loss: {:.4f}, miou: ({:.4f})\n'.format(test_loss, test_miou))
        writer.add_scalar('Test/Loss', test_loss, epoch)
        writer.add_scalar('Test/mIOU', test_miou, epoch)
        params['cur_epoch'] = epoch
