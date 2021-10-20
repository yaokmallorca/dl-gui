import vgg
import os
from multiprocessing import Process

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Tensorboard Function
def startTensorboard(logdir):
    # Start tensorboard with system call
    os.system("tensorboard --logdir {} --host=0.0.0.0".format(logdir))
    return

def load_network(pre_trained_model, pretrained=True):
    if pre_trained_model == "VGG-11-bn":
        return vgg.vgg11_bn(pretrained=pretrained)
    else:
        NotImplemented

def classification_train(model, lr, pre_trained_model, pretrained, cpu_gpu,
                         train_data_gen, test_data_gen, save_path, epoch):
    model = load_network(pre_trained_model, pretrained)
    model = vgg.vgg11_bn(pretrained=True)
    loss = nn.CrossEntropyLoss().cuda()
    if cpu_gpu == "/GPU:0":
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    Process(target=startTensorboard, args=("logs/{}".format(pre_trained_model),)).start()
    writer = SummaryWriter("logs/{}".format(pre_trained_model))
    # start training
    for epoch in range(1, epoch+1):
        model.train()
        losses = []
        for batch_idx, (data, target) in enumerate(train_data_gen):
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
        for batch_idx, (data, target) in enumerate(test_data_gen):
            if cpu_gpu == "/GPU:0":
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += loss(output, target).data # sum up batch loss
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
        test_loss /= len(test_data_gen)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_data_gen),
            100. * correct / len(test_data_gen)))
        writer.add_scalar('Test/Loss', test_loss, epoch)
        writer.add_scalar('Test/Acc', 100. * correct / len(test_data_gen), epoch)
