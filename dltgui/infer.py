from pathlib import WindowsPath
import numpy as np
import os
import os.path as osp

from .models import vgg
from .models.deeplabv3 import deeplabv3

import torch
import torch.nn as nn
from torchvision import transforms

from .utils.img_to_json import *
from .utils.vis import *
from PIL import Image


def get_activation(name, model, input, activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def infer_classification(noc, img_path, model_path, input_size,
                 class_dict, means, stds, model_names, feature_name=''):
    img_list = []
    print("img_path: ", img_path)
    if osp.isdir(img_path):
        img_list = os.listdir(img_path)
        for i in range(len(img_list)):
            img_list[i] = osp.join(img_path, img_list[i])
    elif osp.exists(img_path):
        img_list.append(img_path)
    else:
        return []

    class_ind = {}
    print("class_dict: ", class_dict)
    if osp.exists(class_dict):
        with open(class_dict) as f:
            lines = f.readlines()
            num_lines = len(lines)
            for i in range(num_lines):
                line = lines[i]
                if i == 0:
                    key, val = line.split(", ")
                    class_ind[key] = val.strip()
                else:
                    key, val = line.split(", ")
                    class_ind[val.strip()] = key 
    else:
        return []
    print(class_ind)

    pred_labels = []
    img_inputs = []
    print("model_path: ", model_path)
    if osp.exists(model_path):
        snapshot = torch.load(model_path)
    else:
        return []

    if model_names[0] == "VGG-11-bn":
        model = vgg.vgg11_bn(noc=noc, pretrained=True)
    else:
        return []

    print('Snapshot Loaded')
    model.load_state_dict(snapshot['state_dict'])

    model.eval().cuda()
    preprocess = transforms.Compose([transforms.Resize(input_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize(means, stds)])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for img_name in img_list:
            if img_name.endswith(".jpg") or img_name.endswith('.png'):
                img = Image.open(img_name).convert("RGB")
                img_inputs.append(img)
                img_in = preprocess(img).unsqueeze(0).to(device)
                outputs = model(img_in)
                _, pred = torch.max(outputs, 1)
                print(img_name, ", ", class_ind[str(pred.detach().cpu().numpy()[0])])
                pred_labels.append(class_ind[str(pred.detach().cpu().numpy()[0])])


    return img_inputs, pred_labels

def infer_segmentation(noc, img_path, model_path, input_size,
                 class_dict, means, stds, model_names, feature_name=''):
    img_list = []
    print("img_path: ", img_path)
    if osp.isdir(img_path):
        img_list = os.listdir(img_path)
        for i in range(len(img_list)):
            img_list[i] = osp.join(img_path, img_list[i])
    elif osp.exists(img_path):
        img_list.append(img_path)
    else:
        return []

    class_ind = {}
    print("class_dict: ", class_dict)
    if osp.exists(class_dict):
        with open(class_dict) as f:
            lines = f.readlines()
            num_lines = len(lines)
            # for line in lines:
            for i in range(num_lines):
                line = lines[i]
                if i == 0:
                    key, val = line.split(", ")
                    class_ind[key] = val.strip()
                else:
                    key, val = line.split(", ")
                    class_ind[val.strip()] = key 
    else:
        return []
    print(class_ind)

    pred_labels = []
    if osp.exists(model_path):
        snapshot = torch.load(model_path)
    else:
        return []

    if model_names[0] == "deeplabv3":
        if model_names[1] == 'resnet':
            model = deeplabv3.ResDeeplab(backbone='resnet', num_classes=noc)
    else:
        return []

    palette = make_palette(noc)
    print('Snapshot Loaded')
    model.load_state_dict(snapshot['state_dict'])

    model.eval().cuda()
    preprocess = transforms.Compose([transforms.Resize(input_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize(means, stds)])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_save_path = osp.join("results/segmentation", model_names[0], model_names[1], class_ind['dataset'], 'imgs')
    if not osp.exists(img_save_path):
        os.makedirs(img_save_path)
    json_save_path = osp.join("results/segmentation", model_names[0], model_names[1], class_ind['dataset'], 'json')
    if not osp.exists(json_save_path):
        os.makedirs(json_save_path)

    pred_results = []
    img_inputs = []
    with torch.no_grad():
        for img_name in img_list:
            if img_name.endswith(".jpg") or img_name.endswith('.png'):
                print("img_name: ", img_name)
                img = Image.open(img_name).convert("RGB")
                img = img.resize(input_size)
                img_inputs.append(img)
                img_array = np.array(img)
                img_in = preprocess(img).unsqueeze(0).to(device)
                outputs = model(img_in)
                soft_preds = nn.Softmax2d()(outputs)
                soft_np = soft_preds.cpu().numpy()[0]
                hard_pred = np.argmax(soft_np, axis=0).astype(np.uint8)
                masked_im = Image.fromarray(vis_seg(img_array, hard_pred, palette))

                name = img_name.split('/')[-1]
                masked_im.save(osp.join(img_save_path, name))
                pred_results.append(masked_im)
                json_mask = img2json(masked_im, osp.join(json_save_path, name[0:-4]+'.json'))
    return img_inputs, pred_results


def test_classification():
    noc             = 2
    model_path      = "/home/yaok/software/dl-gui-pytorch/models/classification/VGG-11-bn.pth.tar"
    img_path        = "/home/yaok/software/dl-gui-pytorch/tests/testset/8475769_3dea463364_m.jpg" # "/home/yaok/software/dl-gui-pytorch/tests/testset"
    input_size      = (224, 224)
    class_dict      = "/home/yaok/software/dl-gui-pytorch/tests/test_class_dict.txt"
    means           = [0.43759197, 0.43983433, 0.31691911]
    stds            = [0.24613124, 0.22621267, 0.23832114]
    model_name      = "VGG-11-bn"
    infer_classification(noc, img_path, model_path, input_size, class_dict, means, stds, model_name)

def test_segmentation():
    noc             = 2
    model_path      = "/home/yaok/software/dl-gui-pytorch/models/segmentation/deeplabv3_resnet_corrosion.pth.tar"
    img_path        = "/home/yaok/software/dl-gui-pytorch/tests/test_corrosion/gk2_ts_exp20_1220_90.png" # "/home/yaok/software/dl-gui-pytorch/tests/testset"
    input_size      = (513, 513)
    class_dict      = "/home/yaok/software/dl-gui-pytorch/tests/test_corrosion/test_class_dict.txt"
    means           = [0.46314361, 0.44030021, 0.36082766]
    stds            = [0.16120381, 0.15899527, 0.14505895]
    model_name      = "deeplabv3,resnet"
    """
    def infer_segmentation(noc, img_path, model_path, input_size,
                     class_dict, means, stds, model_name, feature_name=''):
    """
    infer_segmentation(noc, img_path, model_path, input_size, class_dict, means, stds, model_name)

if __name__ == "__main__":
    test_segmentation()






        

