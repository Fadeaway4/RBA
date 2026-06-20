#!/bin/python3.8

"""
This file is used to collect all arguments for the experiment, prepare the dataloaders, call the method for forgetting, and gather/log the metrics.
Methods are executed in the strategies file.
"""

import random
import os
from typing import Tuple, List
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, dataset
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import models.models_factory as models_factory
from config import ood_classes
from unlearn import *
from utils import *
import forget_full_class_strategies
import datasets
import config
from utils.training_utils import *


@torch.no_grad()
def eval_training(net,testloader):
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for images, _, labels in testloader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)

        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    return correct.float() / len(testloader.dataset)

"""
Get Args
"""
parser = argparse.ArgumentParser()
parser.add_argument("-net", type=str, default='ResNet18', help="net type")
parser.add_argument(
    "-weight_path",
    type=str,
    default="./log_files/model/pretrain/resnet18-Cifar20-20/best.pth",
    help="Path to model weights. If you need to train a new model use pretrain_model.py",
)
parser.add_argument(
    "-dataset",
    type=str,
    default="Cifar20",
    nargs="?",
    choices=["Cifar10", "Cifar20", "Cifar100", "PinsFaceRecognition"],
    help="dataset to train on",
)
parser.add_argument("-classes", type=int,  help="number of classes")
parser.add_argument("-num_classes", type=int,  help="number of classes")
parser.add_argument("-gpu", default=True, help="use gpu or not")
parser.add_argument("-b", type=int, default=64, help="batch size for dataloader")
parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
parser.add_argument(
    "-method",
    type=str,
    default="baseline",
    help="select unlearning method from choice set",
)

parser.add_argument(
    "-mia_mu_method",
    type=str,
    default="mia_mu_relearning",
    nargs="?",
    choices=[
        "mia_mu_relearning",
    ],
    help="select unlearning method from choice set",
)

parser.add_argument(
    "--forget_class",
    type=int,
    default=0,  # 4
    nargs="?",
    help="class to forget",
)
parser.add_argument(
    "-epochs", type=int, default=1, help="number of epochs of unlearning method to use"
)
parser.add_argument(
    "--para1", type=str, default=0.01, help="the first parameters, lr, etc."
)
parser.add_argument(
    "--para2", type=str, default=0, help="number of epochs"
)
parser.add_argument("--strategy", type=str, default=None, help="optimal strategies for unlearning")

parser.add_argument("-seed", type=int, default=0, help="seed for runs")
parser.add_argument("--mask_path", type=str, default=None, help="mask path")





args = parser.parse_args()

# Set seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


# Check that the correct things were loaded
# if args.dataset == "Cifar20":
#     assert args.forget_class in config.cifar20_classes
# elif args.dataset == "Cifar100":
#     assert args.forget_class in config.cifar100_classes

batch_size = args.b

# get network

net = getattr(models_factory, args.net)(num_classes=args.num_classes)


state_dict = torch.load(args.weight_path, map_location='cuda')  


net = getattr(models_factory, args.net)(num_classes=args.num_classes)


state_dict = torch.load(args.weight_path, map_location='cuda')  

from collections import OrderedDict
new_state_dict = OrderedDict()


if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    for k, v in state_dict.items():
        new_key = 'module.' + k  
        new_state_dict[new_key] = v
    net = nn.DataParallel(net)
else:
    new_state_dict = state_dict  


net.load_state_dict(new_state_dict)


if args.gpu:
    net = net.cuda()

checkpoint_path = os.path.join(config.CHECKPOINT_PATH,
                               "{unlearning_scenarios}".format(unlearning_scenarios="forget_full_class_main"),
                               "{net}-{dataset}-{classes}".format(net=args.net,dataset=args.dataset,classes=args.classes),
                               "{task}".format(task="unlearning"),
                               "{unlearning_method}_{para1}_{para2}".format(unlearning_method=args.method,
                                                                            para1=args.para1,
                                                                            para2=args.para2))

print("#####", checkpoint_path)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
weights_path = os.path.join(checkpoint_path, "{epoch}-{type}.pth").format(epoch=args.forget_class, type="class_var")

# for bad teacher
unlearning_teacher = getattr(models_factory, args.net)(num_classes=args.classes)

if args.gpu:
    net = net.cuda()
    unlearning_teacher = unlearning_teacher.cuda()

# For celebritiy faces
root = "/home/libingyan/备用/Libingyan/OUR/OUR-main/data"
# Scale for ViT (faster training, better performance)
# img_size = 224 if args.net == "ViT" else 32
img_size = 32

trainset = getattr(datasets, args.dataset)(root=root, download=True, train=True, unlearning=True, img_size=img_size)
validset = getattr(datasets, args.dataset)(root=root, download=True, train=False, unlearning=True, img_size=img_size)
trainloader = DataLoader(trainset, batch_size=args.b, shuffle=True)
validloader = DataLoader(validset, batch_size=args.b, shuffle=False)
classwise_train, classwise_test = get_classwise_ds(trainset, num_classes=args.num_classes), \
                                  get_classwise_ds(validset, num_classes=args.num_classes)

if args.dataset == 'Cifar10':
    ood_classes = config.cifar10_ood_classes
else:
    ood_classes = config.ood_classes
(retain_train, retain_valid) = build_retain_sets_in_unlearning(classwise_train, classwise_test, args.num_classes,
                                                               int(args.forget_class), ood_classes)

forget_train, forget_valid = classwise_train[int(args.forget_class)], classwise_test[int(args.forget_class)]

forget_set_idxs = np.where(np.array(trainset.targets) == int(args.forget_class))[0]

if args.method == "rum":
    npz_path = "Cifar10_curvature.npz"
    forget_memorization = construct_forget_memorization(forget_set_idxs, npz_path)

forget_valid_dl = DataLoader(forget_valid, batch_size)
retain_valid_dl = DataLoader(retain_valid, batch_size)
forget_train_dl = DataLoader(forget_train, batch_size)
retain_train_dl = DataLoader(retain_train, batch_size, shuffle=True)
full_train_dl = DataLoader(ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)), batch_size=batch_size,)
full_valid_dl = DataLoader(ConcatDataset((retain_valid_dl.dataset, forget_valid_dl.dataset)), batch_size=batch_size,)

ood_valid_ds = {}
ood_train_ds = {}
ood_valid_dl = []
ood_train_dl = []
for cls in ood_classes:
    ood_valid_ds[cls] = []
    ood_train_ds[cls] = []

    for img, label, clabel in classwise_test[cls]:
        ood_valid_ds[cls].append((img, label, int(args.forget_class)))  # 分布外的数据默认为遗忘类

    for img, label, clabel in classwise_train[cls]:
        ood_train_ds[cls].append((img, label, int(args.forget_class)))  # 分布外的数据默认为遗忘类

    ood_valid_dl.append(DataLoader(ood_valid_ds[cls], batch_size))
    ood_train_dl.append(DataLoader(ood_train_ds[cls], batch_size))

# Change alpha here as described in the paper
# For PinsFaceRe-cognition, we use α=50 and λ=0.1
model_size_scaler = 1
if args.net == "ViT":
    model_size_scaler = 0.5
else:
    model_size_scaler = 1

kwargs = {
    "model": net,
    "unlearning_teacher": unlearning_teacher,
    "retain_train_dl": retain_train_dl,
    "retain_valid_dl": retain_valid_dl,
    "forget_train_dl": forget_train_dl,
    "forget_valid_dl": forget_valid_dl,
    "full_train_dl": full_train_dl,
    "valid_dl": full_valid_dl,  # validloader
    "dampening_constant": 1,
    "selection_weighting": 10 * model_size_scaler,
    "forget_class": int(args.forget_class),
    "num_classes": args.classes,
    "dataset_name": args.dataset,
    "device": "cuda" if args.gpu else "cpu",
    "weights_path": weights_path,
    "strategy": args.strategy,
    "mask_path": args.mask_path,
    'para1': args.para1,
    'para2': args.para2,
    "model_name": args.net,
}

if args.method == "rum":
    kwargs.update({"forget_dataset": list(forget_train),
                   "forget_memorization": forget_memorization})

# d_t, d_f, d_r = eval_training(net,validloader),eval_training(net,forget_train_dl),eval_training(net,retain_train_dl)

# Time the method
import time

# executes the method passed via args
(d_t, d_f, d_r), time_elapsed = getattr(forget_full_class_strategies, args.method)(**kwargs)

print("d_t = ", d_t,  "| d_f = ", d_f, "| d_r = ", d_r, "| time = ", time_elapsed)

logname = os.path.join(checkpoint_path, 'log.tsv')
with open(logname, 'w+') as f:
    columns = ['d_t',
               'd_f',
               'd_r',
               'time'
               ]
    f.write('\t'.join(columns) + '\n')

with open(logname, 'a') as f:
    columns = [f"{d_t}",
               f"{d_f}",
               f"{d_r}",
               f"{time_elapsed}"
               ]
    f.write('\t'.join(columns) + '\n')

