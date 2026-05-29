import csv
import os.path as osp
import random

import config
import os
import argparse

from utils import read_csv_file
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset, dataset, Subset
from pretrain_model_cifar15 import get_classwise_ds
from utils import *
import datasets
import models
import config

def view_mia_results(lognames, mia_path):
    mia_results = []
    for logname in lognames:
        logname = osp.join(mia_path, logname)
        mia_results.append(read_csv_file(logname))

    indic_index = 2
    plt.figure(figsize=(5.1, 4.5))
    colors = ['#619CD9', '#9A9AF8', '#F19E9C', '#EF7F51', '#78D3AC', '#9355B0']

    trajectory = mia_results[0][:, indic_index]*100
    
    plt.plot(trajectory, '-', marker='.', label='forget_dataset', color=colors[0])
    
    for i in range(len(mia_results) - 1):
        trajectory = mia_results[i + 1][:, indic_index]*100.
        window_size = 5
        trajectory = np.convolve(trajectory, np.ones(window_size) / window_size, mode='valid')

        plt.plot(trajectory, '-', marker='.',
                 label='ood_dataset' + str(i + 1), color=colors[i+1])

    plt.ylabel('training acc/%')
    plt.xlabel('training steps')
    plt.ylim([0, 100])
    plt.legend(loc=4)
    plt.savefig(mia_path + "/mia_mu_relearning.pdf", bbox_inches='tight', format='pdf')
    plt.cla()
    plt.close("all")

def build_retain_sets_in_unlearning(classwise_train, classwise_test, num_classes, forget_class, ood_class):
    # Getting the retain validation data
    all_class = list(range(0, num_classes))
    retain_class = list(set(all_class) - set(ood_class))

    retain_valid = []
    retain_train = []

    assert forget_class in retain_class
    index_of_forget_class = retain_class.index(forget_class)

    for ordered_cls, cls in enumerate(retain_class):
        if ordered_cls !=index_of_forget_class:
            for img, label, clabel in classwise_test[cls]:  # label and coarse label
                retain_valid.append((img, label, ordered_cls))  # ordered_clss

            for img, label, clabel in classwise_train[cls]:
                retain_train.append((img, label, ordered_cls))

    return (retain_train, retain_valid)

if __name__ == '__main__':
    """
    Get Args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", type=str, default='ResNet18', help="net type")
    parser.add_argument(
        "-weight_path",
        type=str,
        default=".\log_files\model\pretrain\ResNet18-Cifar20-15\35-best.pth",
        help="Path to model weights. If you need to train a new model use pretrain_model.py",
    )
    parser.add_argument(
        "-dataset",
        type=str,
        default='Cifar20',
        nargs="?",
        choices=["Cifar19", "Cifar10", "Cifar20", "Cifar100", "PinsFaceRecognition"],
        help="dataset to train on",
    )
    parser.add_argument("-classes", type=int, default=15, help="number of classes")
    parser.add_argument("-gpu", action="store_true", default=True, help="use gpu or not")
    parser.add_argument("-b", type=int, default=1, help="batch size for dataloader")  # 128,32
    parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
    parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
    parser.add_argument(
        "-method",
        type=str,
        default="finetune",
        nargs="?",
        # choices=[
        #     "baseline",
        #     "retrain",
        #     "finetune",
        #     "blindspot",
        #     "amnesiac",
        #     "FisherForgetting",
        #     'Wfisher',
        #     'FT_prune',
        #     'negative_grad',
        #     'ssd_tuning',
        #     'UNSIR'
        # ],
        help="select unlearning method from choice set",
    )  # not to use: "UNSIR", "ssd_tuning"?

    parser.add_argument(
        "--forget_class",
        type=int,
        default=0,  # 4
        nargs="?",
        help="class to forget",
    )

    parser.add_argument(
        "-mia_mu_method",
        type=str,
        default="mia_mu_relearning",
        nargs="?",
        # choices=[
        #     "mia_mu_relearning",
        #     "mia_mu_adversarial"
        # ],
        help="select unlearning method from choice set",
    )  # not to use: "UNSIR", "ssd_tuning"

    parser.add_argument(
        "-forget_perc", type=float, default=0.1, help="Percentage of trainset to forget"
    )

    parser.add_argument(
        "-epochs", type=int, default=1, help="number of epochs of unlearning method to use"
    )
    parser.add_argument("-seed", type=int, default=0, help="seed for runs")
    parser.add_argument("--unlearn_data_percent", type=str, default="0.05")
    parser.add_argument("--num_ood_dataset", type=str, default='5')

    args = parser.parse_args()

    batch_size = args.b
    root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"
    img_size = 224 if args.net == "ViT" else 32

    trainset = getattr(datasets, args.dataset)(root=root, download=True, train=True, unlearning=True, img_size=img_size)
    validset = getattr(datasets, args.dataset)(root=root, download=True, train=False, unlearning=True,
                                               img_size=img_size)

    trainloader = DataLoader(trainset, batch_size=args.b, shuffle=True)
    validloader = DataLoader(validset, batch_size=args.b, shuffle=False)
    #
    classwise_train, classwise_test = get_classwise_ds(trainset, num_classes=20), \
        get_classwise_ds(validset, num_classes=20)
    # (retain_train, retain_valid) = build_retain_sets_in_unlearning(classwise_train, classwise_test, 20, int(args.forget_class),
    #                                                  config.ood_classes)
    #
    # forget_train, forget_valid = classwise_train[int(args.forget_class)], classwise_test[int(args.forget_class)]
    #
    # forget_valid_dl = DataLoader(forget_valid, batch_size)
    # retain_valid_dl = DataLoader(retain_valid, batch_size)
    # forget_train_dl = DataLoader(forget_train, batch_size)
    # retain_train_dl = DataLoader(retain_train, batch_size, shuffle=True)
    # full_train_dl = DataLoader(ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
    #                            batch_size=batch_size, )
    # full_valid_dl = DataLoader(ConcatDataset((retain_valid_dl.dataset, forget_valid_dl.dataset)),
    #                            batch_size=batch_size, )
    #
    ood_valid_ds = {}
    ood_train_ds = {}
    ood_valid_dl = []
    ood_train_dl = []  # 存不同类别的分布外数据的dataloader
    for cls in config.ood_classes:
        ood_valid_ds[cls] = []
        ood_train_ds[cls] = []

        for img, label, clabel in classwise_test[cls]:
            ood_valid_ds[cls].append((img, label, int(args.forget_class)))

        for img, label, clabel in classwise_train[cls]:
            ood_train_ds[cls].append((img, label, int(args.forget_class)))

        ood_valid_dl.append(DataLoader(ood_valid_ds[cls], batch_size=batch_size, shuffle=True))
        ood_train_dl.append(DataLoader(ood_train_ds[cls], batch_size=batch_size, shuffle=True))
    #
    # data_length = len(forget_train)
    # retain_index_1 = np.random.choice(range(len(retain_train)), size=data_length, replace=False)
    # retain_index_2 = list(set(range(len(retain_train))) - set(retain_index_1))
    # retrain_train_1 = Subset(retain_train, retain_index_1)
    # retrain_train_2 = Subset(retain_train, retain_index_2)
    # retain_train_dl = DataLoader(retrain_train_1, num_workers=0, batch_size=batch_size, shuffle=True)
    # rest_retain_train_dl = DataLoader(retrain_train_2, num_workers=0, batch_size=14, shuffle=True)
    #
    # full_train_dataloader = DataLoader(ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
    #                                    batch_size=batch_size)
    # valid_poisonedloader = ood_valid_dl[0]

    mia_path = os.path.join(config.CHECKPOINT_PATH,
                            "{unlearning_scenarios}".format(unlearning_scenarios="forget_full_class_main"),
                            "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset,
                                                               classes=args.classes),
                            "{task}".format(task="mia_mu_attack"),
                            "{mia_method}".format(mia_method=args.mia_mu_method),
                            "{unlearning_method}-{unlearn_data_percent}".format(unlearning_method=args.method,
                                                                                unlearn_data_percent=args.unlearn_data_percent))


    if isinstance(ood_train_dl, list):
        lognames = ['log_forget_train_dataloader.tsv']
        for i in range(len(ood_train_dl)):
            lognames.append('log_ood_dataloader_'+str(i)+'.tsv')
    else:
        lognames = ['log_forget_train_dataloader.tsv', 'log_ood_dataloader.tsv']
    if not os.path.exists(mia_path):
        os.makedirs(mia_path)

    view_mia_results(lognames, mia_path)
