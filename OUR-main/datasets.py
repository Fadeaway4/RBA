"""
Datasets used for the experiments (CIFAR and Celebrity Faces)
"""
import csv
import os
from typing import Any, Tuple
from torchvision.datasets import CIFAR100, CIFAR10, ImageFolder
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

# Improves model performance (https://github.com/weiaicunzai/pytorch-cifar100)
CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# Cropping etc. to improve performance of the model (details see https://github.com/weiaicunzai/pytorch-cifar100)
transform_train_from_scratch = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
]

transform_unlearning = [
    # transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
]

transform_test = [
    # transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
]

transform_gtsrb = transforms.Compose([
    transforms.Resize((32, 32)),#224 64
    transforms.ToTensor(),
])

transform_imagenet = transforms.Compose([
    transforms.Resize((32, 32)),#224 64
    transforms.ToTensor(),
])

# www.kaggle.com/datasets/hereisburak/pins-face-recognition
class PinsFaceRecognition(ImageFolder):
    def __init__(self, root, train, unlearning, download, img_size=32):
        if train:
            if unlearning:
                transform = transform_unlearning
            else:
                transform = transform_train_from_scratch
        else:
            transform = transform_test
        transform.insert(0, transforms.Resize((36, 36)))
        transform.append(transforms.Resize((img_size, img_size)))
        transform = transforms.Compose(transform)
        super().__init__(root, transform)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        x, y = super().__getitem__(index)
        return x, torch.Tensor([]), y


class Cifar100(CIFAR100):
    def __init__(self, root, train, unlearning, download, img_size=32):
        if train:
            if unlearning:
                transform = transform_unlearning
            else:
                transform = transform_train_from_scratch
        else:
            transform = transform_test
        transform.append(transforms.Resize(img_size))
        transform = transforms.Compose(transform)

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, torch.Tensor([]), y


class Cifar20(CIFAR100):
    def __init__(self, root, train, unlearning, download, img_size=32):
        if train:
            if unlearning:
                transform = transform_unlearning
            else:
                transform = transform_train_from_scratch
        else:
            transform = transform_test
        transform.append(transforms.Resize(img_size))
        transform = transforms.Compose(transform)

        super().__init__(root=root, train=train, download=download, transform=transform)

        # This map is for the matching of subclases to the superclasses. E.g., rocket (69) to Vehicle2 (19:)
        # Taken from https://github.com/vikram2000b/bad-teaching-unlearning
        self.coarse_map = {
            0: [4, 30, 55, 72, 95],
            1: [1, 32, 67, 73, 91],
            2: [54, 62, 70, 82, 92],
            3: [9, 10, 16, 28, 61],
            4: [0, 51, 53, 57, 83],
            5: [22, 39, 40, 86, 87],
            6: [5, 20, 25, 84, 94],
            7: [6, 7, 14, 18, 24],
            8: [3, 42, 43, 88, 97],
            9: [12, 17, 37, 68, 76],
            10: [23, 33, 49, 60, 71],
            11: [15, 19, 21, 31, 38],
            12: [34, 63, 64, 66, 75],
            13: [26, 45, 77, 79, 99],
            14: [2, 11, 35, 46, 98],
            15: [27, 29, 44, 78, 93],
            16: [36, 50, 65, 74, 80],
            17: [47, 52, 56, 59, 96],
            18: [8, 13, 48, 58, 90],
            19: [41, 69, 81, 85, 89],
        }

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        coarse_y = None
        for i in range(20):
            for j in self.coarse_map[i]:
                if y == j:
                    coarse_y = i
                    break
            if coarse_y != None:
                break
        if coarse_y == None:
            print(y)
            assert coarse_y != None
        return x, y, coarse_y


class Cifar10(CIFAR10):
    def __init__(self, root, train, unlearning, download, img_size=32):
        if train:
            if unlearning:
                transform = transform_unlearning
            else:
                transform = transform_train_from_scratch
        else:
            transform = transform_test
        transform.append(transforms.Resize(img_size))
        transform = transforms.Compose(transform)

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, torch.Tensor([]), y
        # return x, y


class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x = self.forget_data[index][0]
            y = 1
            return x, y
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = 0
            return x, y

import random
from typing import Callable, Optional
from PIL import Image
class TriggerHandler(object):
    def __init__(self, trigger_path, trigger_size, trigger_label, img_width, img_height):
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))
        self.trigger_label = trigger_label
        self.img_width = img_width
        self.img_height = img_height

    def put_trigger(self, img):
        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))
        return img


class Cifar10Poison(CIFAR10):
    def __init__(self, args, root, train, unlearning, download, img_size=32):
        if train:
            if unlearning:
                transform = transform_unlearning
            else:
                transform = transform_train_from_scratch
        else:
            transform = transform_test
        transform.append(transforms.Resize(img_size))
        transform = transforms.Compose(transform)

        super().__init__(root=root, train=train, download=download, transform=transform)

        self.width, self.height, self.channels = self.__shape_info__()

        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        random.seed(args.seed)
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(self.poi_indices[:10])
        self.remain_indices = np.delete(np.arange(len(self.targets)), self.poi_indices, axis=0)
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, torch.Tensor([]), target

class GTSRB(Dataset):
    def __init__(self, root, train=True):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(root, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(root, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transform_gtsrb

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label


import pickle

class Imagenet64(Dataset):
    def __init__(self, root, train=False):
        self.data = []
        self.labels = []
        self.transform = transform_imagenet
        self.train = train
        self.load_data(root)

    def load_data(self, root):
        # 加载所有批次文件
        if self.train:
            for i in range(1, 11):  # 从1到10
                file_path = f"{root}/Imagenet64_train/train_data_batch_{i}"
                with open(file_path, 'rb') as file:
                    batch_data = pickle.load(file, encoding='bytes')
                # ImageNet64 数据通常是 N x 3072 的数组，需要重新塑形为 64x64x3
                images = batch_data['data'].reshape((-1, 3, 64, 64)).transpose((0, 2, 3, 1))
                self.data.extend(images)
                self.labels.extend(batch_data['labels'])
        else:
            file_path = f"{root}/Imagenet64_val/val_data"
            with open(file_path, 'rb') as file:
                batch_data = pickle.load(file, encoding='bytes')
            # ImageNet64 数据通常是 N x 3072 的数组，需要重新塑形为 64x64x3
            images = batch_data['data'].reshape((-1, 3, 64, 64)).transpose((0, 2, 3, 1))
            self.data.extend(images)
            self.labels.extend(batch_data['labels'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx] - 1  # 标签调整为从0开始
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label
