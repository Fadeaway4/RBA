""" 
configurations for this project
"""

# Imports here
import os
from datetime import datetime

CHECKPOINT_PATH = "log_files/model"

forget_class = [0]
cifar10_ood_classes = [ 8, 9]
ood_classes = [15, 16, 17, 18, 19]#np.random.choice(np.setdiff1d(np.arange(0, 19), forget_class), 5, replace=False)
# ood_classes_gtsrb = [9, 10, 11] #the first batch
ood_classes_gtsrb = [40, 9, 11]# the second batch

# Class correspondence as done in https://github.com/vikram2000b/bad-teaching-unlearning
class_dict = {
    "rocket": 69,
    "vehicle2": 19,
    "veg": 4,
    "mushroom": 51,
    "people": 14,
    "baby": 2,
    "electrical_devices": 5,
    "lamp": 40,
    "natural_scenes": 10,
    "sea": 71,
    "42": 42,
    "1": 1,
    "10": 10,
    "20": 20,
    "30": 30,
    "40": 40,
}

# Classes from https://github.com/vikram2000b/bad-teaching-unlearning
cifar20_classes = {"vehicle2", "veg", "people", "electrical_devices", "natural_scenes"}

# Classes from https://github.com/vikram2000b/bad-teaching-unlearning
cifar100_classes = {"rocket", "mushroom", "baby", "lamp", "sea"}

# total training epochs

# Training parameters for the tasks; milestones are when the learning rate gets lowered
PinsFaceRecognition_EPOCHS = 200
PinsFaceRecognition_MILESTONES = [60, 120, 160]

Cifar100_EPOCHS = 200
Cifar100_MILESTONES = [60, 120, 160]

Cifar10_EPOCHS = 20
Cifar10_MILESTONES = [8, 12, 16]

Cifar20_EPOCHS = 40
Cifar20_MILESTONES = [15, 30, 35]

Cifar19_EPOCHS = 40
Cifar19_MILESTONES = [15, 30, 35]

Cifar15_EPOCHS = 40
Cifar15_MILESTONES = [15, 30, 35]

Imagenet64_EPOCHS = 200
Imagenet64_MILESTONES = [120, 180, 250]

Cifar100_EPOCHS = 200
Cifar100_MILESTONES = [60, 120, 160]

GTSRB_EPOCHS = 50
GTSRB_MILESTONES = [30, 35]

Cifar10_ViT_EPOCHS = 8
Cifar10_ViT_MILESTONES = [7]

Cifar20_ViT_EPOCHS = 9
Cifar20_ViT_MILESTONES = [8]

Cifar19_ViT_EPOCHS = 9
Cifar19_ViT_MILESTONES = [8]

Cifar100_ViT_EPOCHS = 8
Cifar100_ViT_MILESTONES = [7]

Imagenet64_ViT_EPOCHS = 8
Imagenet64_ViT_MILESTONES = [7]

GTSRB_ViT_EPOCHS = 15
GTSRB_ViT_MILESTONES = [12]

DATE_FORMAT = "%A_%d_%B_%Y_%Hh_%Mm_%Ss"
# time of script run
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

# log dir
LOG_DIR = "runs"

# save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10
