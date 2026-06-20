#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
import time
import torch
from torch.utils.data import DataLoader
import numpy as np

from models import models_factory
from datasets import get_dataset
from utils.metrics import get_membership_attack_prob_strategy_b
from utils import get_classwise_ds


DATA_DIR = "/home/libingyan/备用/Libingyan/OUR/OUR-main/data"
DATASET_NAME = "cifar10"


ROOT_DIR = "/home/libingyan/备用/RBA/RBA-main/log_files/model/forget_full_class_main/resnet18-Cifar10-10"


OUTPUT_LOG = os.path.join(ROOT_DIR, "mia_all_models.tsv")

NUM_CLASSES = 10
CLASSES = 10
FORGET_CLASS = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32



def clean_state_dict(state_dict):
    
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return {k.replace("module.", ""): v for k, v in state_dict.items()}



def guess_and_match_model(full_path):
    
    path_lower = full_path.lower()
    
    if "resnet50" in path_lower:
        arch = "resnet50"
    elif "resnet34" in path_lower:
        arch = "resnet34"
    elif "resnet18" in path_lower:
        arch = "resnet18"
    elif "vgg" in path_lower:
        arch = "vgg16"
    elif "vit" in path_lower:
        arch = "ViT"
    else:
        arch = "resnet18"
        
        
    
    return getattr(models_factory, arch)(num_classes=NUM_CLASSES)



from torch.utils.data import random_split

def prepare_dataloaders(forget_class):
    data = get_dataset(DATASET_NAME, DATA_DIR)
    trainset = data[0]

    classwise_train = get_classwise_ds(trainset, NUM_CLASSES)
    forget_full = classwise_train[forget_class]

    
    train_size = int(0.5 * len(forget_full))
    test_size = len(forget_full) - train_size

    forget_train, forget_test = random_split(
        forget_full,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    forget_train_dl = DataLoader(forget_train, batch_size=BATCH_SIZE, shuffle=True)
    forget_test_dl = DataLoader(forget_test, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[DATA] Forget Train: {len(forget_train)} | Forget Test: {len(forget_test)}")
    return forget_train_dl, forget_test_dl



def main():
    os.makedirs(os.path.dirname(OUTPUT_LOG), exist_ok=True)

    
    unlearning_dir = os.path.join(ROOT_DIR, "unlearning")
    if os.path.exists(unlearning_dir) and os.path.isdir(unlearning_dir):
        print(f"[INFO] The unlearning directory is detected...")
        scan_target = unlearning_dir
    else:
        print(f"[WARN] The unlearning directory is not detected...")
        scan_target = ROOT_DIR

    model_dirs = [
        os.path.join(scan_target, d)
        for d in os.listdir(scan_target)
        if os.path.isdir(os.path.join(scan_target, d))
    ]
    
    
    model_dirs.sort()

    
    if not os.path.exists(OUTPUT_LOG):
        with open(OUTPUT_LOG, "w") as f:
            f.write("model\tmia_acc\tmia_auc\ttime\n")

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)  
        
        
        possible_weights = ["4-class_var.pth", "0-class_var.pth", "checkpoint.pth", "best.pth"]
        weight_path = None
        for pw in possible_weights:
            tp = os.path.join(model_dir, pw)
            if os.path.exists(tp):
                weight_path = tp
                break

        if not weight_path:
            
            continue

        print(f"\n[RUN]  {model_name}")

        
        try:
            
            net = guess_and_match_model(weight_path).to(DEVICE)

            state_dict = torch.load(weight_path, map_location=DEVICE)
            state_dict = clean_state_dict(state_dict)

            net.load_state_dict(state_dict, strict=False)
            net.eval()

        except Exception as e:
            print(f"[ERROR] {model_name} Failed to load model instance or weights:: {e}")
            continue

        
        try:
            forget_train_dl, forget_test_dl = prepare_dataloaders(FORGET_CLASS)
        except Exception as e:
            print(f"[ERROR] Failed to load the dataset: {e}")
            continue

        
        start_time = time.time()
        try:
            mia_acc, mia_auc = get_membership_attack_prob_strategy_b(
                forget_train_loader=forget_train_dl,
                forget_test_loader=forget_test_dl,
                model=net
            )

            elapsed = time.time() - start_time

            
            with open(OUTPUT_LOG, "a") as f:
                f.write(f"{model_name}\t{mia_acc:.4f}\t{mia_auc:.4f}\t{elapsed:.2f}\n")

            print(f"[✓] {model_name} | MIA Acc={mia_acc:.4f} | MIA AUC={mia_auc:.4f} | 耗时={elapsed:.2f}s")

        except Exception as e:
            print(f"[ERROR] {model_name} MIA calculation failed: {e}")


if __name__ == "__main__":
    main()