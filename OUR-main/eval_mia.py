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

# -----------------------------
# 配置
# -----------------------------
DATA_DIR = "/mnt/data/Libingyan/OUR-master/OUR-main/data"
DATASET_NAME = "Cifar10"
ROOT_DIR = "/mnt/data/Libingyan/OUR-master/OUR-main/survey/survey_table_10"
#ROOT_DIR = "/mnt/data/Libingyan/OUR-master/OUR-main/survey/survey_table/retrain_0.1_150"

OUTPUT_LOG = os.path.join(ROOT_DIR, "mia_all_models.tsv")

NUM_CLASSES = 10
FORGET_CLASS = 0
OOD_CLASSES = [8,9]  # OOD 类，只用于测试

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32


# -----------------------------
# 去 module.
# -----------------------------
def clean_state_dict(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


# -----------------------------
# DataLoader 构造（关键修复）
# -----------------------------
from torch.utils.data import random_split

def prepare_dataloaders(forget_class):

    data = get_dataset(DATASET_NAME, DATA_DIR)
    trainset = data[0]

    classwise_train = get_classwise_ds(trainset, NUM_CLASSES)

    forget_full = classwise_train[forget_class]

    # 🔥 关键：同一分布拆分
    train_size = int(0.5 * len(forget_full))
    test_size = len(forget_full) - train_size

    forget_train, forget_test = random_split(
        forget_full,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    forget_train_dl = DataLoader(forget_train, batch_size=BATCH_SIZE, shuffle=True)
    forget_test_dl = DataLoader(forget_test, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[FIXED] Forget Train: {len(forget_train)} | Forget Test: {len(forget_test)}")

    return forget_train_dl, forget_test_dl


# -----------------------------
# 主函数
# -----------------------------
def main():

    model_dirs = [
        os.path.join(ROOT_DIR, d)
        for d in os.listdir(ROOT_DIR)
        if os.path.isdir(os.path.join(ROOT_DIR, d))
    ]

    with open(OUTPUT_LOG, "w") as f:
        f.write("model\tmia_acc\tmia_auc\ttime\n")

    for model_dir in model_dirs:

        model_name = os.path.basename(model_dir)
        weight_path = os.path.join(model_dir, "0-class_var.pth")

        if not os.path.exists(weight_path):
            print(f"[WARN] {weight_path} 不存在")
            continue

        print(f"\n[INFO] 处理模型: {model_name}")

        # -----------------------------
        # 加载模型
        # -----------------------------
        try:
            net = getattr(models_factory, "resnet18")(num_classes=NUM_CLASSES).to(DEVICE)

            state_dict = torch.load(weight_path, map_location=DEVICE)
            state_dict = clean_state_dict(state_dict)

            net.load_state_dict(state_dict)
            net.eval()

        except Exception as e:
            print(f"[ERROR] {model_name} 加载失败: {e}")
            continue

        # -----------------------------
        # 数据
        # -----------------------------
        forget_train_dl, forget_test_dl = prepare_dataloaders(FORGET_CLASS)

        # -----------------------------
        # MIA 测量（Strategy B）
        # -----------------------------
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

            print(f"[✓] {model_name} | Acc={mia_acc:.4f} | AUC={mia_auc:.4f} | {elapsed:.2f}s")

        except Exception as e:
            print(f"[ERROR] {model_name} MIA失败: {e}")


if __name__ == "__main__":
    main()