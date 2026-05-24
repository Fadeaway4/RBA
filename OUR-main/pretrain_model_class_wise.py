#!/usr/bin/env python3
"""
优化版训练脚本，支持从上次中断处继续训练
"""

import logging
import os
import sys
import argparse
import time
import datetime
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import datasets
import models.models_factory as models_factory
import config
from utils.training_utils import WarmUpLR
from utils.overall_utils import eval_training


class CheckpointManager:
    """检查点管理器，负责保存和加载训练状态（兼容单/多GPU）"""

    def __init__(self, save_dir, model_name, net):
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.net = net
        self.checkpoint_path = self.save_dir / "checkpoint.pth"
        self.best_model_path = self.save_dir / "best.pth"
        self.latest_model_path = self.save_dir / "latest.pth"

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, epoch, optimizer, scheduler, warmup_scheduler,
                        best_acc, is_best=False, is_interrupt=False):
        """保存训练检查点"""
        if isinstance(self.net, nn.DataParallel):
            model_to_save = self.net.module  # 去掉 DataParallel
        else:
            model_to_save = self.net

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'warmup_scheduler_step': getattr(warmup_scheduler, 'step_count', 0),
            'best_acc': best_acc,
            'timestamp': datetime.datetime.now().isoformat(),
            'args': args_dict
        }

        torch.save(checkpoint, self.checkpoint_path)
        torch.save(model_to_save.state_dict(), self.latest_model_path)

        if is_best:
            torch.save(model_to_save.state_dict(), self.best_model_path)

        if is_interrupt:
            logging.info(f"[中断保存] 检查点已保存到: {self.checkpoint_path}")
        else:
            logging.info(f"[检查点] epoch {epoch} 已保存到: {self.checkpoint_path}")

    def load_checkpoint(self, optimizer, scheduler, warmup_scheduler):
        """加载训练检查点，兼容 DataParallel / 单 GPU"""
        if not self.checkpoint_path.exists():
            return 1, 0.0, False

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}

            # 自动处理 module 前缀（单/多GPU 都兼容）
            for k, v in state_dict.items():
                k_new = k
                if k.startswith('module.') and not isinstance(self.net, nn.DataParallel):
                    k_new = k[len('module.'):]  # 去掉 module
                elif not k.startswith('module.') and isinstance(self.net, nn.DataParallel):
                    k_new = 'module.' + k  # 补上 module
                new_state_dict[k_new] = v

            self.net.load_state_dict(new_state_dict)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if hasattr(warmup_scheduler, 'step_count'):
                warmup_scheduler.step_count = checkpoint.get('warmup_scheduler_step', 0)

            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']

            logging.info(f"从检查点恢复训练: epoch {checkpoint['epoch']}, 最佳准确率: {best_acc:.4f}")
            logging.info(f"检查点时间: {checkpoint.get('timestamp', '未知')}")

            return start_epoch, best_acc, True

        except Exception as e:
            logging.warning(f"加载检查点失败: {e}")
            return 1, 0.0, False

def setup_logging(output_path):
    """设置日志记录"""
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)

    # 避免重复添加handler
    if not logger.handlers:
        # 文件handler
        file_handler = logging.FileHandler(output_path / "training.log")
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)

        # 控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(console_handler)

    return logger


def create_data_loaders(args):
    """创建数据加载器（支持剔除OOD类别）"""
    from torch.utils.data import Subset

    # =========================
    # 数据增强
    # =========================
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    # =========================
    # 加载数据
    # =========================
    root = "./data"
    trainset = getattr(datasets, args.dataset)(
        root=root, download=True, train=True, unlearning=False, img_size=32
    )
    testset = getattr(datasets, args.dataset)(
        root=root, download=True, train=False, unlearning=False, img_size=32
    )

    # =========================
    # 🔥 OOD 剔除（核心！！）
    # =========================
    def remove_ood_classes(dataset, ood_classes):
        indices = []
        for i in range(len(dataset)):
            _, _, label = dataset[i]
            if label not in ood_classes:
                indices.append(i)
        return Subset(dataset, indices)

    # 👉 这里你可以改
   #ood_classes = [15, 16, 17, 18, 19]
    ood_classes = [0, 8, 9]
    print("====================================")
    print("OOD classes (NOT used in training):", ood_classes)
    print("====================================")

    trainset = remove_ood_classes(trainset, ood_classes)
    testset = remove_ood_classes(testset, ood_classes)

    # =========================
    # sanity check（强烈建议保留）
    # =========================
    labels_check = set()
    for i in range(min(2000, len(trainset))):
        _, _, y = trainset[i]
        labels_check.add(int(y))

    print("训练集中实际类别:", sorted(labels_check))

    # =========================
    # DataLoader
    # =========================
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )

    testloader = DataLoader(
        testset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )

    return trainloader, testloader, len(trainset)


def create_model_and_optimizer(args, num_classes):
    """创建模型和优化器"""
    # 网络
    net = getattr(models_factory, args.net)(num_classes=num_classes)

    # 多GPU支持
    if args.gpu and torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU")
        net = nn.DataParallel(net)

    if args.gpu:
        net = net.cuda()

    # 损失函数
    loss_function = nn.CrossEntropyLoss()
    if args.gpu:
        loss_function = loss_function.cuda()

    # 优化器
    if args.net == "ViT":
        optimizer = optim.AdamW(
            net.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.SGD(
            net.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

    return net, loss_function, optimizer


def train_epoch(net, trainloader, loss_function, optimizer, epoch, warmup_scheduler, args):
    """训练一个epoch"""
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (images, _, labels) in enumerate(trainloader):
        if args.gpu:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()

        # 梯度裁剪
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

        optimizer.step()

        # 更新warmup调度器
        if epoch <= args.warmup_epochs:
            warmup_scheduler.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 进度显示
        if batch_idx % args.log_interval == 0:
            avg_loss = running_loss / (batch_idx + 1)
            acc = 100. * correct / total
            current_lr = optimizer.param_groups[0]['lr']

            print(f'Epoch: {epoch} [{batch_idx}/{len(trainloader)}] '
                  f'Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | LR: {current_lr:.6f}')

    epoch_time = time.time() - start_time
    avg_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total

    return avg_loss, epoch_acc, epoch_time


def main():
    """主函数"""
    # ------------------------
    # 参数解析
    # ------------------------
    parser = argparse.ArgumentParser(description="支持断点续训的训练脚本")

    # 模型和数据参数
    parser.add_argument("-net", type=str, default='ViT', help="网络类型")
    parser.add_argument("-dataset", type=str, default='Cifar20', help="数据集")
    parser.add_argument("-classes", type=int, default=20, help="类别数")

    # 训练参数
    parser.add_argument("-epochs", type=int, default=3000, help="总训练轮数")
    parser.add_argument("-batch_size", "-b", type=int, default=256, help="批大小")
    parser.add_argument("-learning_rate", "-lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("-weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("-momentum", type=float, default=0.9, help="动量")
    parser.add_argument("-grad_clip", type=float, default=5.0, help="梯度裁剪")

    # 调度器参数
    parser.add_argument("-warmup_epochs", type=int, default=5, help="warmup轮数")
    parser.add_argument("-min_lr", type=float, default=1e-6, help="最小学习率")

    # 系统参数
    parser.add_argument("-gpu", type=bool, default=True, help="使用GPU")
    parser.add_argument("-seed", type=int, default=0, help="随机种子")
    parser.add_argument("-resume", action="store_true", help="从检查点恢复训练")
    parser.add_argument("-no_resume", action="store_true", help="强制从头开始训练")
    parser.add_argument("-log_interval", type=int, default=50, help="日志间隔")
    parser.add_argument("-save_interval", type=int, default=10, help="保存间隔")

    args = parser.parse_args()

    # 保存参数到全局变量，用于检查点保存
    global args_dict
    args_dict = vars(args)

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ------------------------
    # 输出路径
    # ------------------------
    output_path = Path(config.CHECKPOINT_PATH) / "pretrain" / f"{args.net}-{args.dataset}-{args.classes}"
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------
    # 设置日志
    # ------------------------
    logger = setup_logging(output_path)

    # 记录参数
    logger.info("训练参数:")
    for key, value in args_dict.items():
        logger.info(f"  {key}: {value}")

    # ------------------------
    # 创建数据加载器
    # ------------------------
    logger.info("创建数据加载器...")
    trainloader, testloader, dataset_size = create_data_loaders(args)
    logger.info(f"训练集大小: {dataset_size}")
    logger.info(f"测试集大小: {len(testloader.dataset)}")

    # ------------------------
    # 创建模型和优化器
    # ------------------------
    logger.info("创建模型和优化器...")
    net, loss_function, optimizer = create_model_and_optimizer(args, args.classes)

    # ------------------------
    # 创建调度器
    # ------------------------
    iter_per_epoch = len(trainloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warmup_epochs)

    # Cosine退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )

    # ------------------------
    # 检查点管理器
    # ------------------------
    checkpoint_manager = CheckpointManager(output_path, f"{args.net}-{args.dataset}", net)

    # ------------------------
    # 恢复训练或从头开始
    # ------------------------
    if args.no_resume:
        start_epoch = 1
        best_acc = 0.0
        logger.info("强制从头开始训练")
    else:
        start_epoch, best_acc, loaded = checkpoint_manager.load_checkpoint(
            optimizer, scheduler, warmup_scheduler
        )
        if not loaded and not args.resume:
            start_epoch = 1
            best_acc = 0.0
            logger.info("从头开始训练")
        elif loaded:
            logger.info(f"从epoch {start_epoch}恢复训练，最佳准确率: {best_acc:.4f}")

    # ------------------------
    # 训练循环
    # ------------------------
    logger.info("开始训练...")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            # 训练一个epoch
            train_loss, train_acc, epoch_time = train_epoch(
                net, trainloader, loss_function, optimizer,
                epoch, warmup_scheduler, args
            )

            # 更新学习率调度器（warmup阶段之后）
            if epoch > args.warmup_epochs:
                scheduler.step()

            # 评估
            test_acc = eval_training(epoch, net, testloader)
            current_lr = optimizer.param_groups[0]['lr']

            # 记录结果
            logger.info(f"Epoch {epoch}/{args.epochs} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Train Acc: {train_acc:.2f}%, "
                        f"Test Acc: {test_acc:.4f}, "
                        f"LR: {current_lr:.6f}, "
                        f"Time: {epoch_time:.2f}s")

            # 检查是否是最佳模型
            is_best = test_acc > best_acc
            if is_best:
                best_acc = test_acc
                logger.info(f"新的最佳准确率: {best_acc:.4f}")

            # 定期保存检查点
            if epoch % args.save_interval == 0 or epoch == args.epochs or is_best:
                checkpoint_manager.save_checkpoint(
                    epoch, optimizer, scheduler, warmup_scheduler,
                    best_acc, is_best=is_best
                )

    except KeyboardInterrupt:
        logger.info("训练被中断，保存当前状态...")
        checkpoint_manager.save_checkpoint(
            epoch - 1 if 'epoch' in locals() else start_epoch - 1,
            optimizer, scheduler, warmup_scheduler,
            best_acc, is_interrupt=True
        )
        logger.info("已保存中断检查点")

    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        logger.error("尝试保存当前状态...")
        checkpoint_manager.save_checkpoint(
            epoch - 1 if 'epoch' in locals() else start_epoch - 1,
            optimizer, scheduler, warmup_scheduler,
            best_acc, is_interrupt=True
        )
        raise

    finally:
        logger.info(f"训练完成，最佳测试准确率: {best_acc:.4f}")
        logger.info(f"模型保存在: {output_path}")


if __name__ == "__main__":
    main()