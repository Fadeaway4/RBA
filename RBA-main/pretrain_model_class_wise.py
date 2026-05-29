#!/usr/bin/env python3
"""
Optimized training script with resume-from-checkpoint support
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

        if isinstance(self.net, nn.DataParallel):
            model_to_save = self.net.module  # Remove DataParallel wrapper
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
            logging.info(f"Checkpoint saved: {self.checkpoint_path}")
        else:
            logging.info(f"[Checkpoint] Epoch {epoch} saved: {self.checkpoint_path}")

    def load_checkpoint(self, optimizer, scheduler, warmup_scheduler):

        if not self.checkpoint_path.exists():
            return 1, 0.0, False

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}

            for k, v in state_dict.items():
                k_new = k
                if k.startswith('module.') and not isinstance(self.net, nn.DataParallel):
                    k_new = k[len('module.'):]
                elif not k.startswith('module.') and isinstance(self.net, nn.DataParallel):
                    k_new = 'module.' + k
                new_state_dict[k_new] = v

            self.net.load_state_dict(new_state_dict)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            if hasattr(warmup_scheduler, 'step_count'):
                warmup_scheduler.step_count = checkpoint.get('warmup_scheduler_step', 0)

            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']

            logging.info(
                f"Resumed training from epoch {checkpoint['epoch']}, "
                f"best Accuracy: {best_acc:.4f}"
            )
            logging.info(
                f"Checkpoint timestamp: {checkpoint.get('timestamp', 'Unknown')}"
            )

            return start_epoch, best_acc, True

        except Exception as e:
            logging.warning(f"Failed to load checkpoint: {e}")
            return 1, 0.0, False


def setup_logging(output_path):
    """Setup logging"""
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.handlers:

        # File handler
        file_handler = logging.FileHandler(output_path / "training.log")
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(console_handler)

    return logger


def create_data_loaders(args):
    """Create data loaders (supports removing OOD classes)"""
    from torch.utils.data import Subset

    # =========================
    # Data augmentation
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
    # Load dataset
    # =========================
    root = "/home/libingyan/备用/Libingyan/OUR/OUR-main/data"

    trainset = getattr(datasets, args.dataset)(
        root=root, download=True, train=True,
        unlearning=False, img_size=32
    )

    testset = getattr(datasets, args.dataset)(
        root=root, download=True, train=False,
        unlearning=False, img_size=32
    )

    # =========================
    # 🔥 Remove OOD classes
    # =========================
    def remove_ood_classes(dataset, ood_classes):
        indices = []

        for i in range(len(dataset)):
            _, _, label = dataset[i]
            if label not in ood_classes:
                indices.append(i)

        return Subset(dataset, indices)

    # 👉 Modify here if needed
    # ood_classes = [15, 16, 17, 18, 19]
    ood_classes = [0, 8, 9]

    print("====================================")
    print("OOD classes (NOT used in training):", ood_classes)
    print("====================================")

    trainset = remove_ood_classes(trainset, ood_classes)
    testset = remove_ood_classes(testset, ood_classes)

    # =========================
    # Sanity check
    # =========================
    labels_check = set()

    for i in range(min(2000, len(trainset))):
        _, _, y = trainset[i]
        labels_check.add(int(y))

    print("Actual classes in training set:", sorted(labels_check))

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
    """Create model and optimizer"""

    # Network
    net = getattr(models_factory, args.net)(num_classes=num_classes)

    # Multi-GPU support
    if args.gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        net = nn.DataParallel(net)

    if args.gpu:
        net = net.cuda()

    # Loss function
    loss_function = nn.CrossEntropyLoss()

    if args.gpu:
        loss_function = loss_function.cuda()

    # Optimizer
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


def train_epoch(net, trainloader, loss_function, optimizer,
                epoch, warmup_scheduler, args):
    """Train one epoch"""

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

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                net.parameters(),
                args.grad_clip
            )

        optimizer.step()

        # Update warmup scheduler
        if epoch <= args.warmup_epochs:
            warmup_scheduler.step()

        # Statistics
        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Progress logging
        if batch_idx % args.log_interval == 0:

            avg_loss = running_loss / (batch_idx + 1)
            acc = 100. * correct / total
            current_lr = optimizer.param_groups[0]['lr']

            print(
                f'Epoch: {epoch} [{batch_idx}/{len(trainloader)}] '
                f'Loss: {avg_loss:.4f} | '
                f'Acc: {acc:.2f}% | '
                f'LR: {current_lr:.6f}'
            )

    epoch_time = time.time() - start_time

    avg_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total

    return avg_loss, epoch_acc, epoch_time


def main():
    """Main function"""

    # ------------------------
    # Argument parser
    # ------------------------
    parser = argparse.ArgumentParser(
        description="Training script with resume support"
    )

    # Model and dataset parameters
    parser.add_argument(
        "-net", type=str, default='ViT',
        help="Network type"
    )

    parser.add_argument(
        "-dataset", type=str, default='Cifar20',
        help="Dataset name"
    )

    parser.add_argument(
        "-classes", type=int, default=20,
        help="Number of classes"
    )

    # Training parameters
    parser.add_argument(
        "-epochs", type=int, default=3000,
        help="Total training epochs"
    )

    parser.add_argument(
        "-batch_size", "-b", type=int, default=256,
        help="Batch size"
    )

    parser.add_argument(
        "-learning_rate", "-lr", type=float, default=5e-4,
        help="Learning rate"
    )

    parser.add_argument(
        "-weight_decay", type=float, default=1e-4,
        help="Weight decay"
    )

    parser.add_argument(
        "-momentum", type=float, default=0.9,
        help="Momentum"
    )

    parser.add_argument(
        "-grad_clip", type=float, default=5.0,
        help="Gradient clipping"
    )

    # Scheduler parameters
    parser.add_argument(
        "-warmup_epochs", type=int, default=5,
        help="Number of warmup epochs"
    )

    parser.add_argument(
        "-min_lr", type=float, default=1e-6,
        help="Minimum learning rate"
    )

    # System parameters
    parser.add_argument(
        "-gpu", type=bool, default=True,
        help="Use GPU"
    )

    parser.add_argument(
        "-seed", type=int, default=0,
        help="Random seed"
    )

    parser.add_argument(
        "-resume", action="store_true",
        help="Resume training from checkpoint"
    )

    parser.add_argument(
        "-no_resume", action="store_true",
        help="Force training from scratch"
    )

    parser.add_argument(
        "-log_interval", type=int, default=50,
        help="Logging interval"
    )

    parser.add_argument(
        "-save_interval", type=int, default=10,
        help="Checkpoint save interval"
    )

    args = parser.parse_args()

    # Save arguments globally for checkpoint saving
    global args_dict
    args_dict = vars(args)

    # Set random seed
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ------------------------
    # Output path
    # ------------------------
    output_path = (
        Path(config.CHECKPOINT_PATH)
        / "pretrain"
        / f"{args.net}-{args.dataset}-{args.classes}"
    )

    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------
    # Setup logging
    # ------------------------
    logger = setup_logging(output_path)

    # Log arguments
    logger.info("Training arguments:")

    for key, value in args_dict.items():
        logger.info(f"  {key}: {value}")

    # ------------------------
    # Create data loaders
    # ------------------------
    logger.info("Creating data loaders...")

    trainloader, testloader, dataset_size = create_data_loaders(args)

    logger.info(f"Training set size: {dataset_size}")
    logger.info(f"Test set size: {len(testloader.dataset)}")

    # ------------------------
    # Create model and optimizer
    # ------------------------
    logger.info("Creating model and optimizer...")

    net, loss_function, optimizer = create_model_and_optimizer(
        args,
        args.classes
    )

    # ------------------------
    # Create schedulers
    # ------------------------
    iter_per_epoch = len(trainloader)

    warmup_scheduler = WarmUpLR(
        optimizer,
        iter_per_epoch * args.warmup_epochs
    )

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )

    # ------------------------
    # Checkpoint manager
    # ------------------------
    checkpoint_manager = CheckpointManager(
        output_path,
        f"{args.net}-{args.dataset}",
        net
    )

    # ------------------------
    # Resume training or start from scratch
    # ------------------------
    if args.no_resume:

        start_epoch = 1
        best_acc = 0.0

        logger.info("Force training from scratch")

    else:

        start_epoch, best_acc, loaded = checkpoint_manager.load_checkpoint(
            optimizer,
            scheduler,
            warmup_scheduler
        )

        if not loaded and not args.resume:

            start_epoch = 1
            best_acc = 0.0

            logger.info("Training from scratch")

        elif loaded:

            logger.info(
                f"Resumed training from epoch {start_epoch}, "
                f"best accuracy: {best_acc:.4f}"
            )

    # ------------------------
    # Training loop
    # ------------------------
    logger.info("Start training...")

    try:

        for epoch in range(start_epoch, args.epochs + 1):

            # Train one epoch
            train_loss, train_acc, epoch_time = train_epoch(
                net,
                trainloader,
                loss_function,
                optimizer,
                epoch,
                warmup_scheduler,
                args
            )

            # Update learning rate scheduler
            if epoch > args.warmup_epochs:
                scheduler.step()

            # Evaluation
            test_acc = eval_training(epoch, net, testloader)

            current_lr = optimizer.param_groups[0]['lr']

            # Log results
            logger.info(
                f"Epoch {epoch}/{args.epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Acc: {train_acc:.2f}%, "
                f"Test Acc: {test_acc:.4f}, "
                f"LR: {current_lr:.6f}, "
                f"Time: {epoch_time:.2f}s"
            )

            # Check best model
            is_best = test_acc > best_acc

            if is_best:
                best_acc = test_acc
                logger.info(f"New best accuracy: {best_acc:.4f}")

            # Save checkpoint periodically
            if (
                epoch % args.save_interval == 0
                or epoch == args.epochs
                or is_best
            ):
                checkpoint_manager.save_checkpoint(
                    epoch,
                    optimizer,
                    scheduler,
                    warmup_scheduler,
                    best_acc,
                    is_best=is_best
                )

    except KeyboardInterrupt:

        logger.info("Training interrupted, saving current state...")

        checkpoint_manager.save_checkpoint(
            epoch - 1 if 'epoch' in locals() else start_epoch - 1,
            optimizer,
            scheduler,
            warmup_scheduler,
            best_acc,
            is_interrupt=True
        )

        logger.info("Interrupted checkpoint saved")

    except Exception as e:

        logger.error(f"Error during training: {e}")
        logger.error("Attempting to save current state...")

        checkpoint_manager.save_checkpoint(
            epoch - 1 if 'epoch' in locals() else start_epoch - 1,
            optimizer,
            scheduler,
            warmup_scheduler,
            best_acc,
            is_interrupt=True
        )

        raise

    finally:

        logger.info(
            f"Training completed, best test accuracy: {best_acc:.4f}"
        )

        logger.info(f"Model saved in: {output_path}")


if __name__ == "__main__":
    main()