# From https://github.com/vikram2000b/bad-teaching-unlearning
import csv

import torch
from torch.utils.data import Subset

from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F
from utils.training_utils import *

import numpy as np

def construct_forget_memorization(forget_dataset_index, npz_path):
    
    npz_data = np.load(npz_path)
    mem_all = npz_data['mem']  
    
    mem_forget = mem_all[forget_dataset_index]
    return mem_forget

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)) * 100


def training_step(model, batch, device):
    images, labels, clabels = batch
    images, clabels = images.to(device), clabels.to(device)
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, clabels)  # Calculate loss
    return loss

def validation_step(model, batch, device):
    images, labels, clabels = batch   # labels 100, clabels 20
    images, clabels = images.to(device), clabels.to(device)
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, clabels)  # Calculate loss
    acc = accuracy(out, clabels)  # Calculate accuracy
    return {"Loss": loss.detach(), "Acc": acc}

def validation_step_rum(model, batch, device):
    images, labels, _ = batch   # labels 100, clabels 20
    images, labels = images.to(device), labels.to(device)
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, labels)  # Calculate loss
    acc = accuracy(out, labels)  # Calculate accuracy
    return {"Loss": loss.detach(), "Acc": acc}

def validation_epoch_end(model, outputs):
    batch_losses = [x["Loss"] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
    batch_accs = [x["Acc"] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
    return {"Loss": epoch_loss.item(), "Acc": epoch_acc.item()}


def epoch_end(model, epoch, result, train_acc=0.00):
    print(
        "Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, train_acc: {:.4f}".format(
            epoch,
            result["lrs"][-1],
            result["train_loss"],
            result["Loss"],
            result["Acc"],
            train_acc,
        )
    )

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    outputs = [validation_step(model, batch, device) for batch in val_loader]
    return validation_epoch_end(model, outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

@torch.no_grad()
def eval_training(epoch, net, testloader, tb=True):
    
    loss_function = nn.CrossEntropyLoss()
    net.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, _, labels in testloader:
            images = images.cuda()
            labels = labels.cuda()

            outputs = net(images)
            loss = loss_function(outputs, labels)
            test_loss += loss.item() * labels.size(0)  # 按样本累加
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = test_loss / total
    acc = correct / total
    print("Evaluating Network.....")
    print(
        f"Test set: Epoch: {epoch}, Average loss: {avg_loss:.4f}, Accuracy: {acc:.4f}"
    )
    return acc


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)

def fit_one_cycle(
    epochs, model, train_loader, val_loader, device, lr=0.01, model_name='ResNet18', milestones=None, mask=None, l1=False
):
    torch.cuda.empty_cache()
    history = []
    if model_name == 'ViT':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)

    if milestones:
        train_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.2
        )  # learning rate decay
        warmup_scheduler = WarmUpLR(optimizer, len(train_loader))

    for epoch in range(epochs):
        if epoch > 1 and milestones:
            train_scheduler.step(epoch)

        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch, device)
            if l1:
                 loss += 5e-5*l1_regularization(model)
            train_losses.append(loss)
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))

            if epoch <= 1 and milestones:
                warmup_scheduler.step()

        # Validation phase
        train_result = evaluate(model, train_loader, device)
        result = evaluate(model, val_loader, device)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        result["lrs"] = lrs
        epoch_end(model, epoch, result, train_result['Acc'])
        history.append(result)

        #acc = eval_training(epoch, model, val_loader)
    return history

def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for index, row in enumerate(csv_reader):
            if index > 0:
                data.append([float(x) for x in row[0].split('\t')])
    return np.asarray(data)

def build_retain_sets_in_unlearning(classwise_train, classwise_test, num_classes, forget_class, ood_class=None):
    # Getting the retain validation data
    all_class = list(range(0, num_classes))
    if ood_class is not None:
        retain_class = list(set(all_class) - set(ood_class))
    else:
        retain_class = list(all_class)

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

def get_classwise_ds(ds, num_classes):
    classwise_ds = {i: [] for i in range(num_classes)}

    for sample in ds:
        if len(sample) == 2:
            img, label = sample
            clabel = label   
        elif len(sample) == 3:
            img, label, clabel = sample
        else:
            raise ValueError(f"Incorrect Dataset Sample Format: {sample}")

        if clabel not in classwise_ds:
            raise KeyError(f"clabel={clabel} Out of range 0~{num_classes-1}")

        classwise_ds[clabel].append((img, label, clabel))

    return classwise_ds
