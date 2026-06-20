import random
from copy import deepcopy

from sympy import false
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillKL(nn.Module):
    
    def __init__(self, T=4):
        super().__init__()
        self.T = T

    def forward(self, y_s, y_t):
        
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        return loss


import models.models_factory as models_factory


from thirdparty.repdistiller.distiller_zoo import DistillKL
from thirdparty.repdistiller.helper.loops import validate_scrub, train_distill
from thirdparty.repdistiller.helper.util import adjust_learning_rate



from unlearn import *


from utils.metrics import UnLearningScore, get_membership_attack_prob
from utils.overall_utils import *
from utils.training_utils import *
from utils.ssd import *
from utils.utils import *


import config
import time
from boundary_unlearning.boundary_utils import *




import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def standard_mia(model, train_loader, test_loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def collect_losses(loader):
        losses = []
        with torch.no_grad():
            for x, _, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                losses.extend(loss.detach().cpu().numpy())
        return np.array(losses)

    # member vs non-member
    train_losses = collect_losses(train_loader)
    test_losses = collect_losses(test_loader)

    # labels
    y_true = np.concatenate([
        np.ones(len(train_losses)),   # member
        np.zeros(len(test_losses))    # non-member
    ])

    scores = np.concatenate([
        -train_losses,   
        -test_losses
    ])

    # metrics
    acc = accuracy_score(y_true, scores > np.median(scores))
    auc = roc_auc_score(y_true, scores)

    return acc, auc


# Create datasets of the classes
def get_classwise_ds(ds, num_classes):
    classwise_ds = {}
    for i in range(num_classes):
        classwise_ds[i] = []

    for img, label, clabel in ds:
        classwise_ds[clabel].append((img, label, clabel))
    return classwise_ds


def get_metric_scores(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
):
    model.eval()

    loss_acc_dict = evaluate(model, retain_valid_dl, device)
    d_f_acc_dict = evaluate(model, forget_train_dl, device)
    retain_acc_dict = evaluate(model, retain_train_dl, device)

    return (
        loss_acc_dict["Acc"],
        d_f_acc_dict["Acc"],
        retain_acc_dict["Acc"]
    )
# Does nothing; original model



class LabelReassignDataset(torch.utils.data.Dataset):
    def __init__(self, subset, forget_class):
        self.subset = subset
        self.forget_class = forget_class

    def __getitem__(self, index):
        
        x, y, cy = self.subset[index]

        
        new_cy = cy - 1 if cy > self.forget_class else cy

        
        return x, y, new_cy

    def __len__(self):
        return len(self.subset)



    



def baseline(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
        weights_path,
    **kwargs,
):
    end = time.time()
    start = time.time()
    time_elapsed = end - start
    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        valid_dl,
        device,
    ), time_elapsed

def salun(model,
          unlearning_teacher,
          retain_train_dl,
          retain_valid_dl,
          forget_train_dl,
          forget_valid_dl,
          valid_dl,  
          num_classes, 
          device, 
          weights_path, 
          para1='0.0001', 
          para2='2',
          mask_path=None, 
          rum=False,
          **kwargs):
    
    start_time = time.time()

    
    mask = torch.load(mask_path) if mask_path else None
    
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(para1))

    for epoch in range(int(para2)):
        model.train()
        start = time.time()
        losses = AverageMeter()
        top1 = AverageMeter()
        loader_len = len(forget_train_dl) + len(retain_train_dl)

        
        for i, batch in enumerate(forget_train_dl):
            image, _, target = batch
            image = image.cuda()
            target = torch.randint(0, num_classes, target.shape).cuda()

            
            output_clean = model(image)
            loss = criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()

            
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

        
        for i, batch in enumerate(retain_train_dl):
            image, _, target = batch
            image = image.cuda()
            target = target.cuda()

            
            output_clean = model(image)
            loss = criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()

            
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

            
            prec1 = accuracy(output_clean.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

        if (i + 1) % 100 == 0:
            end = time.time()
            print(f'Epoch: [{epoch}][{i}/{loader_len}]\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  f'Time {end - start:.2f}')

            start = time.time()

    
    if not rum:
        torch.save(model.state_dict(), weights_path)

        
        d_t, d_f, d_r = get_metric_scores(
            model, 
            unlearning_teacher, 
            retain_train_dl, 
            retain_valid_dl, 
            forget_train_dl,
            forget_valid_dl, 
            valid_dl, 
            device=device, 
            )

        print("[Final]")
        print(f"d_t = {d_t} | d_f = {d_f} | d_r = {d_r} ")
        
        time_elapsed = time.time() - start_time

        return (d_t, d_f, d_r), time_elapsed
    else:
        return model















def retrain(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    dataset_name,
    model_name,
    device,
    num_classes,
    weights_path,
    para1='0.1',
    para2='200',
    rum=False,
    **kwargs,
):
    import time
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.transforms as T
    import models.models_factory as models_factory
    from utils.training_utils import WarmUpLR

    start_total_time = time.time()
    lr = float(para1)
    epochs = int(para2)

    model = getattr(models_factory, model_name)(num_classes=num_classes)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.cuda()

    if model_name == "ViT":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    iter_per_epoch = len(retain_train_dl)
    warmup_epochs = 5
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warmup_epochs)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    loss_function = nn.CrossEntropyLoss().cuda()

    gpu_transform = T.Compose([
        T.RandomCrop(32, padding=4, fill=0),
        T.RandomHorizontalFlip(p=0.5)
    ])

    
    print("--------------------------------------------------------------------------------")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start_time = time.time()

        train_losses = AverageMeter()
        train_top1 = AverageMeter()

        for batch_idx, (images, _, labels) in enumerate(retain_train_dl):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            images = gpu_transform(images)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if epoch <= warmup_epochs:
                warmup_scheduler.step()

            prec1 = accuracy(outputs.data, labels)[0]
            train_losses.update(loss.item(), images.size(0))
            train_top1.update(prec1.item(), images.size(0))

        if epoch > warmup_epochs:
            scheduler.step()

        model.eval()
        test_top1 = AverageMeter()

        with torch.no_grad():
            for images, _, labels in valid_dl:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                outputs = model(images)
                prec1 = accuracy(outputs.data, labels)[0]
                test_top1.update(prec1.item(), images.size(0))

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch [{epoch:3d}/{epochs}] | "
            f"Train Loss: {train_losses.avg:.4f} | "
            f"RA (Train Acc): {train_top1.avg:.2f}% | "
            f"TA (Test Acc): {test_top1.avg:.2f}% | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

    print("--------------------------------------------------------------------------------")
    

    if not rum:
        torch.save(model.state_dict(), weights_path)

        d_t, d_f, d_r = get_metric_scores(
            model,
            unlearning_teacher,
            retain_train_dl,
            retain_valid_dl,
            forget_train_dl,
            forget_valid_dl,
            valid_dl,
            device=device,
        
    )

        print("[Final]")
        print(
        "d_t =", d_t,
        "| d_f =", d_f,
        "| d_r =", d_r
    )

        
        time_elapsed = time.time() - start_total_time
        return (d_t, d_f, d_r), time_elapsed
    else:
        return model

# Implementation from https://github.com/vikram2000b/bad-teaching-unlearning
def amnesiac(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_classes,
    forget_class,
    device,
    mask=None,
    weights_path=None,
    para1=0.0001,
    para2=3,
    **kwargs,
):
    import time
    start = time.time()


    unlearninglabels = list(range(num_classes))
    unlearninglabels.remove(forget_class)

    unlearning_trainset = []

    for x, _, clabel in forget_train_dl.dataset:
        unlearning_trainset.append((x, _, random.choice(unlearninglabels)))

    for x, _, y in retain_train_dl.dataset:
        unlearning_trainset.append((x, _, y))

    unlearning_train_set_dl = DataLoader(
        unlearning_trainset, 128, pin_memory=True, shuffle=True
    )


    _ = fit_one_unlearning_cycle(
        int(para2),
        model,
        unlearning_train_set_dl,
        retain_valid_dl,
        device=device,
        lr=float(para1),
        mask=mask
    )


    d_t, d_f, d_r = get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device=device,
        
    )

    print("[Final]")
    print("d_t =", d_t, "| d_f =", d_f, "| d_r =", d_r)


    torch.save(model.state_dict(), weights_path)

    time_elapsed = time.time() - start

    return (d_t, d_f, d_r), time_elapsed

def scrub(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    weights_path,
    para1=0.001,
    para2=5,
    **kwargs
):
    import time
    import copy
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from types import SimpleNamespace

    start_time = time.time()

    # teacher / student
    model_t = copy.deepcopy(model)
    model_s = copy.deepcopy(model)


    cfg = SimpleNamespace()
    cfg.optim = "adam"
    cfg.gamma = 1
    cfg.alpha = 0.1
    cfg.beta = 0
    cfg.smoothing = 0.5
    cfg.msteps = 5
    cfg.clip = 0.2
    cfg.sstart = 10
    cfg.kd_T = 4
    cfg.distill = "kd"
    cfg.print_freq = 50


    cfg.sgda_epochs = int(para2)
    cfg.sgda_learning_rate = float(para1)
    cfg.lr_decay_epochs = [3, 5, 9]
    cfg.lr_decay_rate = 0.1
    cfg.sgda_weight_decay = 5e-4
    cfg.sgda_momentum = 0.9

    # -----------------------------
    # module / loss
    # -----------------------------
    module_list = nn.ModuleList([])
    module_list.append(model_s)

    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(cfg.kd_T)
    criterion_kd = DistillKL(cfg.kd_T)

    criterion_list = nn.ModuleList([
        criterion_cls,
        criterion_div,
        criterion_kd
    ])

    module_list.append(model_t)

    # -----------------------------
    # optimizer
    # -----------------------------
    if cfg.optim.lower() == "adam":
        optimizer = optim.Adam(
            trainable_list.parameters(),
            lr=cfg.sgda_learning_rate,
            weight_decay=cfg.sgda_weight_decay
        )
    else:
        optimizer = optim.SGD(
            trainable_list.parameters(),
            lr=cfg.sgda_learning_rate,
            momentum=cfg.sgda_momentum,
            weight_decay=cfg.sgda_weight_decay
        )

    # -----------------------------
    # cuda
    # -----------------------------
    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    model_t.eval()
    for p in model_t.parameters():
        p.requires_grad = False



    for epoch in range(1, cfg.sgda_epochs + 1):
        adjust_learning_rate(epoch, cfg, optimizer)

        if epoch <= cfg.msteps:
            train_distill(
                epoch,
                forget_train_dl,
                module_list,
                None,
                criterion_list,
                optimizer,
                cfg,
                "maximize"
            )

        train_distill(
            epoch,
            retain_train_dl,
            module_list,
            None,
            criterion_list,
            optimizer,
            cfg,
            "minimize"
        )


    model = copy.deepcopy(model_s)


    d_t, d_f, d_r = get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device=device,
        
    )

    print("[Final]")
    print(
        "d_t =", d_t,
        "| d_f =", d_f,
        "| d_r =", d_r
    )

    
    torch.save(model.state_dict(), weights_path)

    time_elapsed = time.time() - start_time

    return (d_t, d_f, d_r), time_elapsed









def unrolling(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
        weights_path,
        para1='0.01',
        para2='0.8',
        model_name=None,
        **kwargs,
):
    import time
    start = time.time()

    lr = float(para1)
    beta = float(para2)
    epochs = 50
    criterion = nn.CrossEntropyLoss()


    if model_name is None:
        cls_name = type(model).__name__  # ResNet18 / ResNet34 / ResNet50 / ViT...
        if 'ResNet' in cls_name:
            model_name = cls_name.lower()  # resnet18, resnet34...
        elif 'ViT' in cls_name:
            model_name = 'vit'
        else:
            model_name = 'unknown'

    if model_name.startswith("ViT"):
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:  
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)


    model.train()

    for epoch in range(epochs):
        retain_iter = iter(retain_train_dl)
        forget_iter = iter(forget_train_dl)

        for _ in range(min(len(retain_train_dl), len(forget_train_dl))):

            img_r, _, label_r = next(retain_iter)
            img_f, _, label_f = next(forget_iter)

            img_r, label_r = img_r.to(device), label_r.to(device)
            img_f, label_f = img_f.to(device), label_f.to(device)

            optimizer.zero_grad()

            loss_retain = criterion(model(img_r), label_r)
            loss_forget = criterion(model(img_f), label_f)

            loss = loss_retain - beta * loss_forget

            loss.backward()
            optimizer.step()


    d_t, d_f, d_r = get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device=device,
        
    )

    print("[Final]")
    print("d_t =", d_t, "| d_f =", d_f, "| d_r =", d_r)

    torch.save(model.state_dict(), weights_path)

    time_elapsed = time.time() - start

    return (d_t, d_f, d_r), time_elapsed





import time
import torch
import torch.nn as nn
import torch.optim as optim



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validate(val_loader, model, criterion, print_freq):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, _, target) in enumerate(val_loader):
        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                    i, len(val_loader), loss=losses, top1=top1
                )
            )

    print("valid_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg

def warmup_lr(epoch, step, optimizer, one_epoch_step, warmup, lr0):
    overall_steps = warmup * one_epoch_step
    current_steps = epoch * one_epoch_step + step

    lr = lr0 * current_steps / overall_steps
    lr = min(lr, lr0)

    for p in optimizer.param_groups:
        p["lr"] = lr

def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)







from torch.autograd import grad
def get_x_y_from_data_dict(data, device):
    x, y = data.values()
    if isinstance(x, list):
        x, y = x[0].to(device), y[0].to(device)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def sam_grad(model, loss):
    params = []
    for param in model.parameters():
        params.append(param)
    sample_grad = grad(loss, params)
    sample_grad = [x.view(-1) for x in sample_grad]
    return torch.cat(sample_grad)

def apply_perturb(model, v):
    curr = 0
    with torch.no_grad():
        for param in model.parameters():
            length = param.view(-1).shape[0]
            param += v[curr: curr + length].view(param.shape)
            curr += length

def woodfisher(model, train_dl, device, criterion, v):
    model.eval()
    k_vec = torch.clone(v)
    N = 1000
    o_vec = None
    for idx, batch in enumerate(tqdm(train_dl)):
        data, labels, clabels = batch
        model.zero_grad()
        data = data.to(device)
        label = clabels.to(device)
        output = model(data)
        loss = criterion(output, label)
        sample_grad = sam_grad(model, loss)
        with torch.no_grad():
            if o_vec is None:
                o_vec = torch.clone(sample_grad)
            else:
                tmp = torch.dot(o_vec, sample_grad)
                k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
                o_vec -= (tmp / (N + tmp)) * o_vec
        if idx > N:
            return k_vec
    return k_vec

def woodfisher_im(model, train_dl, device, criterion, v):
    model.eval()
    k_vec = torch.clone(v)
    N = 300000
    o_vec = None
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    for idx, batch in enumerate(tqdm(train_dl)):
        data, labels, clabels = batch
        model.zero_grad()
        data = data.to(device)
        label = clabels.to(device)
        output = model(data)
        loss = criterion(output, label)
        sample_grad = sam_grad(model, loss)
        with torch.no_grad():
            if o_vec is None:
                o_vec = torch.clone(sample_grad)
            else:
                tmp = torch.dot(o_vec, sample_grad)
                k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
                o_vec -= (tmp / (N + tmp)) * o_vec
        if idx > N:
            return k_vec
    return k_vec




def bad_t(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
        weights_path=None,
        para1='0.0001',
        para2='1',
        model_name=None,
        **kwargs,
):
    """
    Bad-Teaching Unlearning Method (blindspot_unlearner)
    para1: learning rate (default 0.0001)
    para2: epochs (default 1)
    """
    import copy
    import time
    from itertools import cycle
    import torch.nn.functional as F

    start = time.time()
    lr = float(para1)
    epochs = int(para2)
    KL_temperature = 1

    
    full_trained_teacher = copy.deepcopy(model).to(device)
    full_trained_teacher.eval()

    
    unlearning_teacher.eval()


    if model_name is None:
        cls_name = type(model).__name__  # ResNet18 / ResNet34 / ResNet50 / ViT...
        if 'ResNet' in cls_name:
            model_name = cls_name.lower()  # resnet18, resnet34...
        elif 'ViT' in cls_name:
            model_name = 'vit'
        else:
            model_name = 'unknown'

    if model_name == "ViT":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        
        for i, (retain_batch, forget_batch) in enumerate(zip(retain_train_dl, cycle(forget_train_dl))):

            img_r, _, _ = retain_batch
            img_f, _, _ = forget_batch

            
            images = torch.cat((img_r, img_f), dim=0).to(device)

            
            labels_r = torch.zeros(img_r.size(0)).to(device)
            labels_f = torch.ones(img_f.size(0)).to(device)
            labels = torch.cat((labels_r, labels_f), dim=0).unsqueeze(1)

            
            with torch.no_grad():
                full_teacher_logits = full_trained_teacher(images)
                unlearn_teacher_logits = unlearning_teacher(images)

            
            student_logits = model(images)

            
            f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
            u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

            
            overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
            student_out = F.log_softmax(student_logits / KL_temperature, dim=1)

            
            loss = F.kl_div(student_out, overall_teacher_out, reduction='batchmean')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] Step [{i + 1}/{len(retain_train_dl)}] Loss: {loss.item():.4f}")

    end = time.time()
    time_elapsed = end - start


    torch.save(model.state_dict(), weights_path)


    d_t, d_f, d_r = get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device=device,
        
    )

    print("[Final]")
    print("d_t =", d_t, "| d_f =", d_f, "| d_r =", d_r)

    return (d_t, d_f, d_r), time_elapsed





def l_codec_unlearn(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
        weights_path=None,
        para1='0.1',  
        para2='1',    
        model_name=None,
        **kwargs,
):

    import time
    import torch
    import torch.nn as nn
    import torch.optim as optim

    start = time.time()
    threshold = float(para1)
    repair_epochs = int(para2)

    model.eval()
    criterion = nn.CrossEntropyLoss()
    if model_name is None:
        cls_name = type(model).__name__  # ResNet18 / ResNet34 / ResNet50 / ViT...
        if 'ResNet' in cls_name:
            model_name = cls_name.lower()  # resnet18, resnet34...
        elif 'ViT' in cls_name:
            model_name = 'vit'
        else:
            model_name = 'unknown'

    print("Step 1: Analyzing Conditional Dependence using L-CODEC logic...")
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            target_layer = module

    if target_layer is not None:
        for i, (image, _, target) in enumerate(forget_train_dl):
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target)
            model.zero_grad()
            loss.backward()

            
            with torch.no_grad():
                grad_mask = (target_layer.weight.grad.abs() > threshold * target_layer.weight.grad.abs().max())
                target_layer.weight.data[grad_mask] *= -0.1  

            if i > 5:  
                break


    if repair_epochs > 0:
        print(f"Step 2: Repairing model for {repair_epochs} epochs...")
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        model.train()
        for epoch in range(repair_epochs):
            for i, (image, _, target) in enumerate(retain_train_dl):
                image, target = image.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(image)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                if i > 100:  
                    break


    d_t, d_f, d_r = get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device=device,
        
    )

    print("[Final] L-CODEC")
    print("d_t =", d_t, "| d_f =", d_f, "| d_r =", d_r)


    if weights_path:
        torch.save(model.state_dict(), weights_path)

    time_elapsed = time.time() - start
    return (d_t, d_f, d_r), time_elapsed
