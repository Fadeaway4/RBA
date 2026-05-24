from __future__ import print_function
import argparse, ipdb, json
import copy
import math
from datetime import datetime

import torch.optim as optim

import os, random

from torch.utils.data import DataLoader, ConcatDataset

import datasets
import forget_full_class_strategies

from dfme_utils.my_utils import *
import os.path as osp

import config
import models
from utils import build_retain_sets_in_unlearning
import time

print("torch version", torch.__version__)

def myprint(a):
    print(a);

def student_loss(args, s_logit, t_logit, return_t_logits=False):
    """Kl/ L1 Loss for student"""
    print_logits = False
    if args.loss == "l1":
        loss_fn = F.l1_loss
        loss = loss_fn(s_logit, t_logit.detach())
    elif args.loss == "kl":
        loss_fn = F.kl_div
        s_logit = F.log_softmax(s_logit, dim=1)
        t_logit = F.softmax(t_logit, dim=1)  # here has t-logits!
        loss = loss_fn(s_logit, t_logit.detach(), reduction="batchmean")
    else:
        raise ValueError(args.loss)

    if return_t_logits:
        return loss, t_logit.detach()
    else:
        return loss

def dfme_train(args, teacher, student, generator, device, optimizer, epoch):
    """Main Loop for one epoch of Training Generator and Student"""
    global file
    teacher.eval()
    student.train()

    optimizer_S, optimizer_G = optimizer

    gradients = []
    correct = 0
    total = 0
    for i in range(args.epoch_itrs):
        """Repeat epoch_itrs times per epoch"""
        for _ in range(args.g_iter):
            # Sample Random Noise
            z = torch.randn((args.batch_size, args.nz)).to(device)
            optimizer_G.zero_grad()
            generator.train()
            # Get fake image from generator
            fake = generator(z, pre_x=args.approx_grad)  # pre_x returns the output of G before applying the activation

            ## APPOX GRADIENT
            approx_grad_wrt_x, loss_G = estimate_gradient_objective(args, teacher, student, fake,
                                                                    epsilon=args.grad_epsilon, m=args.grad_m,
                                                                    num_classes=args.classes,
                                                                    device=device, pre_x=True)

            fake.backward(approx_grad_wrt_x)

            optimizer_G.step()

            # if i == 0 and args.rec_grad_norm:
            #     x_true_grad = measure_true_grad_norm(args, fake)
        x_g = []
        y_g = []
        for _ in range(args.d_iter):
            z = torch.randn((args.batch_size, args.nz)).to(device)
            fake = generator(z).detach()
            optimizer_S.zero_grad()

            with torch.no_grad():
                t_logit = teacher(fake)
            # Correction for the fake logits
            if args.loss == "l1" and args.no_logits:
                t_logit = F.log_softmax(t_logit, dim=1).detach()
                if args.logit_correction == 'min':
                    t_logit -= t_logit.min(dim=1).values.view(-1, 1).detach()
                elif args.logit_correction == 'mean':
                    t_logit -= t_logit.mean(dim=1).view(-1, 1).detach()

            y_g.append(t_logit)
            x_g.append(fake)
            s_logit = student(fake)
            # print("fake,", fake.shape)
            # print("student,", student)

            loss_S = student_loss(args, s_logit, t_logit)
            loss_S.backward()
            optimizer_S.step()
            total += len(s_logit)
            correct += torch.argmax(s_logit, dim=1, keepdim=True).eq(
                torch.argmax(t_logit, dim=1, keepdim=True)).sum().item()
            train_acc = correct / total * 100.
        # Log Results
        if i % args.log_interval == 0:
            myprint(f'Train Epoch: {epoch}[{i}/{args.epoch_itrs}'
                    f' ({100 * float(i) / float(args.epoch_itrs):.0f}%)]\tG_Loss: {loss_G.item():.6f} S_loss: {loss_S.item():.6f} ACC: '
                    f'{train_acc:.4f}%')
            """
            myprint(f'Train Epoch: {epoch} [{i}/{args.epoch_itrs}'
                    f' ({100 * float(i) / float(args.epoch_itrs):.0f}%)]\tG_Loss: {loss_G.item():.6f} S_loss: {loss_S.item():.6f} ACC: '
                    f'{train_acc:.4f}%')
            """

            # if args.rec_grad_norm and i == 0:
            #     G_grad_norm, S_grad_norm = compute_grad_norms(generator, student)
            #     if i == 0:
            #         with open(args.log_dir + "/norm_grad.csv", "a") as f:
            #             f.write("%d,%f,%f,%f\n"%(epoch, G_grad_norm, S_grad_norm, x_true_grad))

        # update query budget
        args.query_budget -= args.cost_per_iteration

        if args.query_budget < args.cost_per_iteration:
            return x_g, y_g, loss_S, train_acc
    return x_g, y_g, loss_S, train_acc


def dfme_test(student=None, generator=None, device="cuda", test_loader=None, blackbox=None):
    global file
    student.eval()
    generator.eval()

    test_loss = 0
    correct = 0
    equal_item = 0
    equal_item_prac = 0
    with torch.no_grad():
        for i, (data, _, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = student(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            t_pred = blackbox(data)
            t_pred = t_pred.argmax(dim=1, keepdim=True)
            equal_item += pred.eq(t_pred).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    fidelity = equal_item / len(test_loader.dataset) * 100.
    myprint(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%), Fidelity:{}/{} ({:4}/%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            accuracy, equal_item, len(test_loader.dataset), fidelity))

    return accuracy, fidelity

def compute_grad_norms(generator, student):
    G_grad = []
    for n, p in generator.named_parameters():
        if "weight" in n:
            # print('===========\ngradient{}\n----------\n{}'.format(n, p.grad.norm().to("cpu")))
            G_grad.append(p.grad.norm().to("cpu"))

    S_grad = []
    for n, p in student.named_parameters():
        if "weight" in n:
            # print('===========\ngradient{}\n----------\n{}'.format(n, p.grad.norm().to("cpu")))
            S_grad.append(p.grad.norm().to("cpu"))
    return np.mean(G_grad), np.mean(S_grad)

def main_runner():
    # Training settings
    # parser = argparse.ArgumentParser(description='DFAD CIFAR')

    params = vars(args)
    if args.MAZE:
        print("\n" * 2)
        print("#### /!\ OVERWRITING ALL PARAMETERS FOR MAZE REPLCIATION ####")
        print("\n" * 2)
        args.scheduer = "cosine"
        args.loss = "kl"
        # args.batch_size = 32#128
        args.g_iter = 1  # 1
        args.d_iter = 5  # 5
        args.grad_m = 10  # 10
        # args.lr_G = 1e-4
        # args.lr_S = 1e-1#1e-1

    args.query_budget *= 10 ** 6
    args.query_budget = int(args.query_budget)
    nc = args.nc  # 1 #channel
    img_size = args.img_size  # 28 #size

    out_path = osp.join(config.CHECKPOINT_PATH, 'model_extraction', args.dataset)
    args.log_dir = out_path
    # pprint(args, width=80)
    print(args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)

    # Save JSON with parameters
    if not osp.exists(out_path):
        os.makedirs(out_path)

    params_out_path = osp.join(out_path, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Prepare the environment
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:%d" % args.device if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Preparing checkpoints for the best Student

    print(args)
    args.device = device
    args.normalization_coefs = None
    args.G_activation = torch.tanh


    # batch_size = args.b

    # get network
    student = getattr(models, args.student_net)(num_classes=args.classes)
    teacher = getattr(models, args.net)(num_classes=args.classes)

    checkpoint_path = os.path.join(config.CHECKPOINT_PATH,
                                   "{unlearning_scenarios}".format(unlearning_scenarios="forget_full_class_main"),
                                   "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset,
                                                                      classes=args.classes),
                                   "{task}".format(task="unlearning"),
                                   "{unlearning_method}-{para1}-{para2}".format(unlearning_method=args.method,
                                                                                para1=args.para1,
                                                                                para2=args.para2))

    print("#####", checkpoint_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if args.masked_path:
        weights_path = os.path.join(checkpoint_path, "{epoch}-{type}.pth").format(epoch=args.epochs, type="last-masked")
    else:
        weights_path = os.path.join(checkpoint_path, "{epoch}-{type}.pth").format(epoch=args.epochs, type="last")

    teacher.load_state_dict(torch.load(weights_path))
    if args.gpu:
        teacher = teacher.cuda()

    # For celebritiy faces
    root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"

    # Scale for ViT (faster training, better performance)
    img_size = 224 if args.net == "ViT" else 32

    # ood_class = [19]
    trainset = getattr(datasets, args.dataset)(root=root, download=True, train=True, unlearning=True, img_size=img_size)
    validset = getattr(datasets, args.dataset)(root=root, download=True, train=False, unlearning=True,
                                               img_size=img_size)

    classwise_train, classwise_test = forget_full_class_strategies.get_classwise_ds(trainset, num_classes=20), \
                                      forget_full_class_strategies.get_classwise_ds(validset, num_classes=20)
    # (retain_train, retain_valid) = forget_full_class_strategies.build_retain_sets(classwise_train, classwise_test, 20,
    #                                                                               forget_class, ood_class)
    (retain_train, retain_valid) = build_retain_sets_in_unlearning(classwise_train, classwise_test, 20,
                                                                   int(args.forget_class), config.ood_classes)

    # forget_train, forget_valid = classwise_train[forget_class], classwise_test[forget_class]

    test_loader = DataLoader(retain_valid, args.b)
    print("student,", student)

    correct = 0
    with torch.no_grad():
        for i, (data, _, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = teacher(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTeacher - Test set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, len(test_loader.dataset), accuracy))

    # generator = network.gan.GeneratorA(nz=main_args.nz, nc=3, img_size=32, activation=args.G_activation)
    generator = network.gan.GeneratorA(nz=args.nz, nc=nc, img_size=img_size, activation=args.G_activation)

    student = student.to(device)
    generator = generator.to(device)
    print("student", student)

    args.generator = generator
    args.student = student
    args.teacher = teacher

    if args.student_load_path:
        student.load_state_dict(torch.load(args.student_load_path))
        myprint("Student initialized from %s" % (args.student_load_path))
        acc, fidelity = dfme_test(student=student, generator=generator, device=device, test_loader=test_loader,
                                  blackbox=teacher)

    ## Compute the number of epochs with the given query budget:
    args.cost_per_iteration = args.batch_size * (args.g_iter * (args.grad_m + 1) + args.d_iter)

    number_epochs = args.query_budget // (args.cost_per_iteration * args.epoch_itrs) + 1

    print(f"\nTotal budget: {args.query_budget // 1000}k")
    print("Cost per iterations: ", args.cost_per_iteration)
    print("Total number of epochs: ", number_epochs)

    optimizer_S = optim.SGD(filter(lambda p: p.requires_grad, student.parameters()), lr=args.lr_S,
                            weight_decay=args.weight_decay, momentum=0.9)

    if args.MAZE:
        optimizer_G = optim.SGD(generator.parameters(), lr=args.lr_G, weight_decay=args.weight_decay, momentum=0.9)
    else:
        optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)

    steps = sorted([int(step * number_epochs) for step in args.steps])

    print("Learning rate scheduling at steps: ", steps)
    print()

    if args.scheduler == "multistep":
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, steps, args.scale)
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, steps, args.scale)
    elif args.scheduler == "cosine":
        scheduler_S = optim.lr_scheduler.CosineAnnealingLR(optimizer_S, number_epochs)
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, number_epochs)

    best_test_acc = 0.
    best_fidelity = 0.
    log_path = osp.join(out_path, f'{args.attack_set}.log.tsv')
    if not osp.exists(log_path):
        with open(log_path, 'w') as wf:
            columns = ['loss', 'epochs', 'query_number', 'training_acc', 'test_acc@1',
                       'best_test_acc', 'fidelity@1', 'best_fidelilty']
            wf.write('\t'.join(columns) + '\n')
    for epoch in range(1, number_epochs + 1):
        # Train
        if args.scheduler != "none":
            scheduler_S.step()
            scheduler_G.step()

        x_g, y_g, train_loss, train_acc = dfme_train(args, teacher=teacher, student=student, generator=generator,
                                                     device=device,
                                                     optimizer=[optimizer_S, optimizer_G], epoch=epoch)

        # Test
        with torch.no_grad():
            test_acc, test_fidelity = dfme_test(student=student, generator=generator, device=device,
                                                test_loader=test_loader, blackbox=teacher)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            if test_fidelity >= best_fidelity:
                best_fidelity = test_fidelity
                name = args.attack_set
                torch.save(student.state_dict(), out_path + f"/{name}.pth.tar")
                torch.save(generator.state_dict(), out_path + f"/{name}-generator.pth.tar")
                # state = {
                #     'epoch': 100,
                #     'arch': student.__class__,
                #     'state_dict': student.state_dict(),
                #     'best_acc': test_acc,
                #     'optimizer': optimizer_S,
                #     'created_on': str(datetime.now()),
                # }

        if epoch % 10 == 0:
            with open(log_path, 'a') as af:
                train_cols = [train_loss.item(), epoch, (args.cost_per_iteration * args.epoch_itrs) * epoch,
                              train_acc, test_acc,
                              best_test_acc, test_fidelity, best_fidelity]
                af.write('\t'.join([str(c) for c in train_cols]) + '\n')

        torch.cuda.empty_cache()
    with open(log_path, 'a') as af:
        train_cols = [train_loss.item(), epoch, (args.cost_per_iteration * args.epoch_itrs) * epoch,
                      train_acc, test_acc,
                      best_test_acc, test_fidelity, best_fidelity]
        af.write('\t'.join([str(c) for c in train_cols]) + '\n')
    myprint("Best Acc=%.6f" % best_test_acc)

if __name__ == '__main__':
    #############model extraction related##########################
    parser = argparse.ArgumentParser(description='DFAD')
    parser.add_argument('--attack_set', default='cifar_dfme')
    parser.add_argument('--query_budget', type=float, default=20)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--MAZE', type=bool, default=False)
    parser.add_argument('--nz', type=int, default=100, help="Size of random noise input to generator (256), 100")

    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epoch_itrs', type=int, default=50)
    parser.add_argument('--g_iter', type=int, default=1, help="Number of generator iterations per epoch_iter")
    parser.add_argument('--d_iter', type=int, default=5, help="Number of discriminator iterations per epoch_iter")

    parser.add_argument('--lr_S', type=float, default=0.02, metavar='LR', help='Student learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=3e-4, help='Generator learning rate (default: 1e-4)')

    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'kl'], )
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['multistep', 'cosine', "none"], )
    parser.add_argument('--steps', nargs='+', default=[0.1, 0.3, 0.5], type=float,
                        help="Percentage epochs at which to take next step")
    parser.add_argument('--scale', type=float, default=3e-1, help="Fractional decrease in lr")

    # parser.add_argument('--dataset', type=str, default='mnist', choices=['svhn','cifar10', 'mnist'], help='dataset name (default: cifar10)')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--model_arch', type=str, default='resnet18', choices=classifiers,
                        help='Target model name (default: resnet34_8x)')
    # parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='Cifar20')

    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--student_load_path', type=str, default=None)
    parser.add_argument('--model_id', type=str, default="debug")

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default=config.CHECKPOINT_PATH)

    # Gradient approximation parameters
    parser.add_argument('--approx_grad', type=int, default=1, help='Always set to 1')
    parser.add_argument('--grad_m', type=int, default=1, help='Number of steps to approximate the gradients')  # 1
    parser.add_argument('--grad_epsilon', type=float, default=1e-3)

    parser.add_argument('--forward_differences', type=int, default=1, help='Always set to 1')

    # Eigenvalues computation parameters
    parser.add_argument('--no_logits', type=int, default=1)
    parser.add_argument('--logit_correction', type=str, default='mean', choices=['none', 'mean'])
    parser.add_argument('--rec_grad_norm', type=int, default=1)
    parser.add_argument('--store_checkpoints', type=int, default=1)

    #############machine unlearning related##########################
    parser.add_argument("-net", type=str, default='ResNet18', help="net type")
    parser.add_argument("-student_net", type=str, default='ResNet18', help="net type")
    parser.add_argument(
        "-weight_path",
        type=str,
        default="./log_files/model/pretrain/ResNet18-Cifar20-15/best.pth",
        help="Path to model weights. If you need to train a new model use pretrain_model.py",
    )
    parser.add_argument(
        "-dataset",
        type=str,
        default="Cifar20",
        nargs="?",
        choices=["Cifar10", "Cifar19", "Cifar20", "Cifar100", "PinsFaceRecognition"],
        help="dataset to train on",
    )
    parser.add_argument("-classes", type=int, default=19, help="number of classes")
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
        "-forget_class",
        type=str,
        default="veg",  # 4
        nargs="?",
        help="class to forget",
        # choices=list(config.class_dict),
    )

    parser.add_argument(
        "--mia_mu_method",
        type=str,
        default="mia_mu_relearning",
        nargs="?",
        help="select unlearning method from choice set",
    )  # not to use: "UNSIR", "ssd_tuning"

    parser.add_argument(
        "-epochs", type=int, default=1, help="number of epochs of unlearning method to use"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for runs")

    #############masked related##########################
    parser.add_argument("--masked_path", default=None, help="the path where masks are saved")
    parser.add_argument("--para1", type=str, default=None)
    parser.add_argument("--para2", type=str, default=None)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    start=time.time()
    args.query_budget = 50
    args.batch_size = 128#
    args.attack_set = 'cifar_dfme'
    args.MAZE = False  # 4
    main_runner()
    end = time.time()
    print("elaplsed time", end - start)
