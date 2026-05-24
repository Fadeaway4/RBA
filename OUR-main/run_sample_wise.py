import subprocess
import time

import config
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

delay_seconds = 20

DEVICE = '1'
seed = '1'


dataset ='Cifar20'#'Cifar10'

if dataset == 'Cifar10':
    net = 'ResNet18'
    n_classes = '10'
    weight_path = config.CHECKPOINT_PATH+'/pretrain/ResNet18-Cifar10-10/best.pth'
    salun_save_path = config.CHECKPOINT_PATH + '/mask/resnet18_cifar10/'

elif dataset == 'Cifar20':
    net = 'ViT'
    n_classes = '20'
    weight_path = config.CHECKPOINT_PATH+'/pretrain/ViT-Cifar20-20/best.pth'
    salun_save_path = config.CHECKPOINT_PATH + '/mask/vit_cifar20/'


"""run for salun mask generate"""
python_file = 'saliency_mu/generate_mask.py'
subprocess.call(["python", python_file, '--net', net,
                     '--dataset', dataset, '--classes', n_classes,
                      '--mask', weight_path,
                      '--save_dir', salun_save_path,
                     '--seed', seed]) # '-strategy', strategy
time.sleep(delay_seconds)

"""machine unlearning baseline"""
python_file ="forget_sample_main.py"
#Resnet18-CIFAR10 Hyperparameters
mu_method_list = [
#                   #['retrain', '0.1', '150'],#['rum', '0.12', '0.0005'],
#                   ['finetune', '0.11', '10'],
#                   ['negative_grad', '0.04', '8'],
#                   ['relabel', '0.00065', '3'],
#                   ['Wfisher', '130', '0'],
#                   ['FisherForgetting', '7e-8', '0'],
#                   ['scrub', '0.0004', '7'],
#                   ['FT_prune', '0.008', '10'],
#                   ['salun', '0.00065', '3'],
#                   ['sfron', '0.035', '8'],
#                   ['rum', '0.12', '0.0005'],#TODO memorization file is needed
                  ['orthogonality', '0.0012', '0.04']
                  ]

#Vit CIFAR20 Hyperparameters
# mu_method_list = [
                  # ['retrain', '0.0003', '150'],
    #               ['FisherForgetting', '1e-8', '0'],
    #               ['finetune', '1.2e-3', '10'],#,
    #               ['negative_grad', '1.5e-3', '15'],
    #                 ['relabel', '8.5e-4', '8'],
    #               ['Wfisher', '40', '0'],
    #               ['Wfisher', '45', '0'],
    #               ['scrub', '4e-4', '10'],
    #               ['FT_prune', '1e-3', '10'],
    #               ['salun', '1.2e-3', '15'],
    #               ['sfron', '1.1e-4', '6'],
    #               ['rum', '3e-3', '1e-3'],#TODO memorization is needed
    #               ['m_orthogonality', '3e-4', '5e-4'],
    #               ]

for mu_method, para1, para2 in mu_method_list:
     subprocess.call(["python", python_file, '-net', net,
                      '-dataset', dataset, '-classes', n_classes,
                      '-method', mu_method, '-weight_path', weight_path,
                      '--para1', para1, '--para2', para2,
                      '-seed', seed,
                      '--mask_path', config.CHECKPOINT_PATH+f'{salun_save_path}/with_0.5.pt'])
     time.sleep(delay_seconds)

"""perform (ReA) reminiscence phase """
python_file = "rea_reminiscence_random.py"
for mu_method, para1, para2 in mu_method_list:
    subprocess.call(["python", python_file, '-net', net,
                          '-dataset', dataset, '-classes', n_classes,
                          '-method', mu_method,
                          '-weight_path', weight_path,
                          '--para1', para1, '--para2', para2,
                          '-seed', seed
                          ])
    time.sleep(delay_seconds)

"""train shadow models for lira [Run it once only]"""
if dataset == 'Cifar10':
    opt = 'sgd',
    lr = '1e-1'
elif dataset == 'Cifar20':
    opt = 'adam'
    lr = '3e-4'
for id in range(8):
    subprocess.call([
        "python", "train.py",
        "--name", f"{net}-{dataset}-{n_classes}",
        "--dataset", dataset,
        "--save_name", net,
        "--classes", n_classes,
        "--net", net,
        "--opt", opt,
        "--bs", "128",
        "--lr", lr,
        "--pkeep", "0.83",
        "--num_shadow", "8",
        "--shadow_id", str(id)
    ])

"""perform lira mia process"""
for task in ['mia_lira', 'rea']: #'mia_lira' performs lira only
    for mu_method, para1, para2 in mu_method_list:
        print(f'[{task}]:', "mu_method", mu_method, "-para1", para1, "-para2", para2)
        subprocess.call([
            "python", 'mia_lira.py',
            "--net", net,
            "--dataset", dataset,
            "--classes", n_classes,
            "--name", 'ViT-Cifar20-20',
            "--save_name", 'ViT',
            "--num_shadow", "8",
            "--num_aug", '10',  # used to determine the number of aug data
            "--machine_unlearning", mu_method,
            "--para1", para1,
            "--para2", para2,
            '-task', task,
            "--weight_path", weight_path,
            "--start", '40000',
            "--end", '60000'
        ])
        time.sleep(delay_seconds)