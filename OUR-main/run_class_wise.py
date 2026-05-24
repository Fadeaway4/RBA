import subprocess
import time

import config
import os

from datasets import Cifar100

os.chdir(os.path.dirname(os.path.abspath(__file__)))

delay_seconds = 20

DEVICE = '1'
seed = '2'
dataset = 'Cifar20'#'Cifar20'#'Cifar10'

if dataset == 'Cifar20':
    net = 'ViT'
    n_classes = '15'
    total_classes = '20'
    masked = config.CHECKPOINT_PATH+'/pretrain/ViT-Cifar20-15/best.pth'
    # masked = config.CHECKPOINT_PATH+'/pretrain/ResNet18-Cifar20-15/best.pth'
    forget_class = '4'
    salun_save_path = './mask/cifar20_vit_fullclass/'
elif dataset == 'Cifar100':
    net = 'ResNet18'
    n_classes = '95'
    total_classes = '100'
    masked = config.CHECKPOINT_PATH + '/pretrain/ResNet18-Cifar100-95/best.pth'
    forget_class =  '0' #TODO try: 0 TODO original: 4
    salun_save_path = './mask/cifar100_ResNet18_fullclass/'

python_file = 'saliency_mu/generate_mask_fullclass.py'
subprocess.call(["python", python_file, '--net', net,
                    '--dataset', dataset,
                    '--classes', n_classes,
                    '--num_class', total_classes,
                    '--forget_class', forget_class,
                    '--mask', masked,
                    '--save_dir', salun_save_path,
                    '--seed', seed]) # '-strategy', strategy
time.sleep(delay_seconds)

"""machine unlearning baseline"""
python_file ="forget_full_class_main.py"
# ViT-CIFAR20 Hyperparameters
mu_method_list = [
                 # ['baseline', '0.0005', '150'],
                 #  ['retrain', '0.0003', '150'],
                 # ['finetune', '4e-4', '10'],#,
                 # ['negative_grad', '2e-5', '4'],
                #  ['amnesiac', '0.0001', '3'],
                #  ['Wfisher', '10000','0'],
                # ['boundary_shrink', '0.4', '10'],
                # ['scrub', '0.0006', '9'],
                # ['FT_prune', '0.0001', '10'],
                # ['salun', '0.00028', '5'],
                # ['sfron', '1e-5', '6'],
                # ['rum', '2e-4', '0.0003'],
                ['orthogonality', '8e-4', '4e-4']
                  ]

#ResNet Cifar100 Hyperparameters
# mu_method_list = [#['retrain', '0.1', '50'],
#                  ['finetune', '0.01', '10'],
#                  ['amnesiac', '0.0001', '3'],
#                  ['negative_grad', '0.005', '5'],
#                   ['Wfisher', '0.005','0'],
#                   ['FT_prune', '0.005', '10'],
#                   ['boundary_shrink', '0.01', '10'],
#                   ['scrub', '0.0001', '9'],
#                   ['sfron', '0.001', '8'],
#                   ['salun', '0.0004', '3'],
#                   ['rum', '0.01', '0.0003'],
#                  #  ['orthogonality', '0.0012', '0.005']
#                   ]
for forget_class in [forget_class]:
    for mu_method in mu_method_list:
        for mu_method, para1, para2 in mu_method_list:
            subprocess.call(["python", python_file, '-net', net,
                             '-dataset', dataset, '-classes', n_classes,
                             '-num_classes', total_classes,
                             '-method', mu_method, '--forget_class', forget_class, '-weight_path', masked,
                             '--para1', para1,
                             '--para2', para2,
                             '-seed', seed,
                             '--mask_path', salun_save_path+'/with_0.5.pt'])
            time.sleep(delay_seconds)

"""relearning rea attack against class-wise unlearning"""
python_file = 'rea.py'
unlearn_data_percent = '0.03'#help: available unlearned data proportion in rea. setups:'0.20' for CIFAR100, '0.03' for CIFAR10
for forget_class in [forget_class]:
        for mu_method, para1, para2 in mu_method_list:
                subprocess.call(["python", python_file,
                                 '-net', net,
                                 '-dataset', dataset,
                                 '-classes', n_classes,
                                 '-num_classes', total_classes,
                                 '-method', mu_method,
                                 '--unlearn_data_percent', unlearn_data_percent,
                                 '--forget_class', forget_class,
                                 '-weight_path', masked,
                                 '-seed', seed,
                                 '-mia_mu_method', 'rea',
                                 '--para1', para1,
                                 '--para2', para2,
                                 ])
                time.sleep(delay_seconds)
