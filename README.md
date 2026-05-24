# Orthogonal Unlearning and Replay

Welcome to the official repository for the paper **"Reminiscence Attack on Residuals: Exploiting Approximate Machine Unlearning for Privacy"**. This research explores a critical vulnerability in approximate unlearning algorithms and introduces the **Reminiscence Attack (ReA)**, which specifically targets the membership privacy of data that has been “unlearned” (removed) from a model.

In this repository, you will find:

1. **Reminiscence Attack (ReA)**  
   A method to exploit residual information in models that have performed approximate machine unlearning, potentially recovering membership information of the data supposed to be removed.

2. **Orthogonal Unlearning and Replay**  
   A approximate unlearning strategy to more effectively “forget” targeted data, aiming to mitigate privacy risks in approximate unlearning algorithms.

---

## Table of Contents
- [Environment Setup](#environment-setup)
- [Quick Start](#quick-start)
  - [1) Pretraining](#1-pretraining)
  - [2) Perform Approximate Learning](#2-perform-approximate-learning)
    - [Class-wise Unlearning](#class-wise-unlearning)
    - [Sample-wise Unlearning](#sample-wise-unlearning)
- [Reminiscence Attack (ReA)](#reminiscence-attack-rea)
  - [Class-wise ReA](#class-wise-rea)
  - [Sample-wise ReA](#sample-wise-rea)
- [Membership Inference Attack (MIA) with LIRA](#membership-inference-attack-mia-with-lira)

---
## ! Project Update Notice

**Active Development Notice:** The folder ./celeba_sd updates an implementation of a **text-to-image diffusion model**.

## Environment Setup

1. **Python Version:** Use Python 3.8+ (or higher) for better compatibility.
2. **Dependencies:** Install required packages with:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1) Pretraining
Before testing approximate unlearning, you need a normally trained (or pretrained) model. We provide two types of pretraining:

Class-wise Pretraining

```bash
python pretrain_model_class_wise.py --dataset Cifar20 --net ViT --classes 15
```

Trains a model with only certain classes for subsequent “forgetting” of specific classes.

Sample-wise Pretraining

```bash
python pretrain_model_sample_wise.py --dataset Cifar10 --net ResNet18 --classes 10
```
Trains a model with all classes but allows for the forgetting of specific samples.

---

### 2) Perform Approximate Learning

After pretraining, you can run one of the two main scripts below, depending on whether you want class-wise or sample-wise unlearning.

#### Class-wise Unlearning

- **Main Script:** `run_class_wise.py`

This script typically involves the following steps:

1. **Saliency Mask Generation (optional)** if the unlearning method (e.g., `salun`) requires it.  
2. **Approximate Unlearning** using different methods (`baseline`, `retrain`, `finetune`, `negative_grad`, `amnesiac`, `Wfisher`, `boundary_shrink`, `scrub`, `FT_prune`, `salun`, `fsron`, `rum`, `orthogonality`, etc.).  
3. **Reminiscence Attack** to evaluate how much private information remains after unlearning.

A simplified example:
```bash
python run_class_wise.py
```
Key code sections:
```python
# 1. Generate Saliency Mask
python_file = 'saliency_mu/generate_mask_fullclass.py'
subprocess.call([
    "python", python_file,
    '--net', net,
    '--dataset', dataset,
    '--classes', n_classes,
    '--num_class', total_classes,
    '--forget_class', forget_class,
    '--mask', masked,
    '--save_dir', salun_save_path,
    '--seed', seed
])

# 2. Perform approximate unlearning
python_file = "forget_full_class_main.py"
for mu_method, para1, para2 in mu_method_list:
    subprocess.call([
        "python", python_file,
        '-net', net,
        '-dataset', dataset,
        '-classes', n_classes,
        '-num_classes', total_classes,
        '-method', mu_method,
        '--forget_class', forget_class,
        '-weight_path', masked,
        '--para1', para1,
        '--para2', para2,
        '-seed', seed,
        '--mask_path', salun_save_path + '/with_0.5.pt'
    ])

# 3. Reminiscence Attack (ReA)
python_file = 'rea.py'
for mu_method, para1, para2 in mu_method_list:
    subprocess.call([
        "python", python_file,
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
        '--para2', para2
    ])
```

#### Sample-wise Unlearning

- **Main Script:** `run_sample_wise.py`

Similar to the class-wise script, but focuses on forgetting individual samples:

1. **Saliency Mask Generation** (optional).  
2. **Approximate Unlearning** methods such as `finetune`, `negative_grad`, `relabel`, `Wfisher`, `FisherForgetting`, `scrub`, `FT_prune`, `salun`, `sfron`, `rum`, `orthogonality`, etc.  
3. **Reminiscence Attack** for sample-wise scenarios.

A simplified example:
```bash
python run_sample_wise.py
```
Key code sections:
```python
# 1. Generate Saliency Mask
python_file = 'saliency_mu/generate_mask.py'
subprocess.call([
    "python", python_file,
    '--net', net,
    '--dataset', dataset,
    '--classes', n_classes,
    '--mask', weight_path,
    '--save_dir', salun_save_path,
    '--seed', seed
])

# 2. Perform approximate unlearning
python_file = "forget_sample_main.py"
for mu_method, para1, para2 in mu_method_list:
    subprocess.call([
        "python", python_file,
        '-net', net,
        '-dataset', dataset,
        '-classes', n_classes,
        '-method', mu_method,
        '-weight_path', weight_path,
        '--para1', para1,
        '--para2', para2,
        '-seed', seed,
        '--mask_path', config.CHECKPOINT_PATH + f'{salun_save_path}/with_0.5.pt'
    ])
```

---

## Reminiscence Attack (ReA)

### Class-wise ReA
In `run_class_wise.py`, we run `rea.py` to test how effectively a class was “forgotten.” The attack method attempts to detect whether the model retains class-specific information.

### Sample-wise ReA
In `run_sample_wise.py`, we run `rea_reminiscence_random.py` + `mia_lira.py` to evaluate whether individual samples leave identifiable traces in the model.

---

## Membership Inference Attack (MIA) with LIRA

We also provide an optional **Membership Inference Attack** using **LIRA** to further assess privacy risks.

1. **Reminiscence Attack (ReA) for sample-wise**
```bash
  python_file = "rea_reminiscence_random.py"
  for mu_method, para1, para2 in mu_method_list:
      subprocess.call([
          "python", python_file,
          '-net', net,
          '-dataset', dataset,
          '-classes', n_classes,
          '-method', mu_method,
          '-weight_path', weight_path,
          '--para1', para1,
          '--para2', para2,
          '-seed', seed
      ])
```

2.  **Shadow Model Training** (`train.py`):
   ```bash
   python train.py \
       --name <model_name> \
       --dataset <dataset_name> \
       --save_name <save_name> \
       --classes <n_classes> \
       --net <net_name> \
       --opt <opt_method> \
       --bs 128 \
       --lr <learning_rate> \
       --pkeep 0.83 \
       --num_shadow 8 \
       --shadow_id <id>
   ```

3. **MIA with LIRA** (`mia_lira.py`):
   ```bash
   python mia_lira.py \
       --net <net_name> \
       --dataset <dataset_name> \
       --classes <n_classes> \
       --name <model_name> \
       --save_name <save_name> \
       --num_shadow 8 \
       --num_aug 10 \
       --machine_unlearning <mu_method> \
       --para1 <para1> \
       --para2 <para2> \
       -task rea \
       --weight_path <path_to_pretrained_model> \
       --start 40000 \
       --end 60000
   ```
Use `-task mia_lira` if you wish to run mia_lira only in the same evaluation pipeline.

---




