# Relearning Behavior Alignment (RBA) Framework

This repository provides the official PyTorch implementation of **RBA (Relearning Behavior Alignment)**, a dynamic evaluation and auditing framework for class-level machine unlearning.

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Fadeaway4/RBA.git
cd RBA
pip install -r requirements.txt
```

---

## 📖 Workflow Overview

The overall pipeline consists of three stages:

1. **Pretraining**  
   Train a target model on benchmark datasets such as CIFAR10, CIFAR20, and CIFAR100.

2. **Class-level Unlearning**  
   Apply different approximate unlearning methods to remove the influence of a target class.

3. **RBA Evaluation**  
   Perform dynamic relearning-based evaluation to analyze residual forgetting behaviors and privacy leakage.

> Please refer to the scripts in the root directory for detailed configurations and command-line arguments.

---

# 1️⃣ Pretraining

Train the target model before performing unlearning experiments.

## Example

```bash
python pretrain_model_class_wise.py \
    --dataset Cifar20 \
    --net resnet18 \
    --classes 20
```

---

# 2️⃣ Class-level Machine Unlearning



### Step 1. Saliency Mask Generation (Optional)

Some methods (e.g., `salun`) require a saliency mask before unlearning.

```python
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
```

---

### Step 2. Approximate Unlearning

Run different class-level unlearning baselines:

- `baseline`
- `retrain`
- `bad_t`
- `salun`
- `amnesiac`
- `L_codec`
- `unrolling`
- `scrub`

```python
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
```

---

### Step 3. RBA Evaluation

Evaluate residual forgetting behaviors using the proposed RBA framework.

```python
python_file = 'rba.py'

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
        '--para1', para1,
        '--para2', para2
    ])
```

---

## 📊 Supported Datasets

- CIFAR10
- CIFAR20
- CIFAR100

---

## 🧩 Supported Unlearning Methods

The framework currently supports the following class-level machine unlearning baselines:

| Methods |`retrain` ， `salun`, `bad_t` ， `amnesiac`, `L_codec` ， `unrolling`, `scrub` |

---

## 📌 Notes

- Different methods may require different hyperparameter settings.
- For `salun`, the saliency mask must be generated before unlearning.
- Experimental configurations can be modified directly in the corresponding scripts.

---

## 🙏 Acknowledgements

Our implementation is built upon and inspired by the repository **orthogonalunlearning-replay/OUR**. We sincerely thank the authors for their valuable contribution to the machine unlearning community, which provided an important foundation for this work.