# main.py

A unified pipeline for DreamBooth training, unlearning (including OUR unlearning methods), and membership inference attacks (including ReA attacks) on the CelebA-HQ dataset.

## Table of Contents

* [Overview](#overview)
* [Prerequisites](#prerequisites)
* [Dataset Preparation](#dataset-preparation)
* [Directory Structure](#directory-structure)
* [Usage](#usage)
* [Configuration Flags](#configuration-flags)

## Overview

`main.py` includes four stages:

1. Fine-tune a Stable Diffusion model (DreamBooth)
2. Evaluate the fine-tuned model
3. Unlearn specific instances
4. Evaluate unlearning and perform membership inference attacks


## Prerequisites

* Python 3.8+
* PyTorch
* [Hugging Face ](https://github.com/huggingface/diffusers)[`diffusers`](https://github.com/huggingface/diffusers)[ and ](https://github.com/huggingface/diffusers)[`transformers`](https://github.com/huggingface/diffusers)
* Other dependencies listed in `requirements.txt`


## Dataset Preparation

This project reuses the CelebA-HQ setup from the [Anti-DreamBooth repository](https://github.com/VinAIResearch/Anti-DreamBooth).

1. Clone and prepare CelebA-HQ as described:

   ```bash
   ```

git clone [https://github.com/VinAIResearch/Anti-DreamBooth.git](https://github.com/VinAIResearch/Anti-DreamBooth.git) cd Anti-DreamBooth


# follow their instructions to generate `data/CelebA-HQ` folder

````
2. Ensure your local `data/` matches the structure expected by `main.py`:
```text
data/
└── CelebA-15/
    ├── <person_id_1>/
    │   ├── set_A/
    │   └── set_B/
    ├── <person_id_2>/
    │   └── ...
    └── ...
````

## Model Preparation

Download the unlearned model (Salun-OUR) from <[DOWNLOAD_URL](https://drive.google.com/drive/folders/1IGPhyLhbSHJbhtgf725swKPIsz2VjO2j?usp=drive_link)> and place it under ./unlearned_model_2_classes.

In this way, only perform the membership inference attack stage can reproduce the results.

## Usage

Run the pipeline with optional stages enabled via flags:

```bash
python main.py
```

All flags default to `False` except membership inference (MIA).

## Configuration Flags

Within `main.py`, toggle these boolean flags:

* `perform_train` — Train DreamBooth model
* `perform_train_evaluate` — Evaluate trained model
* `perform_unlearn` — Execute unlearning methods
* `perform_unlearning_evaluate` — Evaluate unlearned models
* `perform_mia` — Run membership inference attacks (default: True)


