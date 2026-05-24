import argparse
import copy
import logging
import os
import random
import time
from datetime import datetime
from importlib import import_module

import numpy as np
import torch
from fastargs import Section, Param, get_current_config
from fastargs.decorators import param
from fastargs.validation import OneOf
from torch import optim
from tqdm import tqdm

from evaluation_utlis import matching_score_genimage_id
from my_datasets import people_idx


class DiffAtk_Main:
    def __init__(self, args, dataset, instance_data_dir, cand_idx) -> None:
        self.args = args
        self.dataset = dataset

        self.setup_seed(args.seed)
        self.init_task(args.task)
        self.init_attacker(args.attacker)
        self.init_logger(args.logger)
        self.run(instance_data_dir, cand_idx)

    def setup_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def init_task(self, task):
        kwargs = {
            "concept": self.args.concept,
            "sampling_step_num": self.args.sampling_step_num,
            "model_id": self.args.model_id,
            "model_name_or_path": self.args.model_name_or_path,
            "criterion": self.args.criterion,
            "cache_path": self.args.cache_path,
        }
        self.task = import_module(f'tasks.{task}_').get(**kwargs)
        self.task.replace_dataset(self.dataset)

    def init_attacker(self, attacker):
        attacker_keys = ['insertion_location',
                         'k',
                         'iteration',
                         'seed_iteration',
                         'eval_seed',
                         'attack_idx',
                         'cache_path']
        kwargs = {key: getattr(self.args, key) for key in attacker_keys}
        self.attacker = import_module(f'attackers.{attacker}_').get(**kwargs)

    def init_logger(self, logger):
        kwargs = {
            "name": self.args.name,
            "log_root": self.args.log_root,
        }
        self.logger = import_module(f'loggers.{logger}_').get(**kwargs)

    def run(self, instance_data_dir, cand_idx):
        self.attacker.run(self.task, self.logger, instance_data_dir, cand_idx)

def get_args(forget_prompt, moddel_id, model_saved_path, cache_path):
    parser = argparse.ArgumentParser(description="General Task & Attacker Configs")

    # ========== Overall Configs ==========
    parser.add_argument('--task', type=str, default='sd',
                        help='Task type to attack')

    parser.add_argument('--attacker', type=str, choices=['gcg', 'text_grad', 'hard_prompt',
                                                         'hard_prompt_multi', 'random',
                                                         'seed_search', 'no_attack'],
                        default='text_grad', help='Attack algorithm')

    parser.add_argument('--logger', type=str, choices=['json', 'none'],
                        default='none', help='Logger to use')

    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    # ========== Task Section ==========
    parser.add_argument('--concept', type=str, default=forget_prompt,
                        help='Concept to attack')

    parser.add_argument('--model_id', type=str, default=moddel_id,
                        help='model id to download')

    parser.add_argument('--model_name_or_path', type=str, default=model_saved_path,
                        help='model saved path')

    parser.add_argument('--criterion', type=str, default='l2', help='Loss criterion')

    parser.add_argument('--sampling_step_num', type=int, default=50,
                        help='Sampling step number')

    parser.add_argument('--cache_path', type=str, default=cache_path, help='cache path of adv_attack')
    # ========== Attacker Section ==========
    parser.add_argument('--insertion_location', type=str, choices=['prefix_k', 'suffix_k',
                                                                   'mid_k', 'insert_k',
                                                                   'per_k_words'],
                        default='prefix_k', help='Insertion location')

    parser.add_argument('--k', type=int, default=3, help='k in insertion_location')
    parser.add_argument('--iteration', type=int, default=10, help='Number of attacks iterations')
    parser.add_argument('--seed_iteration', type=int, default=20, help='Number of seed iterations')
    parser.add_argument('--eval_seed', type=int, default=0, help='Evaluation seed')
    parser.add_argument('--attack_idx', type=int, default=0, help='Attack index')

    # ========== Text Gradient Section ==========
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (Only used if attacker is text_grad)')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay (Only used if attacker is text_grad)')

    # ========== Logger Section ==========
    parser.add_argument('--name', type=str,
                        default=datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f'),
                        help='Name of this run')

    # ========== JSON Logger Section ==========
    parser.add_argument('--log_root', type=str, default='log_outputs/CelebA/attack_logs',
                        help='Path to log folder (Only used if logger is json)')

    # 解析命令行参数
    args = parser.parse_args()
    return args


def eval_process(pipeline, input_prompt, save_path, epoch=0, figure_num=5):
    # TODO
    # prompt_list = ['a photo of Mads Mikkelsen']
    if input_prompt is not None:
        for i in range(figure_num):
            with torch.no_grad():
                image = pipeline(input_prompt,
                                 num_inference_steps=50,
                                 guidance_scale=7.5,  # classifier-free guidance
                                 ).images[0]
            # image.show()
            image.save(save_path+f"/{epoch}_{input_prompt}_{i}.png")


def relearning_process(pipeline, train_dataloader, noise_scheduler, input_prompt, lr,
                       save_path,
                       instance_data_dir):
    unet = pipeline.unet
    # 超参
    epochs = 100
    learning_rate = lr#5e-6#1e-5
    scaling_factor = pipeline.vae.config.scaling_factor
    optimizer = optim.AdamW(unet.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        unet.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            latent_dist = batch[0][0]
            latents = latent_dist.sample()
            latents = latents * scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = pipeline.text_encoder(batch[0][1])[0]

            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            # 6. 损失：MSE(预测噪声 vs 实际噪声)
            loss = torch.nn.functional.mse_loss(model_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"Epoch:{epoch}, Step:{step}, Loss:{loss.item():.4f}")

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
        if epoch % 10 == 0:
            unet.eval()

            figure_num=5
            eval_process(pipeline, input_prompt, save_path, epoch, figure_num=figure_num)

            # calculate ISM for forget persons
            image_name_list = [f'{epoch}_{input_prompt}_{i}.png' for i in range(figure_num)]
            print("instance_data_dir", instance_data_dir)
            ism_inferred = matching_score_genimage_id(save_path, [instance_data_dir], image_name_list)
            print("[inferred] ISM and FDR are {}".format(ism_inferred))
            logging.info(f'Epoch: {epoch}\t Infer-ISM {ism_inferred}')

def mia_attack(method, candidate_dataset_list, forget_prompt, model_id, model_saved_path, noise_scheduler,
               tokenizer, device, forget_index=0, ood_index=1, **kwargs):
    #method = ['Relearning', 'UnlearnDiffAtk']
    image_saved_path = kwargs.get("image_saved_path")
    instance_data_dir_list = [f"./data/CelebA-15/{people_idx[forget_index]}/set_A",
                              f"./data/CelebA-15/{people_idx[ood_index]}/set_A"]
    if method == 'Relearning':
        lr=5e-6
        pipeline = kwargs.get("pipeline")
        input_prompt_list = kwargs.get("input_prompt_list")
        for idx, candidate_dataset in enumerate(candidate_dataset_list[:1]):
            # TODO input_prompt = input_prompt_list[idx]
            input_prompt = input_prompt_list[idx]
            pipeline = copy.deepcopy(kwargs.get("pipeline"))
            candidate_dataloader = torch.utils.data.DataLoader(candidate_dataset,
                                                               batch_size=2, collate_fn=lambda x: x, shuffle=True)

            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            timestamp = time.strftime("%m%d_%H%M%S")
            log_path =  f'{image_saved_path}/log'
            os.makedirs(log_path, exist_ok=True)
            log_file_path = f'{log_path}/usim_cand{idx}_{timestamp}_{lr}.log'
            print("log_file_path", log_file_path)
            logging.basicConfig(
                filename=log_file_path,
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            relearning_process(pipeline, candidate_dataloader, noise_scheduler, input_prompt, lr,
                               image_saved_path, instance_data_dir_list[idx])

    elif method == 'UnlearnDiffAtk':
        print("image_saved_path", image_saved_path)
        args=get_args(forget_prompt, model_id, model_saved_path, cache_path=image_saved_path)
        #setup seed
        for idx, candidate in enumerate(candidate_dataset_list):
            DiffAtk_Main(args, dataset=candidate, instance_data_dir=instance_data_dir_list[idx], cand_idx=idx)

