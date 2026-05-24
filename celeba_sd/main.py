import argparse
import hashlib
import itertools
import logging
import os
import time
import warnings

import cv2
import numpy as np
import torch
from PIL import Image

from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from pytorch_fid import fid_score
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel

import models
from celeba_sd.my_datasets.my_datasets import LatentsDataset, DreamBoothDataset, names_list, people_idx, ExplicitPromptDataset
from torch import optim
from tqdm import tqdm
from pathlib import Path

from celeba_sd.mia_attack import mia_attack
from celeba_sd.unlearning_method import unlearning
import torch.nn.functional as F
from diffusers.optimization import get_scheduler

from evaluations.FaceImageQuality.face_image_quality import SER_FIQ
from evaluations.ism_fdfr import matching_score_genimage_id


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch

def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


def train_process(pipeline, train_dataloader, noise_scheduler, saved_path):
    unet = pipeline.unet
    text_encoder = pipeline.text_encoder
    unet.train()

    for param in unet.parameters():
        param.requires_grad = True

    if args.train_text_encoder:
        text_encoder.train()
        for param in text_encoder.parameters():
            param.requires_grad = True

    scaling_factor = pipeline.vae.config.scaling_factor
    # optimizer = optim.AdamW(unet.parameters(), lr=learning_rate)
    # Optimizer creation
    params_to_optimize = (
        itertools.chain(unet.parameters(),
                        text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    for epoch in range(args.num_train_epochs):
        global_step = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            global_step += 1
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

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            if args.with_prior_preservation:
                # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)

                # Compute instance loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Compute prior loss
                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                # Add the prior loss to the instance loss.
                loss = loss + args.prior_loss_weight * prior_loss
            else:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # loss = torch.nn.functional.mse_loss(model_pred, noise)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Epoch:{epoch}, Step:{step}, Loss:{loss.item():.4f}")

            if global_step >= args.max_train_steps:
                break

    # unet.save_pretrained(saved_path)
    # text_encoder.save_pretrained(saved_path)
    # TODO:
    if False:
        unet.eval()
        pipeline.unet = unet
        unet.save_pretrained(saved_path+'/unet')
        if args.train_text_encoder:
            text_encoder.eval()
            pipeline.text_encoder = text_encoder
            text_encoder.save_pretrained(saved_path+'/text_encoder')

def retrain_process(model_id, forget_index, retain_index, saved_path, weight_dtype=torch.float32):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # diffuser pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=weight_dtype, ).to("cuda")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)

    pipeline.text_encoder = text_encoder
    pipeline.tokenizer = tokenizer
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    if not args.train_text_encoder:
        freeze_parameters(pipeline.text_encoder)  # we train the text_encoder
    freeze_parameters(pipeline.vae)

    instance_prompt_list = [f"a photo of {names_list[people_idx[forget_index]]} person",
                            f"a photo of {names_list[people_idx[retain_index]]} person"]
    instance_data_dir_list = [f"./data/CelebA-15/{people_idx[forget_index]}/set_A",
                              f"./data/CelebA-15/{people_idx[retain_index]}/set_A"]

    train_dataset = DreamBoothDataset(
        instance_data_root_list=[instance_data_dir_list[1]],
        instance_prompt_list=[instance_prompt_list[1]],
        class_data_root=args.class_data_dir,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation), )

    latents_cache = []
    text_encoder_cache = []
    for batch in tqdm(train_dataloader, desc="Caching latents"):
        with torch.no_grad():
            batch["pixel_values"] = batch["pixel_values"].to(device, non_blocking=True, dtype=weight_dtype)
            batch["input_ids"] = batch["input_ids"].to(device, non_blocking=True)
            latents_cache.append(pipeline.vae.encode(batch["pixel_values"]).latent_dist)
            text_encoder_cache.append((batch["input_ids"]))

    train_dataset = LatentsDataset(latents_cache, text_encoder_cache)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, collate_fn=lambda x: x, shuffle=True)
    train_process(pipeline, train_dataloader, noise_scheduler, saved_path)

def eval_process(pipeline, instance_prompt_list, input_prompt=None):
    pipeline.unet.eval()
    pipeline.text_encoder.eval()
    # TODO
    # prompt_list = ['a photo of Mads Mikkelsen']
    if input_prompt is not None:
        with torch.no_grad():
            image = pipeline(input_prompt,
                             num_inference_steps=50,
                             guidance_scale=7.5,  # classifier-free guidance
                             ).images[0]
        # image.show()
        image.save(f"{input_prompt}.png")
    else:
        for prompt in instance_prompt_list:
            with torch.no_grad():
                image = pipeline(prompt,
                                 num_inference_steps=50,
                                 guidance_scale=7.5,  # classifier-free guidance
                                 ).images[0]
            # image.show()
            image.save(f"{prompt}.png")


def generate_forget_retain_dataloader(pipeline, tokenizer, forget_index, retain_index, class_data_dir, class_prompt,
                                      resolution, with_prior_preservation, device, weight_dtype):
    instance_prompt_list = [f"{names_list[people_idx[forget_index]]}"
                            ]
    instance_data_dir_list=[f"./data/CelebA-15/{people_idx[forget_index]}/set_A"
                            ]
    train_dataset = DreamBoothDataset(
        instance_data_root_list=instance_data_dir_list,
        instance_prompt_list=instance_prompt_list,
        class_data_root= class_data_dir,
        class_prompt=class_prompt,
        tokenizer=tokenizer,
        size=resolution,
        center_crop=True,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=False,
        collate_fn=lambda examples: collate_fn(examples, with_prior_preservation),)

    latents_cache = []
    text_encoder_cache = []
    for batch in tqdm(train_dataloader, desc="Caching latents"):
        with torch.no_grad():
            batch["pixel_values"] = batch["pixel_values"].to(device, non_blocking=True, dtype=weight_dtype)
            batch["input_ids"] = batch["input_ids"].to(device, non_blocking=True)
            latents_cache.append(pipeline.vae.encode(batch["pixel_values"]).latent_dist)
            text_encoder_cache.append((batch["input_ids"]))

    forget_dataset = LatentsDataset(latents_cache, text_encoder_cache)
    forget_dataloader = torch.utils.data.DataLoader(forget_dataset, batch_size=2, collate_fn=lambda x: x, shuffle=True)

    instance_prompt_list = [f"{names_list[people_idx[retain_index]]}"
                            ]
    instance_data_dir_list=[f"./data/CelebA-15/{people_idx[retain_index]}/set_A"
                            ]
    train_dataset = DreamBoothDataset(
        instance_data_root_list=instance_data_dir_list,
        instance_prompt_list=instance_prompt_list,
        class_data_root= class_data_dir,
        class_prompt=class_prompt,
        tokenizer=tokenizer,
        size=resolution,
        center_crop=True,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=False,
        collate_fn=lambda examples: collate_fn(examples, with_prior_preservation),)

    latents_cache = []
    text_encoder_cache = []
    for batch in tqdm(train_dataloader, desc="Caching latents"):
        with torch.no_grad():
            batch["pixel_values"] = batch["pixel_values"].to(device, non_blocking=True, dtype=weight_dtype)
            batch["input_ids"] = batch["input_ids"].to(device, non_blocking=True)
            latents_cache.append(pipeline.vae.encode(batch["pixel_values"]).latent_dist)
            text_encoder_cache.append((batch["input_ids"]))

    retain_dataset = LatentsDataset(latents_cache, text_encoder_cache)
    retain_dataloader = torch.utils.data.DataLoader(retain_dataset, batch_size=2, collate_fn=lambda x: x, shuffle=True)
    return forget_dataloader, retain_dataloader

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default="./data/CelebA-15/class_image_dir",
        required=False,
        help="A folder containing the training data of class images.",
    )

    parser.add_argument(
        "--class_prompt",
        type=str,
        default="a photo of person",
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--inference_prompts",
        type=str,
        default="a photo of sks person;a DSLR portrait of sks person",
        help="The prompt used to generate images at inference.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=True,
        # action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=200,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    #TODO train the encoder
    parser.add_argument(
        "--train_text_encoder",
        # action="store_true",
        default=True,
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=2000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-7,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default="bf16",
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )

    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args

from brisque import BRISQUE
def evaluate_unlearned_results_sd(pipeline, instance_prompt_list, instance_data_dir_list, save_dir_list, forget_index=0, generate_images=True, device='cuda:0'):
    image_number = 5
    generated_images_forget_person = []
    generated_images_retain_person = []

    generate_images=generate_images
    if generate_images:
        for i in range(image_number):
            forget_person_image = pipeline(instance_prompt_list[forget_index],
                            num_inference_steps=50,
                            guidance_scale=7.5,  # classifier-free guidance
                            ).images[0]
            generated_images_forget_person.append(forget_person_image)
            forget_person_image.save(save_dir_list[forget_index] + f"/{i}.png")
            print("save forget_person_image in ", save_dir_list[forget_index] + f"/{i}.png")

            retain_person_image = pipeline(instance_prompt_list[1],
                                                   num_inference_steps=50,
                                                   guidance_scale=7.5,  # classifier-free guidance
                                                   ).images[0]
            generated_images_retain_person.append(retain_person_image)
            retain_person_image.save(save_dir_list[1] + f"/{i}.png")
            print("save retain_person_image in ", save_dir_list[1] + f"/{i}.png")

    #calculate ISM for forget persons
    ism_forget = matching_score_genimage_id(save_dir_list[forget_index], [instance_data_dir_list[forget_index]])
    print("[forget] ISM and FDR are {}".format(ism_forget))

    #calculate ISM for retain persons
    ism_retain = matching_score_genimage_id(save_dir_list[1], [instance_data_dir_list[1]])
    print("[retain] ISM and FDR are {}".format(ism_retain))

    #calcualte SER FIQ for retain persons
    # ser_fiq = SER_FIQ(gpu=0)
    retained_path = save_dir_list[1]
    # prompt_score = 0
    # count = 0
    # for img_name in os.listdir(retained_path):
    #     if "png" in img_name or "jpg" in img_name:
    #         img_path = os.path.join(retained_path, img_name)
    #         img = cv2.imread(img_path)
    #         aligned_img = ser_fiq.apply_mtcnn(img)
    #         if aligned_img is not None:
    #             score = ser_fiq.get_score(aligned_img, T=100)
    #             prompt_score += score
    #             count += 1
    # ser_fiq_output = prompt_score / count
    # print("FIQ score: {}".format(ser_fiq_output))

    #calculate FID
    fid_value = fid_score.calculate_fid_given_paths([instance_data_dir_list[1], retained_path], batch_size=50, dims=2048,
                                                    device='cuda', num_workers=4)
    print("fid_value: {}".format(fid_value))

    #calculate brisques for retain models
    obj = BRISQUE(url=False)
    prompt_score = 0
    count = 0
    for img_name in os.listdir(retained_path):
        print("img_name: ", img_name)
        if "png" in img_name or "jpg" in img_name:
            img_path = os.path.join(retained_path, img_name)
            img = cv2.imread(img_path)
            brisque_score = obj.score(img)
            print("brisque_score: ", brisque_score)
            if brisque_score > 0:
                prompt_score += brisque_score
                count += 1
    brisque_retain =  prompt_score / count
    print("The brisque score is {}".format(brisque_retain))

def main(args):
    weight_dtype = torch.float32
    # Flag
    perform_train = False
    perform_train_evaluate = False
    perform_unlearn = False
    perform_unlearning_evaluate = False
    perform_mia = True

    forget_index = 0
    retain_index = 2
    ood_index = 1

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    start_time_dreambooth = time.time()
    #get pipeline, tokenizer, scheduler
    model_id = "runwayml/stable-diffusion-v1-5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    # diffuser pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=weight_dtype,).to("cuda")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    pipeline.text_encoder = text_encoder
    pipeline.tokenizer = tokenizer
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    if not args.train_text_encoder:
        freeze_parameters(pipeline.text_encoder) #we train the text_encoder
    freeze_parameters(pipeline.vae)

    #prepare dataset
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            num_new_images = args.num_class_images - cur_class_images
            print(f"Number of class images to sample: {num_new_images}.")
            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=8)

            for example in tqdm(
                    sample_dataloader,
                    desc="Generating class images",
                    disable=True,
            ):
                images = pipeline(example["prompt"]).images
                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    instance_prompt_list = [f"a photo of {names_list[people_idx[forget_index]]} person",
                            f"a photo of {names_list[people_idx[retain_index]]} person"]
    instance_data_dir_list = [f"./data/CelebA-15/{people_idx[forget_index]}/set_A",
                              f"./data/CelebA-15/{people_idx[retain_index]}/set_A"]
    train_dataset = DreamBoothDataset(
        instance_data_root_list=instance_data_dir_list,
        instance_prompt_list=instance_prompt_list,
        class_data_root=args.class_data_dir,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation), )

    latents_cache = []
    text_encoder_cache = []
    for batch in tqdm(train_dataloader, desc="Caching latents"):
        with torch.no_grad():
            batch["pixel_values"] = batch["pixel_values"].to(device, non_blocking=True, dtype=weight_dtype)
            batch["input_ids"] = batch["input_ids"].to(device, non_blocking=True)
            latents_cache.append(pipeline.vae.encode(batch["pixel_values"]).latent_dist)
            text_encoder_cache.append((batch["input_ids"]))

    train_dataset = LatentsDataset(latents_cache, text_encoder_cache)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, collate_fn=lambda x: x, shuffle=True)

    # download from the pretrained_model
    pipeline.safety_checker = None

    """
        ======================================
        Train the target model with Dreambooth
        ======================================
        """
    pretrain_saved_path = "./model/sd_saved_unet_2_classes"
    if perform_train:
        train_process(pipeline, train_dataloader, noise_scheduler,
                      saved_path=pretrain_saved_path)
    end_time_dreambooth = time.time()
    print("Dreambooth Time:", end_time_dreambooth - start_time_dreambooth)

    """
        ======================================
        Evaluate the trained target model
        ======================================
        """
    if perform_train_evaluate:
        unet = UNet2DConditionModel.from_pretrained(pretrain_saved_path+'/unet').to(device)
        pipeline.unet = unet

        if args.train_text_encoder:
            text_encoder = CLIPTextModel.from_pretrained(pretrain_saved_path+'/text_encoder').to(device)
            pipeline.text_encoder = text_encoder
        eval_process(pipeline, instance_prompt_list)
        save_dir_list = [f"./data/CelebA-15/{people_idx[forget_index]}/set_g_target",
                         f"./data/CelebA-15/{people_idx[retain_index]}/set_g_target"]
        for save_dir in save_dir_list:
            os.makedirs(save_dir, exist_ok=True)
        evaluate_unlearned_results_sd(pipeline, instance_prompt_list, instance_data_dir_list, save_dir_list,
                                      forget_index=0, generate_images=True, device=device)

    unlearning_method_lists = ['Salun_Orth'] #'retrain', 'Salun','FSRon', 'Meta_unlearning', 'Orthogonal_unlearning',
    unlearn_saved_path = "./unlearned_model_2_classes"
    for unlearning_method in unlearning_method_lists:
        if unlearning_method == 'retrain':
            unlearned_model_path = unlearn_saved_path + "/retrain_sd_saved_unet"
            unlearned_model_path_text = unlearn_saved_path + "/retrain_sd_saved_text_encoder"
        elif unlearning_method == 'Salun':
            unlearned_model_path = unlearn_saved_path + "/salun_sd_saved_unet"
            unlearned_model_path_text = unlearn_saved_path + "/salun_sd_saved_text_encoder"
        elif unlearning_method == 'FSRon':
            unlearned_model_path = unlearn_saved_path + "/fsron_sd_saved_unet"
            unlearned_model_path_text = unlearn_saved_path + "/fsron_sd_saved_text_encoder"
        elif unlearning_method == 'Meta_unlearning':
            unlearned_model_path = unlearn_saved_path + "/meta_sd_saved_unet"
            unlearned_model_path_text = unlearn_saved_path + "/meta_sd_saved_text_encoder"
        elif unlearning_method == 'Orthogonal_unlearning':
            unlearned_model_path = unlearn_saved_path + "/orth_sd_saved_unet"
            unlearned_model_path_text = unlearn_saved_path + "/orth_sd_saved_text_encoder"
        elif unlearning_method == 'Salun_Orth':
            unlearned_model_path = unlearn_saved_path + "/salun_orth_sd_saved_unet"
            unlearned_model_path_text = unlearn_saved_path + "/salun_orth_sd_saved_text_encoder"

        """
            ======================================
            Unlearn the person at instance_prompt_list[0] in the target model 
            ======================================
            """
        if perform_unlearn:
            unet = UNet2DConditionModel.from_pretrained(pretrain_saved_path + '/unet').to(device)
            pipeline.unet = unet
            if args.train_text_encoder:
                text_encoder = CLIPTextModel.from_pretrained(pretrain_saved_path + '/text_encoder').to(device)
                pipeline.text_encoder = text_encoder

            forget_dataloader, retain_dataloader = generate_forget_retain_dataloader(pipeline,
                                                                                     tokenizer,
                                                                                     forget_index,
                                                                                     retain_index,
                                                                                     args.class_data_dir,
                                                                                     args.class_prompt,
                                                                                     args.resolution,
                                                                                     args.with_prior_preservation,
                                                                                     device,
                                                                                     weight_dtype)
            null_input_embeddings = tokenizer(
                "",
                truncation=True,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
            os.makedirs(unlearn_saved_path, exist_ok=True)
            if unlearning_method == 'retrain':
                retrain_process(model_id, forget_index, retain_index, saved_path=unlearned_model_path)
            else:
                unlearning(method=unlearning_method,
                       forget_dataloader=forget_dataloader,
                       retain_dataloader=retain_dataloader,
                       with_prior_preservation=args.with_prior_preservation,
                       forget_prompt=instance_prompt_list[0], pipeline=pipeline, noise_scheduler=noise_scheduler,
                       null_input_embeddings=null_input_embeddings, tokenizer=tokenizer,
                       saved_model_path=unlearned_model_path,
                       unlearned_model_path_text=unlearned_model_path_text)  # 'Salun' Orthogonal_unlearning

        if unlearning_method == 'retrain':
            unet = UNet2DConditionModel.from_pretrained(unlearned_model_path + '/unet').to(device)
        else:
            unet = UNet2DConditionModel.from_pretrained(unlearned_model_path).to(device)
        pipeline.unet = unet
        if args.train_text_encoder:
            if unlearning_method == 'retrain':
                text_encoder = CLIPTextModel.from_pretrained(unlearned_model_path + '/text_encoder').to(device)
            else:
                text_encoder = CLIPTextModel.from_pretrained(pretrain_saved_path + '/text_encoder').to(device)
            pipeline.text_encoder = text_encoder

        """
        ======================================
        Evaluate unlearning results
        ======================================
        """
        if perform_unlearning_evaluate:
            unet.eval()
            eval_process(pipeline, instance_prompt_list)
            save_dir_list = [f"./data/CelebA-15/{people_idx[forget_index]}/set_g_{unlearning_method}",
                              f"./data/CelebA-15/{people_idx[retain_index]}/set_g_{unlearning_method}"]
            for save_dir in save_dir_list:
                os.makedirs(save_dir, exist_ok=True)

            evaluate_unlearned_results_sd(pipeline, instance_prompt_list, instance_data_dir_list,
                                          save_dir_list, generate_images=True, forget_index=0, device=device)

        """
        ======================================
        Perform membership inference attack, where Relearning is ReA attack
        ======================================
        """
        if perform_mia:
            with_prior_preservation = False
            mia_method = 'Relearning' #'UnlearnDiffAtk'
            if mia_method == 'UnlearnDiffAtk':
                mia_attacked_dataset = ExplicitPromptDataset(
                    instance_data_root_list=instance_data_dir_list,
                    instance_prompt_list=instance_prompt_list,
                    class_data_root=None,
                    class_prompt=args.class_prompt,
                    tokenizer=tokenizer,
                    size=args.resolution,
                    center_crop=True,
                )
                candidate_dataset_list = [mia_attacked_dataset]
            elif mia_method == 'Relearning':
                # TODO construct forget dataset
                instance_prompt_list = [
                    f"a photo of {names_list[people_idx[ood_index]]} person"]
                instance_data_dir_list = [f"./data/CelebA-15/{people_idx[forget_index]}/set_B"]
                mia_attacked_dataset = DreamBoothDataset(
                    instance_data_root_list=instance_data_dir_list,
                    instance_prompt_list=instance_prompt_list,
                    class_data_root=None,  # class_data_dir,
                    class_prompt=args.class_prompt,
                    tokenizer=tokenizer,
                    size=args.resolution,
                    center_crop=True,
                )

                train_dataloader = torch.utils.data.DataLoader(
                    mia_attacked_dataset, batch_size=4, shuffle=False,
                    collate_fn=lambda examples: collate_fn(examples, with_prior_preservation), )

                latents_cache = []
                text_encoder_cache = []
                for batch in tqdm(train_dataloader, desc="Caching latents"):
                    with torch.no_grad():
                        batch["pixel_values"] = batch["pixel_values"].to(device, non_blocking=True, dtype=weight_dtype)
                        batch["input_ids"] = batch["input_ids"].to(device, non_blocking=True)
                        latents_cache.append(pipeline.vae.encode(batch["pixel_values"]).latent_dist)
                        text_encoder_cache.append((batch["input_ids"]))

                mia_attacked_dataset = LatentsDataset(latents_cache, text_encoder_cache)

                #TODO construct ood dataset
                instance_prompt_list = [f"a photo of {names_list[people_idx[ood_index]]} person"]
                instance_data_dir_list = [f"./data/CelebA-15/{people_idx[ood_index]}/set_A"]
                attacked_dataset2 = DreamBoothDataset(
                    instance_data_root_list=instance_data_dir_list,
                    instance_prompt_list=instance_prompt_list,
                    class_data_root=None,  # class_data_dir,
                    class_prompt=args.class_prompt,
                    tokenizer=tokenizer,
                    size=args.resolution,
                    center_crop=True,
                )
                train_dataloader = torch.utils.data.DataLoader(
                    attacked_dataset2, batch_size=4, shuffle=False,
                    collate_fn=lambda examples: collate_fn(examples, with_prior_preservation), )

                latents_cache = []
                text_encoder_cache = []
                for batch in tqdm(train_dataloader, desc="Caching latents"):
                    with torch.no_grad():
                        batch["pixel_values"] = batch["pixel_values"].to(device, non_blocking=True, dtype=weight_dtype)
                        batch["input_ids"] = batch["input_ids"].to(device, non_blocking=True)
                        latents_cache.append(pipeline.vae.encode(batch["pixel_values"]).latent_dist)
                        text_encoder_cache.append((batch["input_ids"]))

                attacked_dataset2 = LatentsDataset(latents_cache, text_encoder_cache)
                candidate_dataset_list = [mia_attacked_dataset, attacked_dataset2]

            image_saved_path = f'log_adv_path/{unlearning_method}/{mia_method}'
            os.makedirs(image_saved_path, exist_ok=True)

            mia_attack(method=mia_method, forget_prompt=instance_prompt_list[0],
                       input_prompt_list=[f"a photo of {names_list[people_idx[ood_index]]} person",
                                          f"a photo of {names_list[people_idx[ood_index]]} person"],
                       candidate_dataset_list=candidate_dataset_list,
                       model_id=model_id, model_saved_path=unlearned_model_path,
                       image_saved_path=image_saved_path,
                       pipeline=pipeline,
                       noise_scheduler=noise_scheduler,
                       tokenizer=tokenizer, device=device,
                       forget_index=forget_index,
                       )

if __name__ == '__main__':
    args = parse_args()
    main(args)