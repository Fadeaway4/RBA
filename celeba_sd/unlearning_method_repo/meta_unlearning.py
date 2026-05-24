import copy
import inspect
import itertools
import os
import random
import time
from typing import List, Tuple, Optional, Union, Dict, Optional, Any

import numpy as np
import torch
from datasets import load_dataset
from diffusers import UNet2DConditionModel, get_scheduler, DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer


def gather_parameters(unet: UNet2DConditionModel) -> Tuple[List[str], List[torch.nn.Parameter]]:
    """Gather the parameters to be optimized by the optimizer."""
    finetuning_method = "xattn"
    names, parameters = [], []
    for name, param in unet.named_parameters():
        if finetuning_method == "full":
            # Train all layers.
            names.append(name)
            parameters.append(param)
        elif finetuning_method == "selfattn":
            # Attention layer 1 is the self-attention layer.
            if "attn1" in name:
                names.append(name)
                parameters.append(param)
        elif finetuning_method == "xattn":
            # Attention layer 2 is the cross-attention layer.
            if "attn2" in name:
                names.append(name)
                parameters.append(param)
        elif finetuning_method == "noxattn":
            # Train all layers except the cross attention and time_embedding layers.
            if name.startswith("conv_out.") or ("time_embed" in name):
                # Skip the time_embedding layer.
                continue
            elif "attn2" in name:
                # Skip the cross attention layer.
                continue
            names.append(name)
            parameters.append(param)
        elif finetuning_method == "notime":
            # Train all layers except the time_embedding layer.
            if name.startswith("conv_out.") or ("time_embed" in name):
                continue
            names.append(name)
            parameters.append(param)
        else:
            raise ValueError(f"Unknown finetuning method: {finetuning_method}")

    return names, parameters

def gather_parameters_full(unet: UNet2DConditionModel) -> Tuple[List[str], List[torch.nn.Parameter]]:
    """Gather the parameters to be optimized by the optimizer."""
    names, parameters = [], []
    for name, param in unet.named_parameters():
        # Train all layers.
        names.append(name)
        parameters.append(param)
    return names, parameters

@torch.no_grad()
def encode_prompt(
        prompt: Union[str, List[str]]=None,
        negative_prompt: Union[str, List[str]]=None,
        removing_prompt: Union[str, List[str]]=None,
        num_images_per_prompt: int=1,
        text_encoder: CLIPTextModel=None,
        tokenizer: CLIPTokenizer=None,
        device: torch.device=None,
):
    """Encode a prompt into a text embedding. Prompt can be None."""
    # Get text embeddings for unconditional and conditional prompts.
    if isinstance(prompt, str):
        prompt = [prompt]

    if removing_prompt is not None and isinstance(removing_prompt, str):
        removing_prompt = [removing_prompt]
        assert len(prompt) == len(removing_prompt), f"Safety concept must be the same length as prompt of length {len(prompt)}."

    if negative_prompt is not None and isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt]
        assert len(prompt) == len(negative_prompt), f"Negative prompt must be the same length as prompt of length {len(prompt)}."

    batch_size = len(prompt) if prompt is not None else 1

    use_attention_mask = hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask
    device = device if device is not None else text_encoder.device

    # Tokenization
    uncond_input = tokenizer(
        [""] * batch_size if negative_prompt is None else negative_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    if prompt is not None:
        prompt_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
    else:
        prompt_input = None

    if removing_prompt is not None:
        removing_input = tokenizer(
            removing_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
    else:
        removing_input = None

    # Encoding
    prompt_embeds = text_encoder(
        input_ids=uncond_input["input_ids"].to(device),
        attention_mask=uncond_input["attention_mask"].to(device) if use_attention_mask else None,
    )[0]
    if prompt_input is not None:
        prompt_emb = text_encoder(
            input_ids=prompt_input["input_ids"].to(device),
            attention_mask=prompt_input["attention_mask"].to(device) if use_attention_mask else None,
        )[0]
        prompt_embeds = torch.cat([prompt_embeds, prompt_emb], dim=0)

    if removing_input is not None:
        removing_emb = text_encoder(
            input_ids=removing_input["input_ids"].to(device),
            attention_mask=removing_input["attention_mask"].to(device) if use_attention_mask else None,
        )[0]
        prompt_embeds = torch.cat([prompt_embeds, removing_emb], dim=0)

    # Duplicate the embeddings for each image.
    if num_images_per_prompt > 1:
        seq_len = prompt_embeds.shape[1]
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.reshape(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds

def prepare_extra_step_kwargs(scheduler, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator

    return extra_step_kwargs

# Sample latents from unet and DDIM scheduler until the given timestep.
# @torch.no_grad()
def sample_until(
        until: int,
        latents: torch.Tensor,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        prompt_embeds: torch.Tensor,
        guidance_scale: float,
        extra_step_kwargs: Optional[Dict[str, Any]]=None,
):
    """Sample latents until t for a given prompt."""
    with torch.no_grad():
        timesteps = scheduler.timesteps

        do_guidance = abs(guidance_scale) > 1.0

        # Denoising loop
        for i, t in enumerate(timesteps):
            latent_model_input = (
                torch.cat([latents] * 2)
                if do_guidance
                else latents
            )
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

            # perform guidance
            if do_guidance:
                noise_pred_out = torch.chunk(noise_pred, 2, dim=0)
                noise_pred_uncond, noise_pred_prompt = noise_pred_out[0], noise_pred_out[1]
                # classifier-free guidance term
                cond_guidance = noise_pred_prompt - noise_pred_uncond
                # add the guidance term to the noise residual
                noise_pred = noise_pred_uncond + (guidance_scale * cond_guidance)

            latents = scheduler.step(model_output=noise_pred, timestep=t, sample=latents, **extra_step_kwargs).prev_sample

            if i == (until-1):
                # print(f"Sampled until t={t}, i={i}.")
                break

    return latents

def train_unlearn_step(
        prompt: str,
        removing_prompt: str,
        generator: torch.Generator,
        noise_scheduler: DDPMScheduler,
        ddim_scheduler: DDIMScheduler,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet_teacher: UNet2DConditionModel,
        unet_student: UNet2DConditionModel,
        device
) -> torch.Tensor:
    """Train the model a single step for a given prompt and return the loss."""
    num_ddim_steps  = 50
    num_ddpm_steps  = 1000
    eta = 0.0
    guidance_scale  = 3.0
    concept_scale   = 3.0

    unet_student.train()

    # Encode prompt
    prompt_embeds = encode_prompt(
        prompt=prompt,
        removing_prompt=removing_prompt,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        device=device,
    )

    uncond_emb, cond_emb, safety_emb = torch.chunk(prompt_embeds, 3, dim=0)
    batch_size = cond_emb.shape[0]

    # Prepare timesteps
    noise_scheduler.set_timesteps(num_ddpm_steps, device)

    # Prepare latent codes to generate z_t
    latent_shape = (batch_size, unet_teacher.config.in_channels, 64, 64)
    latents = torch.randn(latent_shape, generator=generator, device=device)
    # Scale the initial noise by the standard deviation required by the scheduler
    latents = latents * ddim_scheduler.init_noise_sigma # z_T

    # Normally, DDPM takes 1,000 timesteps for training, and DDIM takes 50 timesteps for inference.
    t_ddim = torch.randint(0, num_ddim_steps, (1,))
    t_ddpm_start = round((1 - (int(t_ddim) + 1) / num_ddim_steps) * num_ddpm_steps)
    t_ddpm_end   = round((1 - int(t_ddim)       / num_ddim_steps) * num_ddpm_steps)
    t_ddpm = torch.randint(t_ddpm_start, t_ddpm_end, (batch_size,),)
    # print(f"t_ddim: {t_ddim}, t_ddpm: {t_ddpm}")
    # Prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    extra_step_kwargs = prepare_extra_step_kwargs(noise_scheduler, generator, eta)

    with torch.no_grad():
        # args.guidance_scale: s_g in the paper
        prompt_embeds = torch.cat([uncond_emb, cond_emb], dim=0) if guidance_scale > 1.0 else uncond_emb
        prompt_embeds = prompt_embeds.to(unet_student.device)

        # Generate latents
        latents = sample_until(
            until=int(t_ddim),
            latents=latents,
            unet=unet_student,
            scheduler=ddim_scheduler,
            prompt_embeds=prompt_embeds,
            guidance_scale=guidance_scale,
            extra_step_kwargs=extra_step_kwargs,
        )

        # Stop-grad and send to the second device
        _latents = latents.to(device)
        e_0 = unet_teacher(_latents, t_ddpm.to(device), encoder_hidden_states=uncond_emb).sample
        e_p = unet_teacher(_latents, t_ddpm.to(device), encoder_hidden_states=safety_emb).sample

        e_0 = e_0.detach().to(device)
        e_p = e_p.detach().to(device)

        # args.concept_scale: s_s in the paper
        noise_target = e_0 - concept_scale * (e_p - e_0)

    noise_pred = unet_student(latents, t_ddpm.to(device), encoder_hidden_states=safety_emb.to(device)).sample
    loss = F.mse_loss(noise_pred, noise_target)
    return loss

def train_step(dataloader,task_unet,task_optimizer,task_lr_scheduler,
               vae,text_encoder,noise_scheduler,
               fixed_time_step=1,train_set=False, max_grad_norm=1.0):
    task_unet.train()
    for param in task_unet.parameters():
        param.requires_grad = True

    all_losses = []
    for step, batch in enumerate(dataloader):
        latents = vae.encode(batch["pixel_values"].to(vae.device)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        noise = torch.randn_like(latents)

        bsz = latents.shape[0]

        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = text_encoder(batch["input_ids"].to(vae.device), return_dict=False)[0]

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        # Predict noise residual and compute loss
        model_pred = task_unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        loss.backward()
        all_losses.append(loss.item())

        if train_set == True:
            if max_grad_norm > 0:
                clip_grad_norm_(task_unet.parameters(), max_grad_norm)
            task_optimizer.step()
            task_lr_scheduler.step()
            task_optimizer.zero_grad()
    if train_set == False:
        print(f"timestep: {fixed_time_step}, loss: {np.mean(all_losses)}")
    return task_unet,task_optimizer,task_lr_scheduler

def data_loader(tokenizer,caption_column="text",image_column="image"):
    resolution = 512
    center_crop = True
    random_flip = True
    train_batch_size=2

    hrm_dataset = load_dataset("imagefolder", data_dir="./data/generated_samples/hrm")
    irt_dataset = load_dataset("imagefolder", data_dir="./data/generated_samples/irt")
    tgt_dataset = load_dataset("imagefolder", data_dir="./data/generated_samples/tgt")

    from torchvision import transforms
    # Preprocessing the datasets.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    train_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    # Set the training transforms
    hrm_train_dataset = hrm_dataset["train"].with_transform(preprocess_train)
    hrm_test_dataset = hrm_dataset["test"].with_transform(preprocess_train)

    irt_train_dataset = irt_dataset["train"].with_transform(preprocess_train)
    irt_test_dataset = irt_dataset["test"].with_transform(preprocess_train)

    tgt_train_dataset = tgt_dataset["train"].with_transform(preprocess_train)
    tgt_test_dataset = tgt_dataset["test"].with_transform(preprocess_train)


    # DataLoaders creation:
    hrm_train_dataloader = torch.utils.data.DataLoader(
        hrm_train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=8,
    )

    hrm_test_dataloader = torch.utils.data.DataLoader(
        hrm_test_dataset,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=8,
    )

    irt_train_dataloader = torch.utils.data.DataLoader(
        irt_train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=8,
    )

    irt_test_dataloader = torch.utils.data.DataLoader(
        irt_test_dataset,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=8,
    )

    tgt_train_dataloader = torch.utils.data.DataLoader(
        tgt_train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=8,
    )

    tgt_test_dataloader = torch.utils.data.DataLoader(
        tgt_test_dataset,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=8,
    )
    return hrm_train_dataloader,hrm_test_dataloader,irt_train_dataloader,irt_test_dataloader,tgt_train_dataloader,tgt_test_dataloader

def meta_unlearning(pipeline, forget_prompt, device, tokenizer, noise_scheduler, lr=1e-5, num_train_steps=50):
    start_time = time.time()
    learning_rate = lr  # TODO 1e-5
    # Freeze vae and text_encoder
    unet_student = pipeline.unet
    unet_teacher = pipeline.unet
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder

    model_id = "runwayml/stable-diffusion-v1-5"
    ddim_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    unet_teacher.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    gen = torch.Generator(device=device)

    scaling_factor = 1
    lr_scheduler_name = 'constant'
    lr_warmup_steps = 30
    gradient_accumulation_steps = 1
    max_grad_norm = 1

    gamma1_1 = 0.1
    gamma1_2 = 0.01
    gamma2_1 = 0.1
    gamma2_2 = 0.1  # TODO
    gamma2_3 = 0.01  # TODO

    # Create optimizer and scheduler
    names, parameters = gather_parameters(unet_student)
    optimizer = optim.AdamW(
        parameters,
        lr=learning_rate,
        betas=(0.99, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    lr_scheduler: LambdaLR = get_scheduler(
        name=lr_scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=num_train_steps * gradient_accumulation_steps,
    )

    num_ddim_steps = 50
    ddim_scheduler.set_timesteps(num_ddim_steps, device)

    # TODO generate dataloader
    (hrm_train_dataloader, hrm_test_dataloader,
     irt_train_dataloader, irt_test_dataloader,
     tgt_train_dataloader, tgt_test_dataloader) \
        = data_loader(tokenizer)

    progress_bar = tqdm(range(1, num_train_steps + 1), desc="Training") #50
    for step in progress_bar:
        removing_prompt = forget_prompt
        prompt = removing_prompt

        unet_student.train()
        for param in unet_student.parameters():
            param.requires_grad = True

        sum_grads = [torch.zeros_like(p) for p in parameters]

        if step < 10:
            train_loss = train_unlearn_step(
                prompt=prompt,
                removing_prompt=removing_prompt,
                generator=gen,
                noise_scheduler=noise_scheduler,
                ddim_scheduler=ddim_scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet_teacher=unet_teacher,
                unet_student=unet_student,
                device=device,
            )
            train_loss.backward()

        task_unet = copy.deepcopy(unet_student)
        names_copy, parameters_copy = gather_parameters(task_unet)

        task_optimizer = optim.AdamW(
            parameters_copy,
            lr=learning_rate,
            betas=(0.99, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
        )
        task_lr_scheduler: LambdaLR = get_scheduler(
            name=lr_scheduler_name,
            optimizer=task_optimizer,
            num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
            num_training_steps=100,
        )

        names_copy_full, parameters_copy_full = gather_parameters_full(task_unet)
        task_optimizer_full = optim.AdamW(
            parameters_copy_full,
            lr=learning_rate,
            betas=(0.99, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
        )

        task_lr_scheduler_full: LambdaLR = get_scheduler(
            name=lr_scheduler_name,
            optimizer=task_optimizer_full,
            num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
            num_training_steps=100,
        )
        # TODO
        task_unet, task_optimizer, task_lr_scheduler = train_step(irt_test_dataloader, task_unet, #retained
                                                                  task_optimizer, task_lr_scheduler, vae, text_encoder,
                                                                  noise_scheduler)
        for i, param in enumerate(parameters_copy):
            print('gamma1-1, params.grad', param.grad)
            sum_grads[i] += gamma1_1 * param.grad #0.1
        task_optimizer.zero_grad()
        task_unet, task_optimizer, task_lr_scheduler = train_step(tgt_test_dataloader, task_unet, #related
                                                                  task_optimizer, task_lr_scheduler, vae, text_encoder,
                                                                  noise_scheduler)
        for i, param in enumerate(parameters_copy):
            print('gamma1-2, params.grad', param.grad)
            sum_grads[i] += gamma1_2 * param.grad #0.01
        task_optimizer.zero_grad()

        for epoch in range(1):
            task_unet, task_optimizer_full, task_lr_scheduler_full = train_step(hrm_train_dataloader, task_unet, task_optimizer_full,
                                                                                task_lr_scheduler_full, vae,
                                                                                text_encoder, noise_scheduler, train_set=True)
            task_optimizer_full.zero_grad()
            task_unet, task_optimizer, task_lr_scheduler = train_step(hrm_test_dataloader, task_unet, task_optimizer,
                                                                      task_lr_scheduler, vae, text_encoder, noise_scheduler)
            for i, param in enumerate(parameters_copy):
                print('gamma2-1, params.grad', param.grad)
                sum_grads[i] -= gamma2_1 * param.grad
            task_optimizer.zero_grad()

            task_unet, task_optimizer, task_lr_scheduler = train_step(tgt_test_dataloader, task_unet,
                                                                      task_optimizer, task_lr_scheduler, vae, text_encoder, noise_scheduler)
            for i, param in enumerate(parameters_copy):
                print('gamma2-2, params.grad', param.grad)
                sum_grads[i] -= gamma2_2 * param.grad
            task_optimizer.zero_grad()
            task_unet, task_optimizer, task_lr_scheduler = train_step(irt_test_dataloader, task_unet,
                                                                      task_optimizer, task_lr_scheduler, vae, text_encoder, noise_scheduler)
            for i, param in enumerate(parameters_copy):
                print('gamma2-3, params.grad', param.grad)
                sum_grads[i] -= gamma2_3 * param.grad
            task_optimizer.zero_grad()

        # Apply accumulated gradients to the main model
        for i, param in enumerate(parameters):
            param.grad += sum_grads[i]

        if step % gradient_accumulation_steps == 0:
            if max_grad_norm > 0:
                clip_grad_norm_(parameters, max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            del sum_grads
            torch.cuda.empty_cache()

    progress_bar.set_description(f"Training: {train_loss.item():.4f} on c_p: {prompt} - c_s: {forget_prompt}")

    if (step % 100 == 0):
        print(f"Step: {step} | Loss: {train_loss.item():.4f} | LR: {lr_scheduler.get_last_lr()[0]:.4e}")
    end_time = time.time()
    print("Meta unlearning time: ", end_time - start_time)
    return unet_student

