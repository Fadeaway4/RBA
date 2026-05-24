import random
import torch
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F

def random_label(pipeline, forget_dataloader, retain_dataloader, lr, epochs,
                 noise_scheduler, tokenizer, device, with_prior_preservation=False, mask=None, saved_model_path='salun'):
    unet = pipeline.unet
    pipeline.safety_checker = None
    unet.train()
    for param in unet.parameters():
        param.requires_grad = True

    relabel_prompt_list = ['John',
                           'Anderson',
                           'John Anderson']
    relabel_embeddings_list = [tokenizer(
        relabel_prompt,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids for relabel_prompt in relabel_prompt_list]

    scaling_factor = pipeline.vae.config.scaling_factor

    # train_method = "full"
    # parameters = []
    # for name, param in unet.named_parameters():
    #     # train only x attention layers
    #     if train_method == "xattn":
    #         if "attn2" in name:
    #             parameters.append(param)
    #     # train all layers
    #     if train_method == "full":
    #         parameters.append(param)

    optimizer = optim.AdamW(unet.parameters(), lr=lr)
    alpha = 0.5
    prior_loss_weight = 1.0

    for epoch in range(epochs):
        for step, (forget_batch, retain_batch) in enumerate(tqdm(zip(forget_dataloader, retain_dataloader), desc=f"Epoch {epoch}")):
            latent_dist = retain_batch[0][0]
            latents = latent_dist.sample()
            latents = latents * scaling_factor
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,)
            timesteps = timesteps.long()
            # Add noise to the latents according to the noise magnitude at each timestep (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning torch.vstack([null_input_embeddings]*len(batch[0][1])).to(device
            encoder_hidden_states = pipeline.text_encoder(retain_batch[0][1])[0]
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            if with_prior_preservation:
                # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)
                # Compute instance loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                # Compute prior loss
                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                # Add the prior loss to the instance loss.
                retain_loss = loss + prior_loss_weight * prior_loss
            else:
                retain_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            # retain_loss = torch.nn.functional.mse_loss(model_pred, noise) #retain

            latent_dist2 = forget_batch[0][0]
            latents2 = latent_dist2.sample()
            latents2 = latents2 * scaling_factor
            # Sample noise that we'll add to the latents
            noise2 = torch.randn_like(latents2)
            bsz = latents2.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,)
            timesteps = timesteps.long()
            # Add noise to the latents according to the noise magnitude at each timestep (this is the forward diffusion process)
            #TODO use the same noise
            forget_noisy_latents = noise_scheduler.add_noise(latents2, noise2, timesteps)
            pseudo_noisy_latents = noise_scheduler.add_noise(latents2, noise2, timesteps)

            forget_encoder_hidden_states = pipeline.text_encoder(forget_batch[0][1])[0]
            pseudo_encoder_hidden_states = (
                pipeline.text_encoder(torch.vstack(random.choices(relabel_embeddings_list,
                                                                  k=len(forget_batch[0][1]))).to(device)))[0]

            # Predict the noise residual
            model_pred_forget = unet(forget_noisy_latents, timesteps, forget_encoder_hidden_states).sample
            model_pred_pesudo = unet(pseudo_noisy_latents, timesteps, pseudo_encoder_hidden_states).sample.detach()
            forget_loss = F.mse_loss(model_pred_forget, model_pred_pesudo)

            loss = forget_loss + alpha * retain_loss

            optimizer.zero_grad()
            loss.backward()
            if mask is not None:
                for name, param in unet.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name].to(param.device)
            optimizer.step()

            if step % 10 == 0:
                print(f"Epoch:{epoch}, Step:{step}, Loss:{loss.item():.4f}")

            # Get the target for loss depending on the prediction type
            # if noise_scheduler.config.prediction_type == "epsilon":
            #     target = noise
            # elif noise_scheduler.config.prediction_type == "v_prediction":
            #     target = noise_scheduler.get_velocity(latents, noise, timesteps)
            # else:
            #     raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    unet.save_pretrained(saved_model_path)
