import argparse
import os

import torch
from tqdm import tqdm

def generate_mask(
        pipeline,
        noise_scheduler,
        forget_dataloader,
        null_input_embeddings,
        epochs=1,
        lr=1e-5,
        batch_size=2,
        device=torch.device("cuda"),
        save_path=''
):
    #factors
    c_guidance=7.5

    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    unet = pipeline.unet
    unet.to(device)
    unet.eval()

    scaling_factor = pipeline.vae.config.scaling_factor

    criteria = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    gradients = {}
    for name, param in unet.named_parameters():
        gradients[name] = 0

    with tqdm(total=len(forget_dataloader)) as t:
        for i, batch in enumerate(forget_dataloader):
            latent_dist = batch[0][0] #one sample only
            latents = latent_dist.sample()
            latents = latents * scaling_factor

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
            null_encoder_hidden_states = pipeline.text_encoder(torch.vstack([null_input_embeddings]*len(batch[0][1])).to(device))[0]
            print("encoder_hidden_states", encoder_hidden_states.shape)
            print("null_encoder_hidden_states", null_encoder_hidden_states.shape)
            # Predict the noise residual
            forget_out = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            null_out = unet(noisy_latents, timesteps, null_encoder_hidden_states).sample

            preds = (1 + c_guidance) * forget_out - c_guidance * null_out

            loss = - criteria(noise, preds)
            loss.backward()

            with torch.no_grad():
                for name, param in unet.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad.data.cpu()

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

        threshold_list = [0.5]
        for i in threshold_list:
            sorted_dict_positions = {}
            hard_dict = {}

            # Concatenate all tensors into a single tensor
            all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])

            # Calculate the threshold index for the top 10% elements
            threshold_index = int(len(all_elements) * i)

            # Calculate positions of all elements
            positions = torch.argsort(all_elements)
            ranks = torch.argsort(positions)

            start_index = 0
            for key, tensor in gradients.items():
                num_elements = tensor.numel()
                # tensor_positions = positions[start_index: start_index + num_elements]
                tensor_ranks = ranks[start_index : start_index + num_elements]

                sorted_positions = tensor_ranks.reshape(tensor.shape)
                sorted_dict_positions[key] = sorted_positions

                # Set the corresponding elements to 1
                threshold_tensor = torch.zeros_like(tensor_ranks)
                threshold_tensor[tensor_ranks < threshold_index] = 1
                threshold_tensor = threshold_tensor.reshape(tensor.shape)
                hard_dict[key] = threshold_tensor
                start_index += num_elements

            os.makedirs(save_path+"/mask", exist_ok=True)
            torch.save(hard_dict, os.path.join(save_path+"/mask/with_{}.pt".format(i)))
    return hard_dict
