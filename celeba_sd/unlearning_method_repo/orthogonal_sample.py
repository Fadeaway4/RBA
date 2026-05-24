import copy
import itertools
import random
import torch
from tqdm import tqdm
import torch.nn.functional as F


def orthogonality_loss_with_ori(act_tensor, act_tensor_ori):
    features1 = act_tensor.view(act_tensor.size(0), -1)
    features2 = act_tensor_ori.view(act_tensor_ori.size(0), -1)

    eps=1e-6
    batch_size=2
    # Normalize the features to unit vectors
    features1 = features1 / (features1.norm(p=2, dim=1, keepdim=True) + eps)
    features2 = features2 / (features2.norm(p=2, dim=1, keepdim=True) + eps)

    # Initialize loss variable
    loss = 0.0

    # Process the features in smaller batches to reduce memory consumption
    num_batches = features1.size(0) // batch_size + (1 if features1.size(0) % batch_size != 0 else 0)
    # print("num_batches", num_batches)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, features1.size(0))

        # Select the current batch of features
        batch_features1 = features1[start_idx:end_idx]
        batch_features2 = features2[start_idx:end_idx]

        # Compute the dot product for the current batch
        dot_product = torch.sum(batch_features1 * batch_features2, dim=1)

        # Add the squared dot product to the loss
        loss += (dot_product ** 2).mean()  # Mean of the squared dot product

    return loss / num_batches



def orthogonal_sample(pipeline, forget_dataloader, retain_dataloader, lr, epochs,
                            noise_scheduler, tokenizer, device, lambda_ortho = 1.0, lambda_ortho_2 = 1.0, relabel=False,
                            cleanse_warmup =  8,
                            with_prior_preservation=False, mask=None):
    unet = pipeline.unet
    text_encoder = pipeline.text_encoder
    pipeline.safety_checker = None
    ori_unet = copy.deepcopy(unet)
    ori_unet.eval()

    unet.train()
    text_encoder.train()
    for param in unet.parameters():
        param.requires_grad = True

    for param in text_encoder.parameters():
        param.requires_grad = True

    params_to_optimize = (
        itertools.chain(unet.parameters(),
                        text_encoder.parameters()))
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=lr,
        betas=(0.99, 0.999),
        weight_decay=1e-8,
        eps=1e-4,
    )
    # optimizer = optim.AdamW(unet.parameters(), lr=lr)

    # 全局字典，用于存储各个 hook 捕获到的中间输出
    intermediate_activations = {}
    def recursive_detach(x):
        """
        递归地对输入 x 进行 detach 操作：
          - 如果 x 是 Tensor，则返回 x.detach()；
          - 如果 x 是 tuple 或 list，则对其每个元素递归调用 recursive_detach，并保持原类型返回；
          - 如果 x 是 dict，则对其每个值递归调用 recursive_detach；
          - 否则直接返回 x。
            """
        if isinstance(x, torch.Tensor):
            return x#.detach()
        elif isinstance(x, (tuple, list)):
            return type(x)(recursive_detach(item) for item in x)
        elif isinstance(x, dict):
            return {k: recursive_detach(v) for k, v in x.items()}
        else:
            return x

    def create_hook(name):
        def hook(module, input, output):
            # 使用递归函数处理输出
            intermediate_activations[name] = recursive_detach(output)
        return hook

    def register_unet_hooks(unet_model):
        """
         forward hook。
        假设模型包含：
          - unet_model.input_conv
          - unet_model.down_blocks (列表或 nn.ModuleList)
          - unet_model.middle_block
          - unet_model.up_blocks (列表或 nn.ModuleList)
          - unet_model.output_conv
        返回所有 hook 的 handle 列表，便于后续移除。
        """
        hook_handles = []

        # # 注册 input_conv 的 hook
        # if hasattr(unet_model, 'input_conv'):
        #     handle = unet_model.input_conv.register_forward_hook(create_hook("input_conv"))
        #     hook_handles.append(handle)

        # 注册 down_blocks 中每个模块的 hook
        if hasattr(unet_model, 'down_blocks'):
            # for idx, block in enumerate(unet_model.down_blocks):
            #     handle = block.register_forward_hook(create_hook(f"down_block_{idx}"))
            #     hook_handles.append(handle)
            handle = unet_model.down_blocks[-1].register_forward_hook(create_hook(f"down_block"))
            hook_handles.append(handle)

        # 注册 middle_block 的 hook
        if hasattr(unet_model, 'mid_block'):
            handle = unet_model.mid_block.register_forward_hook(create_hook("middle_block"))
            hook_handles.append(handle)

        # 注册 up_blocks 中每个模块的 hook
        if hasattr(unet_model, 'up_blocks'):
            # for idx, block in enumerate(unet_model.up_blocks):
            #     handle = block.register_forward_hook(create_hook(f"up_block_{idx}"))
            #     hook_handles.append(handle)
            handle = unet_model.up_blocks[-1].register_forward_hook(create_hook(f"up_block"))
            hook_handles.append(handle)

        # 注册 output_conv 的 hook（如果存在）
        if hasattr(unet_model, 'conv_out'):
            handle = unet_model.conv_out.register_forward_hook(create_hook("output_conv"))
            hook_handles.append(handle)
        return hook_handles

    intermediate_activations_ori = {}
    def create_hook_ori(name):
        def hook(module, input, output):
            # 使用递归函数处理输出
            intermediate_activations_ori[name] = recursive_detach(output)
        return hook

    def register_unet_hooks_ori(unet_model):
        """
         forward hook。
        假设模型包含：
          - unet_model.input_conv
          - unet_model.down_blocks (列表或 nn.ModuleList)
          - unet_model.middle_block
          - unet_model.up_blocks (列表或 nn.ModuleList)
          - unet_model.output_conv
        返回所有 hook 的 handle 列表，便于后续移除。
        """
        hook_handles = []

        # # 注册 input_conv 的 hook
        # if hasattr(unet_model, 'input_conv'):
        #     handle = unet_model.input_conv.register_forward_hook(create_hook("input_conv"))
        #     hook_handles.append(handle)

        # 注册 down_blocks 中每个模块的 hook
        if hasattr(unet_model, 'down_blocks'):
            handle = unet_model.down_blocks[-1].register_forward_hook(create_hook_ori(f"down_block"))
            hook_handles.append(handle)

        # 注册 middle_block 的 hook
        if hasattr(unet_model, 'mid_block'):
            handle = unet_model.mid_block.register_forward_hook(create_hook_ori("middle_block"))
            hook_handles.append(handle)

        # 注册 up_blocks 中每个模块的 hook
        if hasattr(unet_model, 'up_blocks'):
            handle = unet_model.up_blocks[-1].register_forward_hook(create_hook_ori(f"up_block"))
            hook_handles.append(handle)

        # 注册 output_conv 的 hook（如果存在）
        if hasattr(unet_model, 'conv_out'):
            handle = unet_model.conv_out.register_forward_hook(create_hook_ori("output_conv"))
            hook_handles.append(handle)
        return hook_handles

    hook_handles = register_unet_hooks(unet)
    hook_handles_ori = register_unet_hooks_ori(ori_unet)

    alpha = 1.0
    for param in unet.parameters():
        param.requires_grad = True

    relabel_prompt_list = ['a photo of David person',
                           'a portrait of Jason person',
                           'a portrait of person',
                           'a portrait of James person',
                           'a photo of general person',
                           'walker']

    relabel_embeddings_list = [tokenizer(
        relabel_prompt,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids for relabel_prompt in relabel_prompt_list]

    scaling_factor = pipeline.vae.config.scaling_factor
    # 优化器
     #int(epochs/2)
    prior_loss_weight = 1.0
    for epoch in range(epochs):
        for step, (forget_batch, retain_batch) in enumerate(tqdm(zip(forget_dataloader, retain_dataloader), desc=f"Epoch {epoch}")):
            if epoch > cleanse_warmup:#10:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 5e-7

            if epoch < cleanse_warmup:
                latent_dist2 = forget_batch[0][0]
                latents2 = latent_dist2.sample()
                latents2 = latents2 * scaling_factor

                # Sample noise that we'll add to the latents
                noise2 = torch.randn_like(latents2)
                noise3 = torch.randn_like(latents2)
                bsz = latents2.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                          device=latents2.device, )
                timesteps = timesteps.long()
                # Add noise to the latents according to the noise magnitude at each timestep (this is the forward diffusion process)
                # TODO use the same noise
                forget_noisy_latents = noise_scheduler.add_noise(latents2, noise2, timesteps)
                pseudo_noisy_latents = noise_scheduler.add_noise(latents2, noise3, timesteps)

                forget_encoder_hidden_states = pipeline.text_encoder(forget_batch[0][1])[0]
                pseudo_encoder_hidden_states = (
                    pipeline.text_encoder(torch.vstack(random.choices(relabel_embeddings_list,
                                                                      k=len(forget_batch[0][1]))).to(device)))[0]

                # Predict the noise residual
                model_pred_forget = unet(forget_noisy_latents, timesteps, forget_encoder_hidden_states).sample
                _ = ori_unet(forget_noisy_latents, timesteps, forget_encoder_hidden_states)

                ortho_loss = 0.0
                betas = [0.0, 0.0, 0.5, 1] #up and conv_out:  0.5, 8.5 didn;t work
                for index, (activation_1, activation_ori) in enumerate(zip(intermediate_activations.items(), intermediate_activations_ori.items())):
                    key, act = activation_1
                    key_ori, act_ori = activation_ori
                    # print("index", index)
                    # print("key", key)

                    if isinstance(act, (tuple, list)):
                        act_tensor = None
                        for item in act:
                            if isinstance(item, torch.Tensor):
                                act_tensor = item
                                break
                        for item in act_ori:
                            if isinstance(item, torch.Tensor):
                                act_tensor_ori = item
                                break
                        if act_tensor is None:
                            continue
                    else:
                        act_tensor = act
                        act_tensor_ori = act_ori

                    if act_tensor.size(0) > 1:
                        # ortho_loss += betas[index] * sample_wise_orthogonality_loss(act_tensor)
                        # print("have orthogonal")
                        ortho_loss += betas[index] * orthogonality_loss_with_ori(act_tensor, act_tensor_ori)
                if relabel:
                    model_pred_pesudo = ori_unet(pseudo_noisy_latents, timesteps, pseudo_encoder_hidden_states).sample.detach()
                    # forget_loss = F.mse_loss(model_pred_forget, model_pred_pesudo)
                    target_noise = torch.randn_like(latents2)
                    forget_loss = F.mse_loss(model_pred_forget.float(), model_pred_pesudo.float())
                    # forget_loss = F.mse_loss(model_pred_forget.float(), target_noise.float()) + F.mse_loss(model_pred_forget.float(), model_pred_pesudo.float())
                    #0.6 0.7 for forget loss 1; 0.7 0.7, forget loss 2; 0.44, 0.59 for forget loss 1 + forget loss 2???
                    loss = lambda_ortho * ortho_loss + lambda_ortho_2 * forget_loss# + alpha * retain_loss
                else:
                    loss = lambda_ortho * ortho_loss# + alpha * retain_loss #forget_loss + alpha * retain_loss +

                if loss > 0.0:
                    # print("ortho_loss exists")
                    optimizer.zero_grad()
                    loss.backward()

                    if mask is not None:
                        for name, param in unet.named_parameters():
                            if param.grad is not None:
                                param.grad *= mask[name].to(param.device)

                    # random_gradient_dropout(unet, dropout_rate=0.65)
                    optimizer.step()
                # print("ortho_loss: ", ortho_loss)
            else:
                latent_dist = retain_batch[0][0]
                latents = latent_dist.sample()
                latents = latents * scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device, )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning torch.vstack([null_input_embeddings]*len(batch[0][1])).to(device
                encoder_hidden_states = pipeline.text_encoder(retain_batch[0][1])[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

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

                # TODO l1 regularization
                # l1_lambda = 5e-9  # L1 regularization strength, adjust as needed
                # l1_reg = 0.0
                # for param in unet.parameters():
                #     l1_reg += param.abs().sum()
                # retain_loss += l1_lambda * l1_reg

                # optimizer.zero_grad()
                # retain_loss.backward()
                # optimizer.step()
                loss = retain_loss
                optimizer.zero_grad()
                loss.backward()


                optimizer.step()

            if step % 10 == 0:
                try:
                    print(f"Epoch:{epoch}, Step:{step}, Loss:{loss.item():.10f}")
                except:
                    print(f"Epoch:{epoch}, Step:{step}, Loss:{loss:.4f}")

    return unet, text_encoder
