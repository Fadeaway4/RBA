import copy
import itertools
import random
import time

import numpy as np
import torch
from diffusers import get_scheduler
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

def orthogonality_loss(features):
    """
    计算正交性损失：
      - 输入 features 形状为 (N, d)（每个样本被 flatten 成 d 维向量）
      - 归一化每个样本向量，然后计算 Gram 矩阵；
      - 对非对角项施加平方惩罚。
    """
    eps = 1e-8
    norm = features.norm(dim=1, keepdim=True) + eps
    features_norm = features / norm
    gram = torch.mm(features_norm, features_norm.t())
    eye = torch.eye(gram.size(0), device=gram.device)
    loss = ((gram - eye) ** 2).sum() / (gram.size(0) * (gram.size(0) - 1) + eps)
    return loss

def sample_wise_orthogonality_loss(features):
    """
    对输入 features（形状为 [B, ...]）先 flatten 成 (B, d)，
    再计算样本间正交性损失。
    """
    features_flat = features.view(features.size(0), -1)
    return orthogonality_loss(features_flat)


# Function to apply random dropout to gradients
def random_gradient_dropout(model, dropout_rate=0.5):
    """
    Randomly drops a portion of the gradients by applying a random mask.

    Args:
    - model: The neural network model whose parameters are to be modified.
    - dropout_rate: The fraction of gradients to be "dropped". For example, 0.5 means 50% of the gradients are dropped.
    """
    # Iterate over model parameters
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Generate a random mask with the same shape as the gradient
            mask = (torch.rand_like(param.grad) > dropout_rate).float().to(param.grad.device)

            # Apply the mask to the gradients
            param.grad *= mask

def orthogonal_random_label(pipeline, forget_dataloader, retain_dataloader, lr, epochs,
                            noise_scheduler, tokenizer, device, lambda_ortho = 1.0, lambda_ortho_2 = 1.0, relabel=False,
                            cleanse_warmup =  8,
                            with_prior_preservation=False, mask=None):
    unet = pipeline.unet
    text_encoder = pipeline.text_encoder
    pipeline.safety_checker = None
    unet.train()
    text_encoder.train()
    train_method = 'xattn'#"full"
    parameters = []
    for name, param in unet.named_parameters():
        # train only x attention layers
        if train_method == "xattn":
            if "attn2" in name:
                parameters.append(param)
        # train all layers
        if train_method == "full":
            parameters.append(param)
    for param in unet.parameters():
        param.requires_grad = True
    for param in parameters:
        param.requires_grad = True

    # for param in text_encoder.parameters():
    #     param.requires_grad = True

    params_to_optimize = (
        itertools.chain(parameters,#unet.parameters(),
                        text_encoder.parameters())
        # parameters
    )
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
        在 UNet 模型的关键中间模块上注册 forward hook。
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

    hook_handles = register_unet_hooks(unet)

    alpha = 1.0
    for param in unet.parameters():
        param.requires_grad = True

    relabel_prompt_list = ['a photo of Laura person',
                           'a portrait of Laura person']

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
            if epoch > 10:#cleanse_warmup: #15
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 5e-7

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

            #TODO l1 regularization
            # l1_lambda = 5e-9  # L1 regularization strength, adjust as needed
            # l1_reg = 0.0
            # for param in unet.parameters():
            #     l1_reg += param.abs().sum()
            # retain_loss += l1_lambda * l1_reg

            # optimizer.zero_grad()
            # retain_loss.backward()
            # optimizer.step()

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
                                          device=latents.device, )
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

                ortho_loss = 0.0
                # betas = [0.0, 0.0, 0.01, 0.05, 0.1, 0.1, 0.6, 1.0] #work
                # betas = [1, 1, 0.5, 0.5] #l1=5e-9; 0.2464, 0.56; l1=0: 0.28, 0.49/0.37, 0.52

                # betas = [0.5, 0.5, 1, 1] #2e-9; 0.3455, 0,48
                # betas = [1.5, 1, 0., 0.] #2e-9 0.26, 0.43; 5e-9, 0.22, 0.34
                # betas = [1, 1, 0.4, 0.3] #1e-9 0.34, 0.31
                betas = [0.0, 0.00, 0.05, 0.5]
                # betas = [0.0, 0.0, 0.05, 0.1] #0.3,0.5
                for index, (key, act) in enumerate(intermediate_activations.items()):
                    # print("key", key)
                    # 如果 act 是 tuple 或 list，遍历查找其中第一个 Tensor
                    if isinstance(act, (tuple, list)):
                        act_tensor = None
                        for item in act:
                            if isinstance(item, torch.Tensor):
                                act_tensor = item
                                break
                        if act_tensor is None:
                            continue
                    else:
                        act_tensor = act

                    if act_tensor.size(0) > 1:
                        # print("have orthogonal")
                        ortho_loss += betas[index] * sample_wise_orthogonality_loss(act_tensor)
                if relabel:
                    model_pred_pesudo = unet(pseudo_noisy_latents, timesteps, pseudo_encoder_hidden_states).sample.detach()
                    # forget_loss = F.mse_loss(model_pred_forget, model_pred_pesudo)
                    target_noise = torch.randn_like(latents2)
                    # forget_loss = F.mse_loss(model_pred_forget.float(), model_pred_pesudo.float())
                    forget_loss = F.mse_loss(model_pred_forget.float(), target_noise.float()) + F.mse_loss(model_pred_forget.float(), model_pred_pesudo.float())
                    #0.6 0.7 for forget loss 1; 0.7 0.7, forget loss 2; 0.44, 0.59 for forget loss 1 + forget loss 2???
                    loss = lambda_ortho * ortho_loss + lambda_ortho_2 * forget_loss + alpha * retain_loss
                else:
                    loss = lambda_ortho * ortho_loss + alpha * retain_loss #forget_loss + alpha * retain_loss +

                optimizer.zero_grad()
                loss.backward()
                if mask is not None:
                    for name, param in unet.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name].to(param.device)

                # random_gradient_dropout(unet, dropout_rate=0.65)

                optimizer.step()
            else:
                loss = retain_loss
                optimizer.zero_grad()
                loss.backward()
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
    return unet, text_encoder


def gather_parameters(unet):
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

def gather_parameters_full(unet):
    """Gather the parameters to be optimized by the optimizer."""
    names, parameters = [], []
    for name, param in unet.named_parameters():
        # Train all layers.
        names.append(name)
        parameters.append(param)
    return names, parameters


def train_step(dataloader, task_unet, task_optimizer,task_lr_scheduler,
               vae,text_encoder,noise_scheduler,
               fixed_time_step=1,train_set=False, max_grad_norm=1.0, scaling_factor=1.0):
    task_unet.train()
    for param in task_unet.parameters():
        param.requires_grad = True

    all_losses = []
    for step, batch in enumerate(dataloader):
        latent_dist = batch[0][0]
        latents = latent_dist.sample()
        latents = latents * scaling_factor
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (bsz,), device=latents.device, )
        timesteps = timesteps.long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        encoder_hidden_states = text_encoder(batch[0][1])[0]

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        # Predict noise residual and compute loss
        model_pred = task_unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        loss.backward()
        all_losses.append(loss.item())

        if train_set == True:
            if max_grad_norm > 0:
                clip_grad_norm_(task_unet.parameters(), max_grad_norm)
            task_optimizer.step()
            task_lr_scheduler.step()
            task_optimizer.zero_grad()

        break

    if train_set == False:
        print(f"timestep: {fixed_time_step}, loss: {np.mean(all_losses)}")
    return task_unet,task_optimizer,task_lr_scheduler

def orthogonality_loss_two_unet(act_tensor, act_tensor_t):
    act_tensor_flat = act_tensor.view(act_tensor.size(0), -1)  # 展平 act_tensor
    act_tensor_t_flat = act_tensor_t.view(act_tensor_t.size(0), -1)  # 展平 act_tensor_t

    dot_product = torch.mm(act_tensor_flat, act_tensor_t_flat.t())  # 点积

    loss = torch.norm(dot_product, p='fro')  # 使用 Frobenius 范数来度量
    return loss

def train_unlearn_step(prompt, noise_scheduler, text_encoder, tokenizer,
                       unet_teacher, unet_student, forget_dataloader, step, device,
                       scaling_factor, intermediate_activations, intermediate_activations_teacher):
    unet_student.train()
    ortho_loss = 0.0
    for step, forget_batch in enumerate(tqdm(forget_dataloader, desc=f"Step {step}")):
        latent_dist = forget_batch[0][0]
        latents = latent_dist.sample()
        latents = latents * scaling_factor
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (bsz,), device=latents.device, )
        timesteps = timesteps.long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states =text_encoder(forget_batch[0][1])[0]
        _ = unet_student(noisy_latents, timesteps, encoder_hidden_states).sample
        _ =  unet_teacher(noisy_latents, timesteps, encoder_hidden_states).sample
        betas =np.array([0.0, 0.0, 0., 0.65])*5e-8#last np.array([0.0, 0.0, 0., 1.0])*1e-7 #last np.array([0.005, 0.01, 0.1, 0.8])*2e-7
        for index,((key, act), (key_t, act_t) ) in enumerate(zip(intermediate_activations.items(),
                                                                 intermediate_activations_teacher.items())):
            if isinstance(act, (tuple, list)):
                act_tensor = None
                for item in act:
                    if isinstance(item, torch.Tensor):
                        act_tensor = item
                        break
                if act_tensor is None:
                    continue
            else:
                act_tensor = act
            if isinstance(act_t, (tuple, list)):
                act_tensor_t = None
                for item in act_t:
                    if isinstance(item, torch.Tensor):
                        act_tensor_t = item
                        break
                if act_tensor_t is None:
                    continue
            else:
                act_tensor_t = act_t

            if act_tensor.size(0) > 1:
                # print("have orthogonal")
                ortho_loss += betas[index] * orthogonality_loss_two_unet(act_tensor, act_tensor_t)
        break
    return ortho_loss


def our_version_3(pipeline, forget_dataloader, retain_dataloader, lr, epochs,
                            noise_scheduler, tokenizer, device, forget_prompt,
                         lambda_ortho = 1.0, lambda_ortho_2 = 1.0, relabel=False,
                        cleanse_warmup =  8, with_prior_preservation=False, mask=None, **kwargs ):
    train_text_encoder=False #indicator

    unet = pipeline.unet
    vae = pipeline.vae
    unet_teacher = copy.deepcopy(pipeline.unet)
    unet_teacher.requires_grad_(False)
    vae.requires_grad_(False)

    pipeline.safety_checker = None
    unet.train()

    train_method = 'full'#last: 'full'#'xattn'#"full"
    parameters_unet = []
    for name, param in unet.named_parameters():
        if train_method == "xattn": # train only x attention layers
            if "attn2" in name:
                parameters_unet.append(param)
        if train_method == "full":  # train all layers
            parameters_unet.append(param)
    for param in parameters_unet:
        param.requires_grad = True
    """
    text_encoder = pipeline.text_encoder
    if train_text_encoder:
        text_encoder.train()
        params_to_optimize = (
            itertools.chain(parameters,#unet.parameters(),
                            text_encoder.parameters()))
    else:
        params_to_optimize = (parameters)
    optimizer = torch.optim.AdamW(params_to_optimize, lr=lr, betas=(0.99, 0.999), weight_decay=1e-8, eps=1e-4,)
    """
    text_encoder = pipeline.text_encoder
    # optimizer = optim.AdamW(unet.parameters(), lr=lr)

    alpha = 1.0
    prior_loss_weight = 1.0
    scaling_factor = pipeline.vae.config.scaling_factor
    learning_rate = 5e-6 # TODO 1e-5
    num_train_steps = 10#last: 15
    lr_scheduler_name = 'constant'
    lr_warmup_steps = 30
    gradient_accumulation_steps = 1
    max_grad_norm = 1

    #TODO _, parameters_unet = gather_parameters(unet)
    optimizer = optim.AdamW(
        parameters_unet,
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

    intermediate_activations = {}
    def recursive_detach(x):
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
            intermediate_activations[name] = recursive_detach(output)
        return hook

    def register_unet_hooks(unet_model):
        hook_handles = []
        if hasattr(unet_model, 'down_blocks'):
            handle = unet_model.down_blocks[-1].register_forward_hook(create_hook(f"down_block"))
            hook_handles.append(handle)
        if hasattr(unet_model, 'mid_block'):
            handle = unet_model.mid_block.register_forward_hook(create_hook("middle_block"))
            hook_handles.append(handle)
        if hasattr(unet_model, 'up_blocks'):
            handle = unet_model.up_blocks[-1].register_forward_hook(create_hook(f"up_block"))
            hook_handles.append(handle)
        if hasattr(unet_model, 'conv_out'):
            handle = unet_model.conv_out.register_forward_hook(create_hook("output_conv"))
            hook_handles.append(handle)
        return hook_handles

    unet_handler = register_unet_hooks(unet)

    intermediate_activations_teacher = {}
    def create_hook_teacher(name):
        def hook(module, input, output):
            intermediate_activations_teacher[name] = recursive_detach(output)
        return hook
    def register_unet_hooks_teacher(unet_model):
        hook_handles = []
        if hasattr(unet_model, 'down_blocks'):
            handle = unet_model.down_blocks[-1].register_forward_hook(create_hook_teacher(f"down_block"))
            hook_handles.append(handle)
        if hasattr(unet_model, 'mid_block'):
            handle = unet_model.mid_block.register_forward_hook(create_hook_teacher("middle_block"))
            hook_handles.append(handle)
        if hasattr(unet_model, 'up_blocks'):
            handle = unet_model.up_blocks[-1].register_forward_hook(create_hook_teacher(f"up_block"))
            hook_handles.append(handle)
        if hasattr(unet_model, 'conv_out'):
            handle = unet_model.conv_out.register_forward_hook(create_hook_teacher("output_conv"))
            hook_handles.append(handle)
        return hook_handles

    unet_hander_teacher = register_unet_hooks_teacher(unet_teacher)

    progress_bar = tqdm(range(1, num_train_steps + 1), desc="Training")  # 50
    warmup_cleanse = 1#last 4
    for step in progress_bar:
        if step > warmup_cleanse:  # cleanse_warmup: #15
            for param_group in optimizer.param_groups:
                param_group['lr'] = 8e-7#last 8e-7

        unet.train()
        for param in unet.parameters():
            param.requires_grad = True

        sum_grads = [torch.zeros_like(p) for p in parameters_unet]
        optimizer.zero_grad()
        if step < warmup_cleanse:
            train_loss = train_unlearn_step(
                prompt=forget_prompt,
                noise_scheduler=noise_scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                forget_dataloader=forget_dataloader,
                unet_teacher=unet_teacher,
                unet_student=unet,
                step=step,
                device=device,
                scaling_factor=scaling_factor,
                intermediate_activations=intermediate_activations,
                intermediate_activations_teacher=intermediate_activations_teacher
            )
            print("orthogonal_loss,", train_loss.item())
            train_loss.backward()
            if mask is not None:
                for name, param in unet.named_parameters():
                    if param.grad is not None:# and ('mid_block' not in name) and ('up_block' not in name) and ('conv_out' not in name):
                        param.grad *= mask[name].to(param.device)

            optimizer.step()
        """
        task_unet = copy.deepcopy(unet)
        names_copy, parameters_copy = gather_parameters_full(task_unet)
        task_optimizer = optim.AdamW(parameters_copy, lr=learning_rate, betas=(0.99, 0.999),
            eps=1e-8, weight_decay=1e-4,
        )
        task_lr_scheduler: LambdaLR = get_scheduler(
            name=lr_scheduler_name, optimizer=task_optimizer, num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
            num_training_steps=100,
        )
        """
        if step > 0:
            for step, retain_batch in enumerate(tqdm(retain_dataloader, desc=f"Step {step}")):
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

                if with_prior_preservation :#and step > warmup_cleanse:
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                    loss = loss + prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss.backward()

                optimizer.step()
                #if mask is not None:
                #    for name, param in unet.named_parameters():
                #        param.grad *= mask[name].to(param.device)
                lr_scheduler.step()
            """
            for epoch in range(1):
                
                #TODO here updates
                task_unet, task_optimizer_full, task_lr_scheduler_full = train_step(forget_dataloader, task_unet,
                                                                                    task_optimizer_full,
                                                                                    task_lr_scheduler_full, vae,
                                                                                    text_encoder, noise_scheduler,
                                                                                    scaling_factor=scaling_factor,
                                                                                    train_set=True)
                task_optimizer_full.zero_grad()
                #TODO here grad only
                task_unet, task_optimizer, task_lr_scheduler = train_step(forget_dataloader, task_unet, task_optimizer,
                                                                          task_lr_scheduler, vae, text_encoder,
                                                                          noise_scheduler,
                                                                          scaling_factor=scaling_factor,)
                for i, param in enumerate(parameters_copy):
                    sum_grads[i] -= gamma2_1 * param.grad
                task_optimizer.zero_grad()
                

                #TODO here grad only
                task_unet, task_optimizer, task_lr_scheduler = train_step(retain_dataloader, task_unet, task_optimizer,
                                                                          task_lr_scheduler, vae, text_encoder,
                                                                          noise_scheduler,
                                                                          scaling_factor=scaling_factor,
                                                                          )
                for i, param in enumerate(parameters_copy):
                    sum_grads[i] += gamma2_2 * param.grad
                task_optimizer.zero_grad()

        # Apply accumulated gradients to the main model
        for i, param in enumerate(parameters_unet):
            param.grad += sum_grads[i]

        if step % gradient_accumulation_steps == 0:
            if max_grad_norm > 0:
                clip_grad_norm_(parameters_unet, max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            del sum_grads
            torch.cuda.empty_cache()
            """

    progress_bar.set_description(f"Retain Training: {loss.item():.4f} c_s: {forget_prompt}")
    if (step % 100 == 0):
        print(f"Step: {step} | Loss: {train_loss.item():.4f} | LR: {lr_scheduler.get_last_lr()[0]:.4e}")
    if train_text_encoder:
        return unet, text_encoder
    else:
        return unet