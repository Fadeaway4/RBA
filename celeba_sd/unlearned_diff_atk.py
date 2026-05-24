import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import numpy as np
from transformers import CLIPModel, CLIPProcessor

def UnlearnDiffAtk(candidate_dataset_list, pipeline, tokenizer, device):
    # 一个例示的普通prompt，可以替换成你想要的描述
    base_prompt = "A photo of sarah smith person"
    scaling_factor = pipeline.vae.config.scaling_factor
    text_encoder = pipeline.text_encoder
    # 初始化CLIP，用于计算图像相似度(或差异)
    # clip_model_id = "openai/clip-vit-base-patch32"
    # clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
    # clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

    latent_dist = (candidate_dataset_list[0])[0][0] #the image
    latents = latent_dist.sample()
    latents = latents * scaling_factor
    latents = latents.to(device)
    # print(latents)

    # with torch.no_grad():
    #     target_embeds = clip_model.get_image_features(**target_image)
    #     target_embeds = target_embeds / target_embeds.norm(p=2, dim=-1, keepdim=True)

    # ========== 2. 可学习的对抗前缀 ==========
    NUM_ADV_TOKENS = 5  # 想插入多少个对抗Token
    # tokenizer = pipeline.tokenizer

    adv_token_ids = torch.randint(low=0, high=tokenizer.vocab_size, size=(NUM_ADV_TOKENS,)).to(device)

    # 将这些 tokens 做为可学习embedding
    # 获取原始embedding table
    text_embeds = pipeline.text_encoder.get_input_embeddings()  # nn.Embedding
    embedding_dim = text_embeds.embedding_dim  # 通常768或1024

    # 创建可学习参数
    adv_embeddings = nn.Parameter(torch.randn(NUM_ADV_TOKENS, embedding_dim, device=device) * 0.1)
    # 优化器
    optimizer = optim.Adam([adv_embeddings], lr=1e-2)

    # ========== 3. 定义训练循环 ==========
    def generate_image_with_adv(prefix_emb: torch.Tensor, prompt: str):
        """
        使用给定的 "对抗前缀 embedding" + prompt 生成图像。
        prefix_emb: [NUM_ADV_TOKENS, embedding_dim]
        prompt:  用户正常的文字，如 "a photograph of an astronaut riding a horse"
        """
        # 1) tokenizer 处理普通prompt --> text_ids shape: [1, seq_len]
        text_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        # 2) 获取普通prompt的embedding
        base_emb = pipeline.text_encoder(text_ids)  # shape: [1, seq_len, embed_dim]

        # 3) 拼接 [对抗前缀embedding] + [普通prompt embedding]
        #    prefix_emb shape: [NUM_ADV_TOKENS, embed_dim] -> [1, NUM_ADV_TOKENS, embed_dim]
        prefix_emb = prefix_emb.unsqueeze(0)
        full_emb = torch.cat([prefix_emb, base_emb], dim=1)  # [1, NUM_ADV_TOKENS + seq_len, embed_dim]

        # 4) 手动调用 text_encoder 的 forward，得到 text_encoder_output
        #    正常来说 text_encoder 是把 input_ids -> final hidden states
        #    我们自己构造embedding，需要用 text_encoder 的 forward(..., inputs_embeds=...) 参数
        attention_mask = torch.ones(full_emb.shape[:-1], device=device)
        text_encoder_out = pipeline.text_encoder(
            input_ids=None,
            inputs_embeds=full_emb,
            attention_mask=attention_mask
        )
        text_model_output = text_encoder_out.last_hidden_state

        # 5) 用 pipe 的 __call__ 时，需要给定 encoder hidden states
        #    diffusers >=0.10.0版本: 通过 "pipe(prompt_embeds=...)" 方式
        image = pipeline(
            prompt=None,              # 我们用的是 prompt_embeds
            prompt_embeds=text_model_output,
            num_inference_steps=30,   # 生成步数，自己调
            guidance_scale=7.5,       # CFG权重
        ).images[0]

        return image

    def generate_image_tensor(prefix_emb: torch.Tensor, prompt: str,
                              num_inference_steps=25, guidance_scale=7.5):
        """
        尝试在不打断计算图的情况下生成图像Tensor。
        需要 diffusers >= 0.14(?) 并支持 output_type='latent' 或 'torch'。
        如果不支持，需要自行实现 U-Net + VAE 解码部分。
        """
        # 1) tokenizer普通prompt
        text_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        base_emb = text_embeds(text_ids)    # shape [1, seq_len, embed_dim]

        # 2) 拼接对抗前缀
        prefix_emb = prefix_emb.unsqueeze(0)  # [1, NUM_ADV_TOKENS, embed_dim]
        full_emb = torch.cat([prefix_emb, base_emb], dim=1)

        # 3) 送入 text_encoder 得到 final hidden state
        # text_encoder_out = text_encoder(
        #     full_emb.to(device))
        # prompt_embeds = text_encoder_out.last_hidden_state

        # 4) 调用 pipe，使用 prompt_embeds 参数，返回 Tensor 而非 PIL
        #    需要 diffusers >= 0.14 并支持 "torch" 或 "latent"
        image_out = pipeline(
            prompt=None,
            prompt_embeds=full_emb,#prompt_embeds,
            num_inference_steps=50,
            guidance_scale=7.5,
            output_type="torch",   # 或者 "latent" 再自己 decode
        )
        gen_tensor = image_out.images  # [1, 3, 512, 512], float, [0,1]
        return gen_tensor

    mse_loss = nn.MSELoss()
    EPOCHS = 100  # 迭代次数（越多越慢，也不一定越好；可自行调参）
    for step in tqdm(range(EPOCHS)):
        optimizer.zero_grad()

        # 1) 用当前对抗前缀 + base_prompt 生成图像
        # gen_image = generate_image_with_adv(adv_embeddings, base_prompt)

            # 1) 生成图像tensor (保持计算图)
        gen_tensor = generate_image_tensor(
            prefix_emb=adv_embeddings,
            prompt=base_prompt,
            num_inference_steps=25,
            guidance_scale=7.5,
        )
        # gen_tensor: [1,3,H,W], range [0,1] (假设pipe这样返回)

        # 2) 计算像素空间的 MSE Loss
        loss = mse_loss(gen_tensor, latents)

        # 3) 反向传播 & 更新
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    # ========== 4. 查看结果 & 得到对抗字符 ==========

    # 训练结束后，再用训练后的前缀生成一次图
    final_image = generate_image_with_adv(adv_embeddings, base_prompt)
    final_image.save("adv_attack_output.png")

    # 如果需要「对抗字符」的可视化：
    # (1) 可以把训练好的 adv_embeddings 最近似的 token id 查出来 (会有不确定性)
    # (2) 或者直接尝试贪心找embedding里最相似的词表向量
    # 这里演示一个简化的最近邻查找:
    learned_prefix = []
    vocab = pipeline.tokenizer.get_vocab()  # {token_string: token_id}
    inv_vocab = {v: k for k, v in vocab.items()}  # token_id -> token_string

    embedding_weights = pipeline.text_encoder.get_input_embeddings().weight.data  # [vocab_size, embed_dim]

    for i in range(NUM_ADV_TOKENS):
        # 对于 adv_embeddings[i], 找和embedding_weights里最接近的 token
        emb = adv_embeddings[i].detach()  # [embed_dim]
        # 计算与所有vocab token embedding的余弦相似度
        sim = torch.nn.functional.cosine_similarity(
            emb.unsqueeze(0), embedding_weights, dim=-1
        )
        token_id = torch.argmax(sim).item()
        token_str = inv_vocab[token_id]
        learned_prefix.append(token_str)

    print("Learned adversarial token IDs:", learned_prefix)
    print("Decoded (approx) prefix:", pipeline.tokenizer.convert_tokens_to_string(learned_prefix))

