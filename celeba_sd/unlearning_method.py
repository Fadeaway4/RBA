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

from celeba_sd.generate_mask import generate_mask
from unlearning_method_repo.fsron_unlearning import fsron
from unlearning_method_repo.meta_unlearning import meta_unlearning
from unlearning_method_repo.our_rl import our_random_label
from unlearning_method_repo.our_rl1 import orthogonal_random_label, our_version_3
from unlearning_method_repo.salun_unlearning import random_label


def unlearning(method, forget_dataloader, retain_dataloader, with_prior_preservation, forget_prompt, pipeline, noise_scheduler,
               null_input_embeddings, tokenizer, saved_model_path, unlearned_model_path_text):
    device=torch.device("cuda")
    if method == 'Salun':
        start_time = time.time()
        #generate mask
        # mask = generate_mask(pipeline=pipeline, noise_scheduler=noise_scheduler, forget_dataloader=forget_dataloader, device=device,
        #                      null_input_embeddings=null_input_embeddings, save_path=save_path)
        mask = torch.load("./unlearned_model_2_classes/mask/with_0.5.pt")
        #perform salun unlearn
        random_label(pipeline, forget_dataloader, retain_dataloader, lr=3e-7, epochs=10,
                     tokenizer=tokenizer, noise_scheduler=noise_scheduler,
                     device=device,
                     mask=mask,
                     #with_prior_preservation=with_prior_preservation, #TODO it's False
                     saved_model_path=saved_model_path)
        end_time = time.time()
        print("Salun unlearning time: ", end_time - start_time)
    elif method == 'FSRon':
        start_time = time.time()
        mask = torch.load("./unlearned_model_2_classes/mask/with_0.5.pt")
        # perform salun unlearn
        fsron(pipeline, forget_dataloader, retain_dataloader, lr=3e-7, epochs=10,
                     tokenizer=tokenizer, noise_scheduler=noise_scheduler,
                     device=device,
                     mask=mask,
                     # with_prior_preservation=with_prior_preservation, #TODO it's False
                     saved_model_path=saved_model_path)
        end_time = time.time()
        print("FSRon unlearning time: ", end_time - start_time)
    elif 'Orth' in method:
        start_time = time.time()
        if method == 'Salun_Orth':
            print('Salun mask exists')
            # save_path = './unlearned_model_2_classes/mask'
            # mask = generate_mask(pipeline=pipeline, noise_scheduler=noise_scheduler,
            #                      forget_dataloader=forget_dataloader, device=device,
            #                      null_input_embeddings=null_input_embeddings, save_path=save_path)
            mask = torch.load("./unlearned_model_2_classes/mask/with_0.5.pt")
        elif method == 'Orthogonal_unlearning':
            mask = None
        #TODO version 2
        # unet, text_encoder=orthogonal_random_label(pipeline, forget_dataloader, retain_dataloader, lr=5e-6, epochs=15, #3e-6; 5
        #                             tokenizer=tokenizer,
        #                             noise_scheduler=noise_scheduler,
        #                             # lambda_ortho=0.7,#0.5, #lambda_ortho=1, 0.14/0.58
        #                             # lambda_ortho_2=0.5,#0.4,
        #                             #               relabel=False,
        #                             # #relabel=True,
        #                             # cleanse_warmup=2,
        #                             lambda_ortho=0.2, #TODO this works #no orth, the affect on retain tasks is large. maybe because lambda ortho 2 is twoo larget.
        #                                           lambda_ortho_2=0.3,
        #                                           relabel=True,
        #                                           cleanse_warmup=4,#3
        #                             with_prior_preservation=with_prior_preservation, #5e-6/3e-6; 0.2, 0.5; 3
        #                             device=device,
        #                             mask=mask)

        #TODO no Salun
        # unet, text_encoder = orthogonal_sample(pipeline, forget_dataloader, retain_dataloader, lr=5e-7, epochs=10,
        #                                        # 5e-7
        #                                        tokenizer=tokenizer,
        #                                        noise_scheduler=noise_scheduler,
        #                                        lambda_ortho=0.0,#0.7,  # 0.5, #lambda_ortho=1, 0.14/0.58
        #                                        lambda_ortho_2=0.8,  # 0.4,
        #                                        relabel=True,
        #                                        # relabel=True,
        #                                        cleanse_warmup=10,
        #                                        # lambda_ortho=0.2,
        #                                        #               lambda_ortho_2=0.5,
        #                                        #               relabel=True,
        #                                        #               cleanse_warmup=3,
        #                                        # with_prior_preservation=with_prior_preservation,
        #                                        device=device,
        #                                        mask=mask)

        # unet, text_encoder = our_random_label(pipeline, forget_dataloader, retain_dataloader, lr=1e-6, epochs=10,
                     # tokenizer=tokenizer, noise_scheduler=noise_scheduler,
                     # device=device,
                     # mask=mask,
                     # # with_prior_preservation=with_prior_preservation, #TODO it's False
                     # saved_model_path=saved_model_path)
        #TOOD version 3 using meta unlearning framework
        unet = our_version_3(pipeline, forget_dataloader, retain_dataloader, lr=1e-6, epochs=10,
                     tokenizer=tokenizer, noise_scheduler=noise_scheduler,
                     device=device,
                    forget_prompt=forget_prompt,
                     #with_prior_preservation=with_prior_preservation, #TODO it's False
                     saved_model_path=saved_model_path,
                             mask=mask)
        end_time = time.time()
        print("orthogonal unlearning time: ", end_time - start_time)
        unet.save_pretrained(saved_model_path)
        #TODO
        # os.makedirs(saved_model_path + '/unet', exist_ok=True)
        # os.makedirs(saved_model_path + '/text_encoder', exist_ok=True)
        # unet.save_pretrained(saved_model_path + '/unet')
        # text_encoder.save_pretrained(saved_model_path + '/text_encoder')


    # elif method == 'Salun_Orth':
    #     # generate mask
    #     # mask = generate_mask(pipeline=pipeline, noise_scheduler=noise_scheduler, forget_dataloader=forget_dataloader, device=device,
    #     #                      null_input_embeddings=null_input_embeddings, save_path=save_path)
    #     mask = torch.load("./unlearned_model_2_classes/mask/with_0.5.pt")
    #     start_time = time.time()
    #     unet, text_encoder = orthogonal_random_label(pipeline, forget_dataloader, retain_dataloader, lr=5e-7, epochs=15,
    #                                                  # 5e-7
    #                                                  tokenizer=tokenizer,
    #                                                  noise_scheduler=noise_scheduler,
    #                                                  lambda_ortho=0.58, cleanse_warmup=3,
    #                                                  # with_prior_preservation=with_prior_preservation,
    #                                                  device=device,
    #                                                  mask=mask)
    #     end_time = time.time()
    #     print("salun orthogonal unlearning time: ", end_time - start_time)
    #     os.makedirs(saved_model_path + '/unet', exist_ok=True)
    #     os.makedirs(saved_model_path + '/text_encoder', exist_ok=True)
    #     unet.save_pretrained(saved_model_path + '/unet')
    #     text_encoder.save_pretrained(saved_model_path + '/text_encoder')

    elif method == 'Meta_unlearning':
        unet_student = meta_unlearning(pipeline, lr=1e-5, num_train_steps=50,
                                       forget_prompt=forget_prompt, device=device, tokenizer=tokenizer, noise_scheduler=noise_scheduler)
        unet_student.save_pretrained(saved_model_path)
