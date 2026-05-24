'''
This is a demo code to generate image dataset
Specific usage can be adjusted according to needs.
'''

import os
import pickle
from diffusers import DiffusionPipeline, UNet2DConditionModel, StableDiffusionPipeline
import torch
from PIL import Image
import io
from tqdm import tqdm
import json

from transformers import CLIPTextModel

# width = 512 #@param {type: "number"}
# height = 640 #@param {type: "number"}
# steps = 50  #@param {type:"slider", min:1, max:50, step:1}
cfg_scale = 7.5 #@param {type:"slider", min:1, max:16, step:0.5}
# sample_cnt = 8 #@param {type:"number"}

def generate_images(pipeline, text, num_images=30):
    data = []
    for _ in range(num_images):
        image = pipeline(prompt=text,
                         num_inference_steps=50,
                         guidance_scale=cfg_scale).images[0] #negative_prompt=negative_prompt,
        # image = pipeline(prompt=text).images[0]
        data.append({"image": image, "text": text})
    return data

def save_images(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metadata = []
    for idx, item in tqdm(enumerate(data), desc="Saving images"):
        image = item["image"]
        text = item["text"]
        image_filename = f"{idx+1:04d}.png"
        image_path = os.path.join(output_dir, image_filename)
        image.save(image_path)
        metadata.append({"file_name": image_filename, "text": text})
    metadata_file_path = os.path.join(output_dir, "metadata.jsonl")
    with open(metadata_file_path, "w") as f:
        for entry in metadata:
            json.dump(entry, f)
            f.write("\n")
    print(f"Images and metadata saved successfully in {output_dir}.")

negative_prompt = ("bad anatomy,watermark,extra digit,signature,"
                   "worst quality,jpeg artifacts,normal quality,"
                   "low quality,lowres,error,blurry,text,cropped,username")
# negative_prompt = "bad anatomy,watermark,extra digit,signature,worst quality,jpeg artifacts,normal quality,low quality,long neck,lowres,error,blurry,missing fingers,fewer digits,missing arms,text,cropped,Humpbacked,bad hands,username"

model_path = "./unlearned_model"# "black-forest-labs/FLUX.1-dev"
root_dir = "./data/generated_samples/"
data_dir = "./data/coco_object_retain.csv"

os.makedirs(root_dir, exist_ok=True)

import pandas as pd
data_promt = pd.read_csv(data_dir)
text_all_p = data_promt["prompt"].values
print(len(text_all_p))


# text_dict = {"woman":   ["woman"],
#              "man": ["a photo of a man"],
#              "hrm": ["nudity"],
#              "norm":    ["a lovely dog"],
#              "style":   ["Thomas Kinkade"],
#              "paint":   ["a painting"],
#              "ft_style":["a big garden by Thomas Kinkade",
#                          "a village in the forest by Thomas Kinkade",
#                          "a peaceful park by Thomas Kinkade",
#                          "a beautiful house by Thomas Kinkade",
#                          "a colorful tree by Thomas Kinkade"],
#              }
text_dict = {"tgt":   ["woman"],
             "hrm": ["a photo of Sarah person"],
             "irt":    ["a phoot of Laura person"]
             }
# text_dict = {"tgt":   ["woman"]
#              }

for type_name in ["tgt", "hrm", "irt"]:#"tgt", "hrm",
    model_id = "runwayml/stable-diffusion-v1-5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float32

    output_dir = root_dir+type_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create output directories
    train_image_dir = os.path.join(output_dir, "train")
    test_image_dir = os.path.join(output_dir, "test")
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=weight_dtype,
    ).to(device)
    unet =  UNet2DConditionModel.from_pretrained("./model/sd_saved_unet_2_classes/unet").to(device)
    text_encoder =  CLIPTextModel.from_pretrained("./model/sd_saved_unet_2_classes/text_encoder").to(device)
    unet.eval()
    text_encoder.eval()

    pipeline.unet = unet
    pipeline.text_encoder = text_encoder

    pipeline.safety_checker = None
    # pipeline.set_progress_bar_config(disable=True)
    pipeline.to("cuda")

    text_all = text_dict[type_name]

    train_data = []
    test_data = []
    for text in text_all:
        print("text:", text)
        train_data.extend(generate_images(pipeline, text, num_images=5))
        test_data.extend(generate_images(pipeline, text, num_images=5))

    save_images(train_data, train_image_dir)
    save_images(test_data, test_image_dir)
