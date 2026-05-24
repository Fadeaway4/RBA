from pathlib import Path
from torch.utils.data import Dataset
import os
import json
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

first_names = ["Sarah", "Emily",  "Laura",  "David", "John", "Sophia",  "Michael", "James",  "Chris"]
last_names = ["Smith", "Johnson", "Brown", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin"]
people_idx = [5, 26, 34, 47, 'n000050', 67]

names_list = {people_idx[i]: f'{first_names[i]}' for i in range(6)} # {last_names[i]}

class LatentsDataset(Dataset):
    def __init__(self, latents_cache, text_encoder_cache):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        return self.latents_cache[index], self.text_encoder_cache[index]

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
            self,
            instance_data_root_list,
            instance_prompt_list,
            tokenizer,
            class_data_root=None,
            class_prompt=None,
            size=512,
            center_crop=False,
    ):
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop

        self.instance_data_root_list = []
        self.instance_images_path_list = []

        self.instance_data_root_list=[Path(instance_data_root) for instance_data_root in instance_data_root_list]
        if not Path(instance_data_root_list[0]).exists():
            raise ValueError(f"Instance {self.instance_data_root_list[0]} images root doesn't exists.")

        self.instance_images_path_list=[list(Path(instance_data_root).iterdir()) for instance_data_root in self.instance_data_root_list]
        self.instance_prompt_list = instance_prompt_list

        self._length = 0
        for i in range(len(self.instance_images_path_list)):
            self._length += len(self.instance_images_path_list[i])

        self.num_instance_images = len(self.instance_images_path_list[0]) #each 4 samples
        self.class_number = len(self.instance_images_path_list) #2

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None
        self.class_prompt = class_prompt

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        class_idx = int(np.floor(index % (self.num_instance_images * self.class_number) / self.num_instance_images))
        example = {}
        instance_image = Image.open(self.instance_images_path_list[class_idx][index % self.num_instance_images])

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt_list[class_idx],
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example

class ExplicitPromptDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
            self,
            instance_data_root_list,
            instance_prompt_list,
            tokenizer,
            class_data_root=None,
            class_prompt=None,
            size=512,
            center_crop=False,
    ):
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop

        self.instance_data_root_list = []
        self.instance_images_path_list = []

        self.instance_data_root_list=[Path(instance_data_root) for instance_data_root in instance_data_root_list]
        if not Path(instance_data_root_list[0]).exists():
            raise ValueError(f"Instance {self.instance_data_root_list[0]} images root doesn't exists.")

        self.instance_images_path_list=[list(Path(instance_data_root).iterdir()) for instance_data_root in self.instance_data_root_list]
        self.instance_prompt_list = instance_prompt_list

        self.num_instance_images = len(self.instance_images_path_list[0]) #each 4 samples

        self._length = 0
        for i in range(len(self.instance_images_path_list)):
            self._length += len(self.instance_images_path_list[i])

        self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        class_idx = int(np.floor(index / self.num_instance_images))
        example = {}
        instance_image = Image.open(self.instance_images_path_list[class_idx][index % self.num_instance_images])

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        return self.image_transforms(instance_image), self.instance_prompt_list[class_idx]
