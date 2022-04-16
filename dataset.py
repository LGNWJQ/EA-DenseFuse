# -*- coding: utf-8 -*-
"""
Writer: WJQpe
Date: 2022 04 02 
"""
import torch
import os
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class COCO2014(Dataset):
    def __init__(self, images_path, transform=None, image_num=None):
        self.images_path = images_path
        self.transform = transform
        self.image_list = os.listdir(images_path)
        if image_num is not None:
            self.image_list = self.image_list[:image_num]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image_name = os.path.join(self.images_path, self.image_list[item])
        image = read_image(image_name, mode=ImageReadMode.RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def collate_fn(batch):
        images = zip(*batch)
        images = torch.stack(images, dim=0)
        return images


def transform_image(resize=256, gray=False):
    if gray:
        tf_list = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(400),
                transforms.RandomCrop(resize),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ]
        )
    else:
        tf_list = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(400),
                transforms.RandomCrop(resize),
                transforms.ToTensor()
            ]
        )
    return tf_list

