import warnings
import albumentations as A
import numpy as np
import torchvision.transforms.functional as TF
import torch
# reference: https://albumentations.ai/
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale
from albumentations import OneOf, Compose

warnings.simplefilter("ignore")


class ToTensor:
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, sample):
        # msks = [(sample["mask"] == v) for v in self.classes]
        # msk = np.stack(msks, axis=-1).astype(np.float32)
        # background = 1 - msk.sum(axis=-1, keepdims=True)
        # sample["mask"] = TF.to_tensor(np.concatenate((background, msk), axis=-1))
        sample["mask"] = torch.tensor(sample["mask"])

        for key in [k for k in sample.keys() if k != "mask"]:
            # sample[key] = sample[key].astype(np.float32)
            # sample[key] = TF.to_tensor(sample[key].astype(np.float32))
            sample[key] = TF.to_tensor(sample[key].astype(np.float32) / 255.0)
        return sample

def valid_augm(sample, size=512):
    augms = [A.Resize(height=size, width=size, p=1.0)]
    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])

def test_augm(sample):
    augms = [A.Flip(p=0.1)]
    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])

def train_augm(sample, size=512):
    augms = [
        A.ShiftScaleRotate(
            scale_limit=0.2, rotate_limit=45, border_mode=0, value=0, p=0.7
        ),
        A.RandomCrop(size, size, p=1.0),
        A.Flip(p=0.75),
        A.Downscale(scale_min=0.5, scale_max=0.75, p=0.05),
        A.MaskDropout(max_objects=3, image_fill_value=0, mask_fill_value=0, p=0.1),
        # color transforms
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=1
                ),
                A.RandomGamma(gamma_limit=(70, 130), p=1),
                A.ChannelShuffle(p=0.2),
                A.HueSaturationValue(
                    hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1
                ),
                A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1),
            ],
            p=0.8,
        ),
        # distortion
        A.OneOf(
            [
                A.ElasticTransform(p=1),
                A.OpticalDistortion(p=1),
                A.GridDistortion(p=1),
                A.IAAPerspective(p=1),
            ],
            p=0.2,
        ),
        # noise transforms
        A.OneOf(
            [
                A.GaussNoise(p=1),
                A.MultiplicativeNoise(p=1),
                A.IAASharpen(p=1),
                A.GaussianBlur(p=1),
            ],
            p=0.2,
        ),
    ]
    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])

def train_augm1(sample, size=512):
    augms = [
        A.PadIfNeeded(size, size, border_mode=0, value=0, p=1.0),
        A.RandomCrop(size, size, p=1.0),
        A.OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.75),
        # A.Normalize()
    ]
    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])

def valid_augm1(sample, size=512):
    augms = [A.Resize(height=size, width=size, p=1.0)] #, A.Normalize()]
    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])

def train_augm3(sample, size=512):
    augms = [
        A.PadIfNeeded(size, size, border_mode=0, value=0, p=1.0),
        A.RandomCrop(size, size, p=1.0),
    ]
    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])

def valid_augm2(sample, size=512):
    augms = [A.Resize(height=size, width=size, p=1.0)]
    return A.Compose(augms,additional_targets={'osm': 'image'})(image=sample["image"], mask=sample["mask"], osm=sample["osm"])


def train_augm2(sample, size=512):
    augms = [
        A.ShiftScaleRotate(
            scale_limit=0.2, rotate_limit=45, border_mode=0, value=0, p=0.7
        ),
        A.RandomCrop(size, size, p=1.0),
        A.Flip(p=0.75),
        A.Downscale(scale_min=0.5, scale_max=0.75, p=0.05),
        A.MaskDropout(max_objects=3, image_fill_value=0, mask_fill_value=0, p=0.1),
        # distortion
        A.OneOf(
            [
                A.ElasticTransform(p=1),
                A.OpticalDistortion(p=1),
                A.GridDistortion(p=1),
                A.IAAPerspective(p=1),
            ],
            p=0.2,
        ),
        # noise transforms
        A.OneOf(
            [
                A.GaussNoise(p=1),
                A.MultiplicativeNoise(p=1),
                A.IAASharpen(p=1),
                A.GaussianBlur(p=1),
            ],
            p=0.2,
        ),
    ]
    return A.Compose(augms,additional_targets={'osm': 'image'})(image=sample["image"], mask=sample["mask"], osm=sample["osm"])
