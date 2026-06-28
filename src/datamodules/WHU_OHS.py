import kornia.augmentation as K
import os
import torch
import torchvision.transforms as T
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.datasets import WHUOHS
from src.datamodules.augmentations import get_train_augmentation, get_augmentation
# adapted from https://github.com/isaaccorley/resize-is-all-you-need/


class WHUOHSDataModule(LightningDataModule):
    @staticmethod
    def preprocess(sample):
        sample["image"] = sample["image"].float()
        return sample

    def __init__(
        self,
        root,
        pad_missing_bands=False,
        batch_size=32,
        num_workers=8,
        seed=0,
        transforms=None,
        cfg=None,
    ):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pad_missing_bands = pad_missing_bands
        self.generator = torch.Generator().manual_seed(seed)
        self.transforms = transforms
        self.cfg = cfg

    def is_image_file(self, filename):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif'
        ]
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def setup(self, fit):

        data_path_train_image = os.path.join(self.root+'/tr', 'image')
        data_path_val_image = os.path.join(self.root+'/val', 'image')
        data_path_test_image = os.path.join(self.root+'/ts', 'image')

        train_image_list = []
        train_label_list = []
        val_image_list = []
        val_label_list = []
        test_image_list = []
        test_label_list = []

        for root, paths, fnames in sorted(os.walk(data_path_train_image)):
            for fname in fnames:
                if self.is_image_file(fname):
                    image_path = os.path.join(data_path_train_image, fname)
                    label_path = image_path.replace('image', 'label')
                    assert os.path.exists(label_path)
                    assert os.path.exists(image_path)
                    train_image_list.append(image_path)
                    train_label_list.append(label_path)

        for root, paths, fnames in sorted(os.walk(data_path_val_image)):
            for fname in fnames:
                if self.is_image_file(fname):
                    image_path = os.path.join(data_path_val_image, fname)
                    label_path = image_path.replace('image', 'label')
                    assert os.path.exists(label_path)
                    assert os.path.exists(image_path)
                    val_image_list.append(image_path)
                    val_label_list.append(label_path)

        for root, paths, fnames in sorted(os.walk(data_path_test_image)):
            for fname in fnames:
                if self.is_image_file(fname):
                    image_path = os.path.join(data_path_test_image, fname)
                    label_path = image_path.replace('image', 'label')
                    assert os.path.exists(label_path)
                    assert os.path.exists(image_path)
                    test_image_list.append(image_path)
                    test_label_list.append(label_path)

        assert len(train_image_list) == len(train_label_list)
        assert len(val_image_list) == len(val_label_list)
        assert len(test_image_list) == len(test_label_list)

        # dataset
        traintransform = get_train_augmentation(self.cfg.data.re_img_size, seg_fill=self.cfg.data.IGNORE_LABEL)
        valtransform = get_augmentation(self.cfg.data.re_img_size)
        testtransform = get_augmentation(self.cfg.data.re_img_size)
        self.train_dataset = WHUOHS(image_file_list=train_image_list,
                                    label_file_list=train_label_list, transform=traintransform)
        self.val_dataset = WHUOHS(image_file_list=val_image_list,
                                    label_file_list=val_label_list, transform=valtransform)
        self.test_dataset = WHUOHS(image_file_list=test_image_list,
                                    label_file_list=test_label_list, transform=testtransform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
