import os
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.datasets import MTS12
from src.datamodules.augmentations import get_train_augmentation, get_augmentation

class MTS12DataModule(LightningDataModule):
    @staticmethod
    def preprocess(sample):
        sample["image"] = sample["image"].float()
        return sample

    def __init__(
        self,
        root,
        batch_size=32,
        num_workers=8,
        seed=0,
        transforms=None,
        cfg=None,
    ):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.generator = torch.Generator().manual_seed(seed)
        self.transforms = transforms
        self.cfg = cfg
        
    def is_image_file(self, filename):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif', '.mat'
        ]
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def setup(self, fit):

        train_image_list, train_label_list = [], []
        with open('data_list/MTS12_train_image_list.txt', 'r') as file:
            for line in file:  
                tmp_list = line.strip()  
                if tmp_list:  
                    train_image_list.append(os.path.join(self.root, tmp_list))
                    label_path = tmp_list.replace('image', 'label')
                    label_path = label_path.split('.')[0]+'_label'+'.mat'
                    train_label_list.append(os.path.join(self.root, label_path))
        test_image_list, test_label_list = [], []
        with open('data_list/MTS12_test_image_list.txt', 'r') as file:  
            for line in file:  
                tmp_list = line.strip()  
                if tmp_list:  
                    test_image_list.append(os.path.join(self.root, tmp_list))
                    label_path = tmp_list.replace('image', 'label')
                    label_path = label_path.split('.')[0]+'_label'+'.mat'
                    test_label_list.append(os.path.join(self.root, label_path))

        assert len(train_image_list) == len(train_label_list)
        assert len(test_image_list) == len(test_label_list)

        # dataset
        traintransform = get_train_augmentation(self.cfg.data.re_img_size, seg_fill=self.cfg.data.IGNORE_LABEL)
        testtransform = get_augmentation(self.cfg.data.re_img_size)
        self.train_dataset = MTS12(image_file_list=train_image_list,
                                    label_file_list=train_label_list, transform=traintransform)
        self.test_dataset = MTS12(image_file_list=test_image_list,
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
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
