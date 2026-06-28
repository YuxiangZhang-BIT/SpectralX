import kornia.augmentation as K
import os
import torch
import torchvision.transforms as T
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.datasets import WHUOHSST
from src.datamodules.augmentations import get_train_augmentation, get_augmentation

class WHUOHSSTDataModule(LightningDataModule):
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
        if self.cfg.data.SD_domain == 'ST':
            self.image_prefix =  ['S1','S2','S3','S4','S5','S6','S7','S8']
            self.test_image_prefix =  ['T1','T2','T3','T4','T5','T6','T7','T8']
            self.S_ori_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24] # 没有20
            self.T_ori_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24] # 没有18
        elif self.cfg.data.SD_domain == 'TS':
            self.test_image_prefix =  ['S1','S2','S3','S4','S5','S6','S7','S8']
            self.image_prefix =  ['T1','T2','T3','T4','T5','T6','T7','T8']
            self.T_ori_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24]
            self.S_ori_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24]

    def is_image_file(self, filename):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif'
        ]
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def setup(self, fit):

        if self.cfg.data.SD_domain == 'ST':
            data_path_train_image = os.path.join(self.root+'/source', 'image')
            data_path_val_image = os.path.join(self.root+'/source', 'image', 'val')
            data_path_test_image = os.path.join(self.root+'/target', 'image')
        elif self.cfg.data.SD_domain == 'TS':
            data_path_train_image = os.path.join(self.root+'/target', 'image')
            data_path_val_image = os.path.join(self.root+'/target', 'image', 'val')
            data_path_test_image = os.path.join(self.root+'/source', 'image')

        train_image_list = []
        train_label_list = []
        val_image_list = []
        val_label_list = []
        test_image_list = []
        test_label_list = []

        for index_i in ['tr', 'ts']:
            data_path_test_image_i = os.path.join(data_path_train_image, index_i)
            for root, paths, fnames in sorted(os.walk(data_path_test_image_i)):
                for fname in fnames:
                        image_path = os.path.join(
                                    data_path_test_image_i, fname)
                        label_path = image_path.replace('image', 'label')
                        train_image_list.append(image_path)
                        train_label_list.append(label_path)

        for root, paths, fnames in sorted(os.walk(data_path_val_image)):
            for fname in fnames:
                if self.is_image_file(fname):
                    for image_prefix_i in self.image_prefix:
                        if ((image_prefix_i + '_') in fname):
                            image_path = os.path.join(data_path_val_image, fname)
                            label_path = image_path.replace('image', 'label')
                            assert os.path.exists(label_path)
                            assert os.path.exists(image_path)
                            val_image_list.append(image_path)
                            val_label_list.append(label_path)

        for index_i in ['tr', 'val', 'ts']:
            data_path_test_image_i = os.path.join(data_path_test_image, index_i)
            for root, paths, fnames in sorted(os.walk(data_path_test_image_i)):
                for fname in fnames:
                    if self.is_image_file(fname):
                        for image_prefix_i in self.test_image_prefix:
                            if ((image_prefix_i + '_') in fname):
                                image_path = os.path.join(
                                    data_path_test_image_i, fname)
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
        self.train_dataset = WHUOHSST(image_file_list=train_image_list,
                                    label_file_list=train_label_list, ori_label=self.S_ori_label, test_ori_label=self.T_ori_label, transform=traintransform, domain='source')
        self.val_dataset = WHUOHSST(image_file_list=val_image_list,
                                    label_file_list=val_label_list, ori_label=self.S_ori_label, test_ori_label=self.T_ori_label, transform=valtransform, domain='source')
        self.test_dataset = WHUOHSST(image_file_list=test_image_list,
                                    label_file_list=test_label_list, ori_label=self.S_ori_label, test_ori_label=self.T_ori_label, transform=testtransform, domain='target')

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
