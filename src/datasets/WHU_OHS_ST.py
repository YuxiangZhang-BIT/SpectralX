import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from osgeo import gdal
from pathlib import Path
from typing import Tuple
import os


class WHUOHSST(Dataset):

    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

    def __init__(self, image_file_list, label_file_list, ori_label, test_ori_label, domain, transform=None, channel_last=False):
        self.image_file_list = image_file_list
        self.label_file_list = label_file_list
        self.channel_last = channel_last
        self.ori_label = ori_label
        self.test_ori_label = test_ori_label
        self.domain = domain
        self.transform = transform

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, index):
        image_file = self.image_file_list[index]
        label_file = self.label_file_list[index]
        name = os.path.basename(image_file)
        image_dataset = gdal.Open(image_file, gdal.GA_ReadOnly)
        label_dataset = gdal.Open(label_file, gdal.GA_ReadOnly)

        image = image_dataset.ReadAsArray()
        label = label_dataset.ReadAsArray()

        set1 = set(self.ori_label)
        set2 = set(self.test_ori_label)
        common_label = list(set1 & set2)
        if self.domain == 'source':
            non_common_label = list(set1 - set2)
        elif self.domain == 'target':
            non_common_label = list(set2 - set1)
        non_label_map = {label_i: 0 for i, label_i in enumerate(non_common_label)}
        # mapped_labels = np.vectorize(non_label_map.get)(label)
        label_map = {label_i: i for i, label_i in enumerate(common_label)}
        label_map.update(non_label_map)
        mapped_labels = np.vectorize(label_map.get)(label)

        if(self.channel_last):
            image = image.transpose(1, 2, 0)

        # image = torch.tensor(image.astype(float), dtype=torch.float)  / 10000.0
        image = torch.tensor(band_normalization(image.astype(float)), dtype=torch.float)
        label = torch.tensor(mapped_labels).unsqueeze(0).to(torch.uint8)
        if self.transform:
            image, label = self.transform(torch.tensor(image), label)

        # label = torch.tensor(label, dtype=torch.long) - 1
        sample = {"image": image, "mask": label.long(), 'file_path': image_file}
        return sample

def band_normalization(data):
    """ normalize the matrix to (0,1), r.s.t A axis (Default=0)
        return normalized matrix and a record matrix for normalize back
    """
    size = data.shape

    for i in range(size[0]):
        _range = np.max(data[i, :, :]) - np.min(data[i, :, :])
        if _range != 0:
            data[i, :, :] = (data[i, :, :] - np.min(data[i, :, :])) / (_range+ 1e-8) if _range != 0 else data[i, :, :]
    return data

