import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from osgeo import gdal
from pathlib import Path
from typing import Tuple
import os
import scipy.io as scio

class MTS12ST(Dataset):

    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # 0 represents no data, 1 represents cultivated land, 2 represents forest, 3 represents grassland, 4 represents shrubland, 
    # 5 represents water, 6 represents wetland, 7 represents artificial surface, and 8 represents bare land.
    def __init__(self, image_file_list, label_file_list, transform=None, channel_last=False):
        self.image_file_list = image_file_list
        self.label_file_list = label_file_list
        self.channel_last = channel_last
        self.transform = transform

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, index):
        image_file = self.image_file_list[index]
        label_file = self.label_file_list[index]
        image_dataset = gdal.Open(image_file, gdal.GA_ReadOnly)
        image = image_dataset.ReadAsArray()
        label = scio.loadmat(label_file)['label']
        
        # image = torch.tensor(image.astype(float), dtype=torch.float)  / 10000.0
        image = torch.tensor(band_normalization(image.astype(float)/10000.0), dtype=torch.float)

        label = torch.tensor(label, dtype=torch.long).unsqueeze(0).to(torch.uint8)

        # label = torch.tensor(label).unsqueeze(0).to(torch.uint8)
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
