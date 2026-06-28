import torch
import numpy as np
from torch.utils.data import Dataset
from osgeo import gdal
import os

class DFC(Dataset):

    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    def __init__(self, image_file_list, label_file_list, transform=None, channel_last=False):
        self.image_file_list = image_file_list
        self.label_file_list = label_file_list
        self.channel_last = channel_last
        self.transform = transform
    # Statistics of samples of each class in the dataset
    def sample_stat(self):
        sample_per_class = torch.zeros([24])
        for label_file in self.label_file_list:
            label = gdal.Open(label_file, gdal.GA_ReadOnly)
            label = label.ReadAsArray()
            count = np.bincount(label.ravel(), minlength=25)
            count = count[1:25]
            count = torch.tensor(count)
            sample_per_class = sample_per_class + count

        return sample_per_class

    def remap_labels(self, label):
        mapping = {1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 7: 6, 9: 7, 10: 8}
        remapped_label = label.copy()

        for original_label, new_label in mapping.items():
            remapped_label[label == original_label] = new_label
        
        return remapped_label

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

        label = self.remap_labels(label)
        if(self.channel_last):
            image = image.transpose(1, 2, 0)
        
        # image = torch.tensor(image.astype(float), dtype=torch.float)  / 10000.0
        image = torch.tensor(band_normalization(image.astype(float)), dtype=torch.float)

        label = torch.tensor(label, dtype=torch.long).unsqueeze(0)

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
