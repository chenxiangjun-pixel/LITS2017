"""Torch dataset for 2D slice based training."""
import os
import sys
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset as dataset
import parameter2D as para
sys.path.append(os.path.split(sys.path[0])[0])


class SliceDataset(dataset):
    def __init__(self, ct_dir, seg_dir):
        self.ct_list = sorted(os.listdir(ct_dir))
        self.seg_list = [i.replace('volume', 'segmentation') for i in self.ct_list]
        self.ct_list = [os.path.join(ct_dir, i) for i in self.ct_list]
        self.seg_list = [os.path.join(seg_dir, i) for i in self.seg_list]

    def __getitem__(self, index):
        ct = sitk.ReadImage(self.ct_list[index], sitk.sitkInt16)
        seg = sitk.ReadImage(self.seg_list[index], sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct).astype(np.float32)
        seg_array = sitk.GetArrayFromImage(seg).astype(np.uint8)

        # map PNG mask values (0,128,255) to class labels 0/1/2
        seg_array[seg_array == 128] = 1
        seg_array[seg_array == 255] = 2

        ct_array = ct_array / 200.0

        ct_tensor = torch.from_numpy(ct_array).unsqueeze(0)
        # CrossEntropyLoss expects integer labels without channel dimension
        seg_tensor = torch.from_numpy(seg_array).long()
        return ct_tensor, seg_tensor

    def __len__(self):
        return len(self.ct_list)
