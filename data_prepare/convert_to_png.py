""" Convert 3D LiTS NIfTI volumes into 2D PNG slices for 2D training."""
import os
import sys
import shutil
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import scipy.ndimage as ndimage
import parameter2D as para
sys.path.append(os.path.split(sys.path[0])[0])

# LiTS数据集中原本的nii Mask是背景黑色、肝脏灰色、肿瘤白色，然而通过convert_to_png.py生成的2D slice中只有背景（黑色）和肝脏（白色）
# ct_out = r"D:\CXJ_code\Liver\LiTS_Tumor\data_prepare\2D_data\test_slice\ct"
# seg_out = r"D:\CXJ_code\Liver\LiTS_Tumor\data_prepare\2D_data\test_slice\seg"
ct_out = r"D:\CXJ_code\Liver\LiTS_Tumor\data_prepare\2D_data\train_slice\ct"
seg_out = r"D:\CXJ_code\Liver\LiTS_Tumor\data_prepare\2D_data\train_slice\seg"

# for file in tqdm(os.listdir(para.test_ct_path)):
#     ct = sitk.ReadImage(os.path.join(para.test_ct_path, file), sitk.sitkInt16)
#     seg = sitk.ReadImage(os.path.join(para.test_seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
for file in tqdm(os.listdir(para.train_ct_path)):
    ct = sitk.ReadImage(os.path.join(para.train_ct_path, file), sitk.sitkInt16)
    seg = sitk.ReadImage(os.path.join(para.train_seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)

    ct_array = sitk.GetArrayFromImage(ct)
    seg_array = sitk.GetArrayFromImage(seg)
    ct_array[ct_array > para.upper] = para.upper
    ct_array[ct_array < para.lower] = para.lower

    ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / para.slice_thickness, para.down_scale, para.down_scale), order=3)
    seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / para.slice_thickness, para.down_scale, para.down_scale), order=0)

    z = np.any(seg_array, axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]]
    start_slice = max(0, start_slice - para.expand_slice)
    end_slice = min(seg_array.shape[0] - 1, end_slice + para.expand_slice)

    ct_array = ct_array[start_slice:end_slice + 1]
    seg_array = seg_array[start_slice:end_slice + 1]

    base = file.replace('.nii', '')
    for i in range(ct_array.shape[0]):
        ct_slice = ct_array[i]
        seg_slice = seg_array[i]
        ct_slice = ((ct_slice - para.lower) / (para.upper - para.lower) * 255).astype(np.uint8)
        # map mask classes to 0/128/255 for background/liver/tumor visualization
        seg_slice = seg_slice.astype(np.uint8)
        seg_slice[seg_slice == 1] = 128
        seg_slice[seg_slice == 2] = 255
        ct_img = sitk.GetImageFromArray(ct_slice)
        seg_img = sitk.GetImageFromArray(seg_slice)
        ct_name = f"{base}_slice_{i:03d}.png"
        seg_name = f"{base.replace('volume', 'segmentation')}_slice_{i:03d}.png"
        ct_path = os.path.join(ct_out, ct_name)
        seg_path = os.path.join(seg_out, seg_name)
        sitk.WriteImage(ct_img, ct_path)
        sitk.WriteImage(seg_img, seg_path)
