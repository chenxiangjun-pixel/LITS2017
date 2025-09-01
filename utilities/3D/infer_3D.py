import os
import torch
import numpy as np
import scipy.ndimage as ndimage
import SimpleITK as sitk
import skimage.measure as measure
import skimage.morphology as morphology

from net.ResUNet import ResUNet
import parameter_3D as para

# 指定使用的 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu

# 定义网络并加载参数
net = torch.nn.DataParallel(ResUNet(training=False)).cuda()
net.load_state_dict(torch.load(para.module_path))
net.eval()

# 输入与输出路径
input_dir = para.SUYU_path
output_dir = para.test_path
os.makedirs(output_dir, exist_ok=True)

for file_index, file in enumerate(os.listdir(input_dir)):
    print(f"Processing {file_index}: {file}")

    # 读取 CT 体数据
    ct = sitk.ReadImage(os.path.join(input_dir, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    origin_shape = ct_array.shape

    # 灰度截断并归一化
    ct_array[ct_array > para.upper] = para.upper
    ct_array[ct_array < para.lower] = para.lower
    ct_array = ct_array.astype(np.float32) / 200.0

    # 体素降采样
    ct_array = ndimage.zoom(ct_array, (1, para.down_scale, para.down_scale), order=3)

    # 若切片数量不足，进行 padding
    too_small = False
    if ct_array.shape[0] < para.size:
        depth = ct_array.shape[0]
        tmp = np.ones((para.size, int(512 * para.down_scale), int(512 * para.down_scale)), dtype=np.float32) * para.lower
        tmp[:depth] = ct_array
        ct_array = tmp
        too_small = True

    # 滑动窗口推理
    start_slice = 0
    end_slice = start_slice + para.size - 1
    count = np.zeros((ct_array.shape[0], 512, 512), dtype=np.int16)
    probability_map = np.zeros((3, ct_array.shape[0], 512, 512), dtype=np.float32)

    with torch.no_grad():
        while end_slice < ct_array.shape[0]:
            ct_tensor = torch.from_numpy(ct_array[start_slice:end_slice + 1]).unsqueeze(0).unsqueeze(0).cuda()
            outputs = net(ct_tensor)
            outputs_np = outputs.cpu().numpy()[0]
            probability_map[:, start_slice:end_slice + 1] += outputs_np
            count[start_slice:end_slice + 1] += 1

            del outputs
            start_slice += para.stride
            end_slice = start_slice + para.size - 1

        if end_slice != ct_array.shape[0] - 1:
            end_slice = ct_array.shape[0] - 1
            start_slice = end_slice - para.size + 1
            ct_tensor = torch.from_numpy(ct_array[start_slice:end_slice + 1]).unsqueeze(0).unsqueeze(0).cuda()
            outputs = net(ct_tensor)
            outputs_np = outputs.cpu().numpy()[0]
            probability_map[:, start_slice:end_slice + 1] += outputs_np
            count[start_slice:end_slice + 1] += 1
            del outputs

    probability_map = probability_map / count[np.newaxis, ...]
    # argmax 得到的类别索引为: 0=背景, 1=肝脏, 2=肿瘤
    pred_seg = np.argmax(probability_map, axis=0).astype(np.uint8)

    if too_small:
        pred_seg = pred_seg[:depth]

    # 对肝脏进行最大连通域提取并填充空洞
    liver_seg = (pred_seg == 1).astype(np.uint8)
    liver_seg = measure.label(liver_seg, 4)
    props = measure.regionprops(liver_seg)
    if props:
        max_area = 0
        max_index = 0
        for index, prop in enumerate(props, start=1):
            if prop.area > max_area:
                max_area = prop.area
                max_index = index
        liver_seg[liver_seg != max_index] = 0
        liver_seg[liver_seg == max_index] = 1
        liver_seg = liver_seg.astype(bool)
        morphology.remove_small_holes(liver_seg, para.maximum_hole, connectivity=2, in_place=True)
        liver_seg = liver_seg.astype(np.uint8)

    tumor_seg = (pred_seg == 2).astype(np.uint8)
    tumor_seg[liver_seg == 0] = 0

    final_seg = liver_seg.copy()
    final_seg[tumor_seg == 1] = 2

    # 将类别值转换为固定灰度: 背景=0, 肝脏=128, 肿瘤=255
    save_seg = np.zeros_like(pred_seg, dtype=np.uint8)
    save_seg[liver_seg == 1] = 128
    save_seg[tumor_seg == 1] = 255

    pred_sitk = sitk.GetImageFromArray(save_seg)
    pred_sitk = sitk.Cast(pred_sitk, sitk.sitkUInt8)
    pred_sitk.SetDirection(ct.GetDirection())
    pred_sitk.SetOrigin(ct.GetOrigin())
    pred_sitk.SetSpacing(ct.GetSpacing())

    out_name = file.replace('volume', 'pred') if 'volume' in file else f'pred-{file}'
    sitk.WriteImage(pred_sitk, os.path.join(output_dir, out_name))
    print(f"Saved: {out_name}")
