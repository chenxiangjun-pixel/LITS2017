"""Pure inference script for 2D slice-based segmentation model."""
import os
import numpy as np
import torch
import SimpleITK as sitk
from net.ResUNet2D import build_resunet2d
import parameter2D as para

os.environ["CUDA_VISIBLE_DEVICES"] = para.gpu

model = torch.nn.DataParallel(build_resunet2d(training=False)).cuda()
model.load_state_dict(torch.load(para.module_path))
model.eval()

ct_dir = para.SUYU_path
os.makedirs(para.test_path, exist_ok=True)

with torch.no_grad():
    for idx, fname in enumerate(sorted(os.listdir(ct_dir))):
        ct = sitk.ReadImage(os.path.join(ct_dir, fname), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct).astype('float32')
        ct_array = ct_array / 200.0
        ct_tensor = torch.from_numpy(ct_array).unsqueeze(0).unsqueeze(0).cuda()

        output = model(ct_tensor)
        if output.shape[1] > 1:
            pred = torch.argmax(output, dim=1, keepdim=False)
        else:
            pred = (output >= para.threshold).long()

        pred_np = pred.squeeze().cpu().numpy().astype('uint8')
        mapped = np.zeros_like(pred_np, dtype=np.uint8)
        mapped[pred_np == 1] = 128
        mapped[pred_np == 2] = 255

        out_name = fname.replace('volume', 'pred') if 'volume' in fname else f"pred_{idx:04d}.png"
        sitk.WriteImage(sitk.GetImageFromArray(mapped),
                        os.path.join(para.test_path, out_name))
