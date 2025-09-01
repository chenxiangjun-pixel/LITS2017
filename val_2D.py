import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import SimpleITK as sitk
from dataset.dataset2D import SliceDataset
from net.ResUNet2D import ResUNet2D
import parameter2D as para


def dice_coefficient(pred, target):
    """Compute Dice coefficient for a single 2D mask (binary)."""
    intersection = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item()
    return 2.0 * intersection / (union + 1e-5), intersection, union


def jaccard_index(pred, target):
    """Compute Jaccard index (IoU) for a single 2D mask (binary)."""
    intersection = (pred * target).sum().item()
    union = ((pred + target) > 0).float().sum().item()
    return intersection / (union + 1e-5)


def _stats_excluding_zeros(df, exclude_cols=None):
    """
    Compute mean/std/min/max per column, excluding zeros as outliers.
    Columns in exclude_cols (e.g., 'time') keep their original stats without excluding zero.
    """
    if exclude_cols is None:
        exclude_cols = []
    stats = pd.DataFrame(index=['mean', 'std', 'min', 'max'], columns=df.columns)
    for col in df.columns:
        col_series = df[col]
        if col in exclude_cols:
            # keep as-is
            stats.loc['mean', col] = col_series.mean()
            stats.loc['std',  col] = col_series.std()
            stats.loc['min',  col] = col_series.min()
            stats.loc['max',  col] = col_series.max()
        else:
            # treat 0 as anomaly -> drop zeros
            nonzero = col_series.replace(0, np.nan)
            stats.loc['mean', col] = nonzero.mean(skipna=True)
            stats.loc['std',  col] = nonzero.std(skipna=True)
            stats.loc['min',  col] = nonzero.min(skipna=True)
            stats.loc['max',  col] = nonzero.max(skipna=True)
    return stats


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = para.gpu

    model = torch.nn.DataParallel(ResUNet2D(training=False)).cuda()
    model.load_state_dict(torch.load(para.module_path))
    model.eval()

    os.makedirs(para.val_path, exist_ok=True)

    ds = SliceDataset(os.path.join(para.test_2d_path, "ct"), os.path.join(para.test_2d_path, "seg"))
    dl = DataLoader(ds, 1, False, num_workers=para.num_workers, pin_memory=para.pin_memory)

    file_names = []
    # liver metrics (foreground > 0)
    liver_dice_scores = []
    liver_jaccard_scores = []
    # tumor metrics (class == 2)
    tumor_dice_scores = []
    tumor_jaccard_scores = []

    time_per_slice = []

    # global dice accumulators
    liver_intersection_global = 0.0
    liver_union_global = 0.0
    tumor_intersection_global = 0.0
    tumor_union_global = 0.0

    with torch.no_grad():
        for idx, (ct, seg) in enumerate(dl):
            start_t = time.time()
            ct = ct.cuda()
            seg = seg.cuda()
            if seg.dim() == 3:
                seg = seg.unsqueeze(1)  # ensure Bx1xHxW

            outputs = model(ct)
            if outputs.shape[1] == 1:
                pred = (outputs >= para.threshold).long()
            else:
                pred = torch.argmax(outputs, dim=1, keepdim=True)

            # liver (foreground) masks
            liver_pred = (pred > 0).float()
            liver_gt = (seg > 0).float()

            # tumor masks (class == 2)
            tumor_pred = (pred == 2).float()
            tumor_gt = (seg == 2).float()

            # liver metrics
            d_liver, inter_l, union_l = dice_coefficient(liver_pred, liver_gt)
            j_liver = jaccard_index(liver_pred, liver_gt)
            liver_dice_scores.append(d_liver)
            liver_jaccard_scores.append(j_liver)
            liver_intersection_global += inter_l
            liver_union_global += union_l

            # tumor metrics
            d_tumor, inter_t, union_t = dice_coefficient(tumor_pred, tumor_gt)
            j_tumor = jaccard_index(tumor_pred, tumor_gt)
            tumor_dice_scores.append(d_tumor)
            tumor_jaccard_scores.append(j_tumor)
            tumor_intersection_global += inter_t
            tumor_union_global += union_t

            time_per_slice.append(time.time() - start_t)

            # 获取输入文件名（不含扩展名）
            input_filename = os.path.splitext(os.path.basename(ds.ct_list[idx]))[0]
            file_names.append(input_filename)

            # save prediction as png with 0/128/255 mapping
            pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)
            save_np = np.zeros_like(pred_np, dtype=np.uint8)
            save_np[pred_np == 1] = 128     # liver
            save_np[pred_np == 2] = 255     # tumor
            output_filename = f"{input_filename}.png"
            sitk.WriteImage(sitk.GetImageFromArray(save_np),
                            os.path.join(para.val_path, output_filename))

    # Build DataFrames
    liver_df = pd.DataFrame({
        'dice': liver_dice_scores,
        'jaccard': liver_jaccard_scores,
        'time': time_per_slice
    }, index=file_names)

    tumor_df = pd.DataFrame({
        'dice': tumor_dice_scores,
        'jaccard': tumor_jaccard_scores
    }, index=file_names)

    # Compute statistics (exclude 0 as anomaly for metric columns; keep time as-is)
    liver_stats = _stats_excluding_zeros(liver_df, exclude_cols=['time'])
    tumor_stats = _stats_excluding_zeros(tumor_df, exclude_cols=[])

    # Print quick summary (mean ± std) using the zero-excluded policy for metric columns
    print("Liver Dice (excl. 0): {:.4f} ± {:.4f}".format(
        liver_df['dice'].replace(0, np.nan).mean(skipna=True),
        liver_df['dice'].replace(0, np.nan).std(skipna=True)))
    print("Liver Jaccard (excl. 0): {:.4f} ± {:.4f}".format(
        liver_df['jaccard'].replace(0, np.nan).mean(skipna=True),
        liver_df['jaccard'].replace(0, np.nan).std(skipna=True)))

    print("Tumor Dice (excl. 0): {:.4f} ± {:.4f}".format(
        tumor_df['dice'].replace(0, np.nan).mean(skipna=True),
        tumor_df['dice'].replace(0, np.nan).std(skipna=True)))
    print("Tumor Jaccard (excl. 0): {:.4f} ± {:.4f}".format(
        tumor_df['jaccard'].replace(0, np.nan).mean(skipna=True),
        tumor_df['jaccard'].replace(0, np.nan).std(skipna=True)))

    # Print global dice for liver and tumor
    print('liver dice global:', liver_intersection_global / (liver_union_global + 1e-5))
    print('tumor dice global:', tumor_intersection_global / (tumor_union_global + 1e-5))

    # Write to Excel following val_3D style: liver/liver_statistics & tumor/tumor_statistics
    writer = pd.ExcelWriter(r'D:\CXJ_code\Liver\LiTS_Tumor\result\2D_result\result_2d.xlsx')
    liver_df.to_excel(writer, 'liver')
    liver_stats.to_excel(writer, 'liver_statistics')
    tumor_df.to_excel(writer, 'tumor')
    tumor_stats.to_excel(writer, 'tumor_statistics')
    writer.save()


if __name__ == "__main__":
    main()
