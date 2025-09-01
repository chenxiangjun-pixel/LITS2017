# -*- coding: utf-8 -*-
"""
功能增强版 train_2D_2.0.py：
1) 使用 random_split 将完整数据集划分为训练集与验证集（默认 8:2，可在 parameter_3D.py 中通过 val_ratio 覆盖）；
2) 每个 epoch 结束后在验证集上评估主尺度损失（与训练中 loss4 对应）以监控泛化；
3) 仅在验证损失改善时保存模型（best checkpoint），不再按固定周期保存；
4) 标准输出同步打印训练/验证损失与当前学习率；同时将日志落盘到 ./weight/train_log.csv 便于后续绘图或监控；
5) 仍保留多尺度深度监督与 alpha 衰减策略（每 40 个 epoch 衰减 0.8，保持与原训练逻辑一致）。

说明：
- 为确保划分可复现，默认使用 seed=42（可在 parameter_3D.py 中定义 seed 覆盖）。
- 验证集评估仅使用主尺度输出 outputs[3]，与训练统计的 mean_loss 一致。
- 模型最佳权重保存路径类似：./weight/2Dnet-best-epoch{E}-train{T:.3f}-val{V:.3f}.pth
"""

import os
from time import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split

from dataset.dataset2D import SliceDataset
from loss.Dice import DiceLoss
from loss.ELDice import ELDiceLoss
from loss.WBCE import WCELoss
from loss.Jaccard import JaccardLoss
from loss.SS import SSLoss
from loss.Tversky import TverskyLoss
from loss.Hybrid import HybridLoss
from loss.BCE import BCELoss
from net.ResUNet2D import net
import parameter2D as para


# ------------------------- 训练/验证评估函数 -------------------------
def evaluate(model, loader, loss_func):
    """
    使用验证集评估主尺度损失（outputs[3]）
    参数:
        model: 已在 CUDA 且可能被 DataParallel 包裹的网络
        loader: 验证集 DataLoader（不打乱顺序）
        loss_func: 与训练一致的损失函数实例
    返回:
        平均验证主尺度损失(浮点数)
    """
    model.eval()
    loss_list = []
    with torch.no_grad():
        for ct, seg in loader:
            ct = ct.cuda(non_blocking=True)
            seg = seg.cuda(non_blocking=True)
            outputs = model(ct)
            val_loss = loss_func(outputs[3], seg)  # 主尺度输出
            loss_list.append(val_loss.item())
    model.train()
    return sum(loss_list) / max(len(loss_list), 1)


if __name__ == '__main__':
    # ------------------------- CUDA & CUDNN 设置 -------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu
    cudnn.benchmark = para.cudnn_benchmark

    # ------------------------- 构建模型与优化器 -------------------------
    model = torch.nn.DataParallel(net).cuda()
    model.train()

    # 构建完整数据集（保持原始目录结构不变）
    full_ds = SliceDataset(
        os.path.join(para.training_set_2d_path, 'ct'),
        os.path.join(para.training_set_2d_path, 'seg')
    )

    # 划分训练/验证集（可在 parameter_3D.py 中设置 val_ratio 与 seed）
    val_ratio = getattr(para, 'val_ratio', 0.2)  # 默认 20% 作为验证集
    seed = getattr(para, 'seed', 42)
    train_size = int(len(full_ds) * (1 - val_ratio))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(
        full_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    print(f'[Data] 总样本: {len(full_ds)} | 训练: {train_size} | 验证: {val_size} (seed={seed})')

    # DataLoader（训练集打乱，验证集不打乱；pin_memory 与 num_workers 继承原参数）
    train_dl = DataLoader(
        train_ds, batch_size=para.batch_size, shuffle=True,
        num_workers=para.num_workers, pin_memory=para.pin_memory
    )
    val_dl = DataLoader(
        val_ds, batch_size=para.batch_size, shuffle=False,
        num_workers=para.num_workers, pin_memory=para.pin_memory
    )

    # 损失函数与优化器（与原脚本保持一致）
    loss_func_list = [DiceLoss(), ELDiceLoss(), WCELoss(), JaccardLoss(), SSLoss(), TverskyLoss(), HybridLoss(), BCELoss()]
    loss_func = loss_func_list[5]  # 使用 TverskyLoss（原脚本下标为 5）
    opt = torch.optim.Adam(model.parameters(), lr=para.learning_rate)
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, para.learning_rate_decay)
    alpha = para.alpha  # 深度监督系数

    # 目录与日志文件
    os.makedirs('../../weight', exist_ok=True)
    log_path = '../../weight/2D/train_log.csv'
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('epoch,lr,train_main_loss,val_main_loss,best_val\n')

    # ------------------------- 训练主循环 -------------------------
    best_val = float('inf')
    start = time()
    for epoch in range(para.Epoch):
        lr_decay.step()
        train_main_losses = []  # 仅记录主尺度 loss4，方便与验证保持一致

        for step, (ct, seg) in enumerate(train_dl):
            ct = ct.cuda(non_blocking=True)
            seg = seg.cuda(non_blocking=True)

            # 前向传播（多尺度输出）
            outputs = model(ct)
            loss1 = loss_func(outputs[0], seg)
            loss2 = loss_func(outputs[1], seg)
            loss3 = loss_func(outputs[2], seg)
            loss4 = loss_func(outputs[3], seg)  # 主尺度
            loss = (loss1 + loss2 + loss3) * alpha + loss4

            train_main_losses.append(loss4.item())

            # 反向与更新
            opt.zero_grad()
            loss.backward()
            opt.step()

            # 可选：大步长打印（避免过度刷屏）
            if step % 500 == 0:
                print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4(main):{:.3f}, time:{:.3f} min'.format(
                    epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), (time() - start) / 60))

        # ---- 每个 epoch 结束：统计训练主尺度损失并进行验证 ----
        train_main_loss = sum(train_main_losses) / max(len(train_main_losses), 1)
        val_main_loss = evaluate(model, val_dl, loss_func)
        current_lr = opt.param_groups[0]['lr']

        # 打印并记录日志（用于监控过拟合：train vs. val）
        print('Epoch {:03d} | lr:{:.6f} | train_main_loss:{:.4f} | val_main_loss:{:.4f} | best_val:{:.4f} | time:{:.2f} min'.format(
            epoch, current_lr, train_main_loss, val_main_loss, best_val, (time() - start) / 60))
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write('{},{:.8f},{:.6f},{:.6f},{:.6f}\n'.format(epoch, current_lr, train_main_loss, val_main_loss, min(best_val, val_main_loss)))

        # 仅在验证损失改善时保存模型（best checkpoint）
        if val_main_loss < best_val:
            best_val = val_main_loss
            save_path = './weight/2Dnet-best-epoch{}-train{:.3f}-val{:.3f}.pth'.format(epoch, train_main_loss, val_main_loss)
            torch.save(model.state_dict(), save_path)
            print('[Checkpoint] 验证损失改善，已保存最佳模型到: {}'.format(save_path))

        # 衰减深度监督系数 alpha（保持与原脚本一致）
        if epoch % 40 == 0 and epoch != 0:
            alpha *= 0.8
            print('[Alpha] 深度监督系数衰减至: {:.4f}'.format(alpha))

