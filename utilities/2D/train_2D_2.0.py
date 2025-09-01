"""
改进要点：
1) 使用 random_split 进行 train/val 划分（默认 8:2，可在 parameter_3D.py 里通过 val_ratio 与 seed 调整）；
2) 每个 epoch 结束后在验证集上评估，评估以“主尺度输出”的损失为指标；
3) 仅当验证损失下降时保存 best checkpoint；
4) 训练/验证损失与学习率写入 ./weight/train_log.csv 便于绘图监控；
5) 自动识别主尺度输出（规则：若为 list/tuple 取空间分辨率最大的那个；若为 dict 优先 out/logits/pred/y/main/seg/seg_logits；若为 Tensor 直接使用）。
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


# ------------------------- 输出解析工具 -------------------------
def _area_like(x: torch.Tensor) -> int:
    """返回张量的空间面积（H*W），降维到 0 以避免未四维时出错。"""
    if not torch.is_tensor(x) or x.ndim < 4:
        return 0
    return int(x.shape[-1]) * int(x.shape[-2])


def unpack_outputs(outputs):
    """
    将网络输出统一为 list[tensor] 与 main_out，适配多种返回类型：
    - list/tuple: 过滤 None，选择空间分辨率最大的作为主输出
    - dict: 优先 'out'/'logits'/'pred'/'y'/'main'/'seg'/'seg_logits'，否则取 values 中空间分辨率最大的
    - tensor: 直接作为主输出
    返回： (out_list, main_out)
    """
    # tensor
    if torch.is_tensor(outputs):
        return [outputs], outputs

    # list/tuple
    if isinstance(outputs, (list, tuple)):
        outs = [o for o in outputs if torch.is_tensor(o)]
        if len(outs) == 0:
            raise ValueError("网络返回的 list/tuple 中没有有效的张量输出。")
        main_out = max(outs, key=_area_like)
        return outs, main_out

    # dict
    if isinstance(outputs, dict):
        preferred_keys = ['out', 'logits', 'pred', 'y', 'main', 'seg', 'seg_logits']
        for k in preferred_keys:
            if k in outputs and torch.is_tensor(outputs[k]):
                v = outputs[k]
                # 在 dict 的情况下，我们仍返回所有张量以便深度监督
                outs = [t for t in outputs.values() if torch.is_tensor(t)]
                if len(outs) == 0:
                    outs = [v]
                return outs, v
        # 退化：取所有张量，选分辨率最大者
        outs = [t for t in outputs.values() if torch.is_tensor(t)]
        if len(outs) == 0:
            raise ValueError("网络返回的 dict 中没有有效的张量输出。")
        main_out = max(outs, key=_area_like)
        return outs, main_out

    raise TypeError(f"不支持的输出类型：{type(outputs)}")


# ------------------------- 训练/验证评估函数 -------------------------
def evaluate(model, loader, loss_func):
    """在验证集上评估主尺度损失。"""
    model.eval()
    loss_list = []
    with torch.no_grad():
        for ct, seg in loader:
            ct = ct.cuda(non_blocking=True)
            seg = seg.cuda(non_blocking=True)
            outputs = model(ct)
            outs, main_out = unpack_outputs(outputs)
            val_loss = loss_func(main_out, seg)
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
    # 构建完整数据集
    full_ds = SliceDataset(os.path.join(para.training_set_2d_path, 'ct'), os.path.join(para.training_set_2d_path, 'seg'))
    # 划分训练/验证集
    val_ratio = getattr(para, 'val_ratio', 0.2)
    seed = getattr(para, 'seed', 42)
    train_size = int(len(full_ds) * (1 - val_ratio))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    print(f'[Data] 总样本: {len(full_ds)} | 训练: {train_size} | 验证: {val_size} (seed={seed})')

    # DataLoader
    train_dl = DataLoader(train_ds, batch_size=para.batch_size, shuffle=True, num_workers=para.num_workers, pin_memory=para.pin_memory)
    val_dl = DataLoader(val_ds, batch_size=para.batch_size, shuffle=False, num_workers=para.num_workers, pin_memory=para.pin_memory)

    # 损失函数与优化器
    loss_func_list = [DiceLoss(), ELDiceLoss(), WCELoss(), JaccardLoss(), SSLoss(), TverskyLoss(), HybridLoss(), BCELoss()]
    loss_func = loss_func_list[5]  # TverskyLoss（与原脚本一致）
    opt = torch.optim.Adam(model.parameters(), lr=para.learning_rate)
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, para.learning_rate_decay)
    alpha = para.alpha

    # 目录与日志
    os.makedirs('../../weight', exist_ok=True)
    log_path = '../../weight/2D/train_log.csv'
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('epoch,lr,train_main_loss,val_main_loss,best_val\n')

    # ------------------------- 训练主循环 -------------------------
    best_val = float('inf')
    start = time()
    for epoch in range(para.Epoch):
        lr_decay.step()
        train_main_losses = []

        for step, (ct, seg) in enumerate(train_dl):
            ct = ct.cuda(non_blocking=True)    # ([8, 1, 256, 256])
            seg = seg.cuda(non_blocking=True)  # ([8, 256, 256])
            # print("ct.shape = ", ct.shape)
            # print("seg.shape = ", seg.shape)
            outputs = model(ct)
            outs, main_out = unpack_outputs(outputs)

            # 计算深度监督损失（除了主输出以外的所有输出 * alpha）
            ds_loss = 0.0
            if len(outs) > 1:
                for o in outs:
                    if o is main_out:
                        continue
                    ds_loss = ds_loss + loss_func(o, seg)
                ds_loss = ds_loss * alpha

            main_loss = loss_func(main_out, seg)
            loss = ds_loss + main_loss

            train_main_losses.append(main_loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 500 == 0:
                print('epoch:{}, step:{}, main_loss:{:.4f}, ds_loss:{:.4f}, total:{:.4f}, time:{:.2f} min'.format(
                    epoch, step, main_loss.item(), float(ds_loss) if torch.is_tensor(ds_loss) else ds_loss, loss.item(), (time() - start) / 60))

        # ---- epoch 结束：验证 ----
        train_main_loss = sum(train_main_losses) / max(len(train_main_losses), 1)
        val_main_loss = evaluate(model, val_dl, loss_func)
        current_lr = opt.param_groups[0]['lr']

        print('Epoch {:03d} | lr:{:.6f} | train_main_loss:{:.6f} | val_main_loss:{:.6f} | best_val:{:.6f} | time:{:.2f} min'.format(
            epoch, current_lr, train_main_loss, val_main_loss, best_val, (time() - start) / 60))
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write('{},{:.8f},{:.6f},{:.6f},{:.6f}\n'.format(
                epoch, current_lr, train_main_loss, val_main_loss, min(best_val, val_main_loss)))

        # 仅当验证损失改善时保存
        if val_main_loss < best_val:
            best_val = val_main_loss
            save_path = './weight/2D/2Dnet-best-epoch{}-train{:.4f}-val{:.4f}.pth'.format(epoch, train_main_loss, val_main_loss)
            torch.save(model.state_dict(), save_path)
            print('[Checkpoint] 验证损失改善，已保存最佳模型到: {}'.format(save_path))

        # 衰减 alpha（保持与原始逻辑一致）
        if epoch % 25 == 0 and epoch != 0:
            alpha *= 0.8
            print('[Alpha] 深度监督系数衰减至: {:.4f}'.format(alpha))
        print("-----------------------------------------------------------------------------------------------------")

