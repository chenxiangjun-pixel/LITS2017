import os
from time import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split
from dataset.dataset2D import SliceDataset
from net.ResUNet2D import build_resunet2d
import parameter2D as para


def main() -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu
    cudnn.benchmark = para.cudnn_benchmark

    model = torch.nn.DataParallel(build_resunet2d(training=True)).cuda()
    class_weights = torch.tensor([1.0, 1.0, 2.0], device='cuda')
    loss_func = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.Adam(model.parameters(), lr=para.learning_rate)
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, para.learning_rate_decay)
    alpha = para.alpha

    full_ds = SliceDataset(os.path.join(para.training_set_2d_path, 'ct'), os.path.join(para.training_set_2d_path, 'seg'))
    val_len = max(1, int(0.2 * len(full_ds)))
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, para.batch_size, True, num_workers=para.num_workers, pin_memory=para.pin_memory)
    val_dl = DataLoader(val_ds, para.batch_size, False, num_workers=para.num_workers, pin_memory=para.pin_memory)

    start = time()
    best_val = float('inf')
    os.makedirs('./weight/2D', exist_ok=True)

    for epoch in range(para.Epoch):
        model.train()
        mean_loss, train_ds_losses = [], []

        for step, (ct, seg) in enumerate(train_dl):
            ct = ct.cuda()
            seg = seg.cuda().long()
            outputs = model(ct)

            loss1 = loss_func(outputs[0], seg)
            loss2 = loss_func(outputs[1], seg)
            loss3 = loss_func(outputs[2], seg)
            loss4 = loss_func(outputs[3], seg)

            ds_loss = (loss1 + loss2 + loss3) * alpha
            main_loss = loss4
            loss = ds_loss + main_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            mean_loss.append(main_loss.item())
            train_ds_losses.append(ds_loss.item())

            if step % 500 == 0:
                print('epoch:{}, step:{}, main_loss:{:.4f}, ds_loss:{:.4f}, total:{:.4f}, time:{:.2f} min'
                      .format(epoch, step, main_loss.item(), ds_loss.item(), loss.item(), (time() - start) / 60))

        train_main_loss = sum(mean_loss) / len(mean_loss)

        model.eval()
        val_main_losses = []
        with torch.no_grad():
            for ct, seg in val_dl:
                ct = ct.cuda()
                seg = seg.cuda().long()
                outputs = model(ct)
                pred = outputs[3] if isinstance(outputs, (tuple, list)) else outputs
                val_main_losses.append(loss_func(pred, seg).item())
        val_main_loss = sum(val_main_losses) / len(val_main_losses)

        lr_now = opt.param_groups[0]['lr']
        print('Epoch {:03d} | lr:{:.6f} | train_main_loss:{:.6f} | val_main_loss:{:.6f} | best_val:{:.6f} | time:{:.2f} min'
              .format(epoch, lr_now, train_main_loss, val_main_loss, best_val, (time() - start) / 60))

        if val_main_loss < best_val:
            best_val = val_main_loss
            ckpt = './weight/2D/2Dnet-best-epoch{}-train{:.4f}-val{:.4f}.pth'.format(epoch, train_main_loss, val_main_loss)
            torch.save(model.state_dict(), ckpt)
            print('[Checkpoint] 验证损失改善，已保存最佳模型到: {}'.format(ckpt))

        lr_decay.step()
        print('-' * 100)


if __name__ == '__main__':
    main()
