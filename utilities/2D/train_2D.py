import os
from time import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
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


os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu
cudnn.benchmark = para.cudnn_benchmark
if __name__ == '__main__':
    net = torch.nn.DataParallel(net).cuda()
    net.train()
    train_ds = SliceDataset(os.path.join(para.training_set_2d_path, 'ct'), os.path.join(para.training_set_2d_path, 'seg'))
    train_dl = DataLoader(train_ds, para.batch_size, True, num_workers=para.num_workers, pin_memory=para.pin_memory)
    loss_func_list = [DiceLoss(), ELDiceLoss(), WCELoss(), JaccardLoss(), SSLoss(), TverskyLoss(), HybridLoss(), BCELoss()]
    loss_func = loss_func_list[5]
    opt = torch.optim.Adam(net.parameters(), lr=para.learning_rate)
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, para.learning_rate_decay)
    alpha = para.alpha

    start = time()
    for epoch in range(para.Epoch):
        lr_decay.step()
        mean_loss = []
        for step, (ct, seg) in enumerate(train_dl):
            ct = ct.cuda()
            seg = seg.cuda()
            outputs = net(ct)
            loss1 = loss_func(outputs[0], seg)
            loss2 = loss_func(outputs[1], seg)
            loss3 = loss_func(outputs[2], seg)
            loss4 = loss_func(outputs[3], seg)
            loss = (loss1 + loss2 + loss3) * alpha + loss4
            mean_loss.append(loss4.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 100 == 0:
                print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, time:{:.3f} min'.format(
                        epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), (time() - start) / 60))

        # 统计当前 epoch 的平均主尺度损失
        mean_loss = sum(mean_loss) / len(mean_loss)
        # 按周期保存模型权重（保持与原脚本一致的命名与目录结构）
        if epoch % 50 == 0 and epoch != 0:
            os.makedirs('../module', exist_ok=True)
            torch.save(net.state_dict(), './weight/2Dnet{}-{:.3f}-{:.3f}.pth'.format(epoch, loss, mean_loss))
        # 衰减深度监督系数 alpha（保持与原脚本一致）
        if epoch % 40 == 0 and epoch != 0:
            alpha *= 0.8
