import os
from time import time
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from dataset3D import Dataset
from net.ResUNet import net
import parameter_3D as para

# 设置显卡相关
os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu
cudnn.benchmark = para.cudnn_benchmark

if __name__ == '__main__':
    net = torch.nn.DataParallel(net).cuda()
    net.train()
    train_ds = Dataset(os.path.join(para.training_set_path, 'ct'), os.path.join(para.training_set_path, 'seg'))
    train_dl = DataLoader(train_ds, para.batch_size, True, num_workers=para.num_workers, pin_memory=para.pin_memory)
    # loss_func_list = [DiceLoss(), ELDiceLoss(), WCELoss(), JaccardLoss(), SSLoss(), TverskyLoss(), HybridLoss(), BCELoss()]
    # loss_func = loss_func_list[5]
    # 使用交叉熵损失进行多类别分割
    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=para.learning_rate)
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, para.learning_rate_decay)
    alpha = para.alpha
    # 训练网络
    start = time()
    for epoch in range(para.Epoch):
        lr_decay.step()
        mean_loss = []
        for step, (ct, seg) in enumerate(train_dl):
            ct = ct.cuda()
            seg = seg.cuda().long()
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
            if step % 5 == 0:
                print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, time:{:.3f} min'
                      .format(epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), (time() - start) / 60))

        # 在每个epoch结束后调用学习率调度器
        lr_decay.step()
        mean_loss = sum(mean_loss) / len(mean_loss)
        # 保存模型
        if epoch % 50 == 0 and epoch != 0:
            # 网络模型的命名方式为：epoch轮数+当前minibatch的loss+本轮epoch的平均loss
            torch.save(net.state_dict(), './weight/3D/net{}-{:.3f}-{:.3f}.pth'.format(epoch, loss, mean_loss))
        # 对深度监督系数进行衰减
        if epoch % 40 == 0 and epoch != 0:
            alpha *= 0.8
