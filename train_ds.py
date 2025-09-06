"""

训练脚本
"""

import os
from time import time

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader

from visdom import Visdom

from dataset.dataset import Dataset

# 仍然保留其他损失的导入以便根据需要切换
from loss.Dice import DiceLoss
from loss.ELDice import ELDiceLoss
from loss.WBCE import WCELoss
from loss.Jaccard import JaccardLoss
from loss.SS import SSLoss
from loss.Tversky import TverskyLoss
from loss.Hybrid import HybridLoss
from loss.BCE import BCELoss

from net.ResUNet import net

import parameter as para

# 设置visdom
viz = Visdom(port=666)
step_list = [0]
win = viz.line(X=np.array([0]), Y=np.array([1.0]), opts=dict(title='loss'))

# 设置显卡相关
os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu
cudnn.benchmark = para.cudnn_benchmark

# 定义网络
net = torch.nn.DataParallel(net).cuda()
net.train()

# 定义Dateset
train_ds = Dataset(os.path.join(para.training_set_path, 'ct'), os.path.join(para.training_set_path, 'seg'))

# 定义数据加载
train_dl = DataLoader(train_ds, para.batch_size, True, num_workers=para.num_workers, pin_memory=para.pin_memory)

# 使用加权交叉熵提升肿瘤类别的损失权重
class_weights = torch.FloatTensor([1.0, 2.0]).cuda()  # 肝脏/背景, 肿瘤
ce_loss = nn.CrossEntropyLoss(weight=class_weights)


def loss_func(pred, target):
    """计算带权重的交叉熵损失

    由于网络输出为单通道概率图，这里将其转换为两通道形式，再
    使用带权重的交叉熵，从而对肿瘤类别施加更大的惩罚。
    """
    pred_two = torch.cat((1 - pred, pred), dim=1)
    return ce_loss(pred_two, target.long())

# 定义优化器
opt = torch.optim.Adam(net.parameters(), lr=para.learning_rate)

# 学习率衰减
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, para.learning_rate_decay)

# 深度监督衰减系数
alpha = para.alpha

# 训练网络
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

        if step % 5 is 0:
            
            step_list.append(step_list[-1] + 1)
            viz.line(X=np.array([step_list[-1]]), Y=np.array([loss4.item()]), win=win, update='append')
            
            print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, time:{:.3f} min'
                  .format(epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), (time() - start) / 60))

    mean_loss = sum(mean_loss) / len(mean_loss)

    # 保存模型
    if epoch % 50 is 0 and epoch is not 0:

        # 网络模型的命名方式为：epoch轮数+当前minibatch的loss+本轮epoch的平均loss
        torch.save(net.state_dict(), './module/net{}-{:.3f}-{:.3f}.pth'.format(epoch, loss, mean_loss))

    # 对深度监督系数进行衰减
    if epoch % 40 is 0 and epoch is not 0:
        alpha *= 0.8

# 深度监督的系数变化
# 1.000
# 0.800
# 0.640
# 0.512
# 0.410
# 0.328
# 0.262
# 0.210
# 0.168
# 0.134
# 0.107
# 0.086
# 0.069
# 0.055
# 0.044
# 0.035
# 0.028
# 0.023
# 0.018
# 0.014
# 0.012
# 0.009
# 0.007
# 0.006
# 0.005
# 0.004
# 0.003
# 0.002
# 0.002
# 0.002
# 0.001
# 0.001
# 0.001
# 0.001
# 0.001
# 0.000
# 0.000
