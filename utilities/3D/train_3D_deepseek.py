import os
from time import time
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split
import csv
from dataset3D import Dataset
from net.ResUNet import net
import parameter_3D as para

# 设置显卡相关
os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu
cudnn.benchmark = para.cudnn_benchmark

# 初始化日志文件
log_file = r'D:\CXJ_code\Liver\LiTS_Tumor\weight\3D\train_log.csv'
with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate', 'time_min'])

if __name__ == '__main__':
    net = torch.nn.DataParallel(net).cuda()
    net.train()

    # 加载完整数据集
    full_ds = Dataset(os.path.join(para.training_set_path, 'ct'), os.path.join(para.training_set_path, 'seg'))

    # 划分训练集和验证集 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_dl = DataLoader(train_ds, para.batch_size, True, num_workers=para.num_workers, pin_memory=para.pin_memory)
    val_dl = DataLoader(val_ds, para.batch_size, False, num_workers=para.num_workers, pin_memory=para.pin_memory)

    # 使用交叉熵损失进行多类别分割
    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=para.learning_rate)
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, para.learning_rate_decay)
    alpha = para.alpha

    # 初始化最佳验证损失
    best_val_loss = float('inf')

    # 训练网络
    start = time()
    for epoch in range(para.Epoch):
        # 训练阶段
        net.train()
        mean_loss = []

        for step, (ct, seg) in enumerate(train_dl):

            ct = ct.cuda()
            seg = seg.cuda().long()
            outputs = net(ct)

            # 检查输出结构，适应多尺度输出
            if isinstance(outputs, (list, tuple)):
                loss1 = loss_func(outputs[0], seg)
                loss2 = loss_func(outputs[1], seg)
                loss3 = loss_func(outputs[2], seg)
                loss4 = loss_func(outputs[3], seg)
                loss = (loss1 + loss2 + loss3) * alpha + loss4
                mean_loss.append(loss4.item())
            else:
                # 如果只有一个输出
                loss4 = loss_func(outputs, seg)
                loss = loss4
                mean_loss.append(loss4.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 20 == 0:
                if isinstance(outputs, (list, tuple)):
                    print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, time:{:.3f} min'
                          .format(epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(),(time() - start) / 60))
                else:
                    print('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'
                          .format(epoch, step, loss4.item(), (time() - start) / 60))

        # 更新学习率
        lr_decay.step()
        current_lr = opt.param_groups[0]['lr']

        # 计算训练平均损失
        train_mean_loss = sum(mean_loss) / len(mean_loss)

        # 验证阶段
        net.eval()
        val_losses = []
        with torch.no_grad():
            for ct, seg in val_dl:
                ct = ct.cuda()
                seg = seg.cuda().long()
                outputs = net(ct)

                # 检查输出结构，适应多尺度输出
                if isinstance(outputs, (list, tuple)):
                    # 只计算主尺度损失（与训练中的loss4对应）
                    loss4 = loss_func(outputs[3], seg)
                else:
                    # 如果只有一个输出
                    loss4 = loss_func(outputs, seg)

                val_losses.append(loss4.item())

        # 计算验证平均损失
        val_mean_loss = sum(val_losses) / len(val_losses)

        # 打印训练和验证损失
        print('Epoch: {}, Train Loss: {:.3f}, Val Loss: {:.3f}, LR: {:.6f}, Time: {:.3f} min'
              .format(epoch, train_mean_loss, val_mean_loss, current_lr, (time() - start) / 60))

        # 记录日志到CSV文件
        try:
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_mean_loss, val_mean_loss, current_lr, (time() - start) / 60])
        except PermissionError:
            print(f"Warning: Cannot write to log file {log_file}. Permission denied.")

        # 仅在验证损失改善时保存模型
        if val_mean_loss < best_val_loss:
            best_val_loss = val_mean_loss
            model_path = r"D:\CXJ_code\Liver\LiTS_Tumor\weight\3D\best_model.pth"
            torch.save(net.state_dict(), model_path)
            print(f"Validation loss improved. Model saved as {model_path} with loss {val_mean_loss:.3f}")

        # 对深度监督系数进行衰减
        if epoch % 40 == 0 and epoch != 0:
            alpha *= 0.8
            print(f"Alpha decayed to: {alpha}")

        print("*" * 50)
