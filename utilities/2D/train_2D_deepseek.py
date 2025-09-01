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
import parameter as para

os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu
cudnn.benchmark = para.cudnn_benchmark

if __name__ == '__main__':
    # 初始化网络
    net = torch.nn.DataParallel(net).cuda()
    net.train()

    # 1. 加载完整训练数据集并划分
    full_dataset = SliceDataset(os.path.join(para.training_set_2d_path, 'ct'), os.path.join(para.training_set_2d_path, 'seg'))

    # 计算划分大小 (80% 训练, 20% 验证)
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    # 随机划分数据集
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 设置随机种子以确保可重复性
    )

    print(f"数据集划分: 训练集 {train_size} 样本, 验证集 {val_size} 样本")

    # 创建数据加载器
    train_dl = DataLoader(train_dataset, para.batch_size, True, num_workers=para.num_workers, pin_memory=para.pin_memory)
    val_dl = DataLoader(val_dataset, para.batch_size, False, num_workers=para.num_workers, pin_memory=para.pin_memory)

    # 2. 初始化损失函数、优化器和学习率调度器
    loss_func_list = [DiceLoss(), ELDiceLoss(), WCELoss(), JaccardLoss(), SSLoss(), TverskyLoss(), HybridLoss(), BCELoss()]
    loss_func = loss_func_list[5]  # 使用TverskyLoss
    opt = torch.optim.Adam(net.parameters(), lr=para.learning_rate)
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, para.learning_rate_decay)
    alpha = para.alpha

    # 3. 初始化早停相关变量
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    best_model_weights = None

    # 确保输出目录存在
    os.makedirs('../module', exist_ok=True)
    os.makedirs('../../weight', exist_ok=True)

    start = time()
    print("开始训练...")

    for epoch in range(para.Epoch):
        # 训练阶段
        net.train()
        lr_decay.step()
        mean_loss = []
        train_losses = []

        # 训练循环
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
            train_losses.append(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 1000 == 0 and step > 0:
                print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, time:{:.3f} min'.format(
                    epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), (time() - start) / 60))

        # 计算平均训练损失
        current_train_loss = sum(train_losses) / len(train_losses)
        current_train_loss4 = sum(mean_loss) / len(mean_loss)

        # 验证阶段
        net.eval()
        val_losses = []
        val_losses4 = []

        with torch.no_grad():
            for val_ct, val_seg in val_dl:
                val_ct = val_ct.cuda()
                val_seg = val_seg.cuda()
                val_outputs = net(val_ct)

                # 计算总验证损失和主损失
                v_loss1 = loss_func(val_outputs[0], val_seg)
                v_loss2 = loss_func(val_outputs[1], val_seg)
                v_loss3 = loss_func(val_outputs[2], val_seg)
                v_loss4 = loss_func(val_outputs[3], val_seg)
                v_total_loss = (v_loss1 + v_loss2 + v_loss3) * alpha + v_loss4

                val_losses.append(v_total_loss.item())
                val_losses4.append(v_loss4.item())

        current_val_loss = sum(val_losses) / len(val_losses)
        current_val_loss4 = sum(val_losses4) / len(val_losses4)

        # 输出训练和验证信息
        print(f'Epoch {epoch}: Train Loss: {current_train_loss:.4f} (主损失: {current_train_loss4:.4f}), '
              f'Val Loss: {current_val_loss:.4f} (主损失: {current_val_loss4:.4f}), '
              f'LR: {opt.param_groups[0]["lr"]:.6f}')

        # 4. 早停机制逻辑
        if current_val_loss < best_val_loss:
            # 验证损失提升，保存最佳模型
            improvement = best_val_loss - current_val_loss
            best_val_loss = current_val_loss
            best_epoch = epoch
            patience_counter = 0  # 重置耐心计数器

            # 保存最佳模型权重
            best_model_weights = net.state_dict().copy()
            torch.save(best_model_weights, f'./module/best_model_epoch_{epoch}_val_loss_{current_val_loss:.4f}.pth')

            print(f'✅ 验证损失提升: {improvement:.4f}, 保存最佳模型 (epoch {epoch})')
        else:
            # 验证损失未提升
            patience_counter += 1
            degradation = current_val_loss - best_val_loss
            print(f'❌ 验证损失未提升 (+{degradation:.4f}), 耐心计数: {patience_counter}/{para.patience}')

            # 检查是否触发早停
            if patience_counter >= para.patience:
                print(f'🛑 早停触发! 连续 {para.patience} 个epoch验证损失未提升')
                print(f'最佳模型在 epoch {best_epoch}, 验证损失: {best_val_loss:.4f}')

                # 恢复最佳模型权重
                if best_model_weights is not None:
                    net.load_state_dict(best_model_weights)
                    final_save_path = f'./module/final_best_model_epoch_{best_epoch}_val_loss_{best_val_loss:.4f}.pth'
                    torch.save(net.state_dict(), final_save_path)
                    print(f'💾 已恢复并保存最终最佳模型: {final_save_path}')
                else:
                    print('⚠️  警告: 未找到最佳模型权重')

                break  # 跳出训练循环

        # 5. 按周期保存检查点（可选）
        if epoch % 50 == 0 and epoch != 0:
            checkpoint_path = f'./weight/2Dnet_epoch{epoch}_train{current_train_loss:.4f}_val{current_val_loss:.4f}.pth'
            torch.save(net.state_dict(), checkpoint_path)
            print(f'📁 保存周期检查点: {checkpoint_path}')

        # 6. 衰减深度监督系数 alpha
        if epoch % 40 == 0 and epoch != 0:
            old_alpha = alpha
            alpha *= 0.8
            print(f'🔧 深度监督系数衰减: {old_alpha:.3f} -> {alpha:.3f}')

    # 7. 训练完成后的处理
    if epoch == para.Epoch - 1:
        print(f'🎉 训练完成! 共训练 {para.Epoch} 个epoch')
        if best_model_weights is not None:
            print(f'最佳模型在 epoch {best_epoch}, 验证损失: {best_val_loss:.4f}')
            # 保存最终模型
            final_path = f'./weight/final_model_epoch{best_epoch}_val_loss{best_val_loss:.4f}.pth'
            torch.save(best_model_weights, final_path)
            print(f'💾 最终模型已保存: {final_path}')

    total_time = (time() - start) / 60
    print(f'⏰ 总训练时间: {total_time:.2f} 分钟')
    print(f'🏆 最佳结果 - Epoch: {best_epoch}, 验证损失: {best_val_loss:.4f}')