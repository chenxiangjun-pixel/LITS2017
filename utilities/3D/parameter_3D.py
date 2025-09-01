# -----------------------路径相关参数---------------------------------------
train_ct_path = r'D:\CXJ_code\Data_collection\LiTS 2017\Train_data\CT'     # 原始CT数据
train_seg_path = r'D:\CXJ_code\Data_collection\LiTS 2017\Train_data\Mask'  # 原始mask

training_set_path = r'D:\CXJ_code\Liver\LiTS_Tumor\data_prepare\3D_data\train_datasets'  # 用原始数据生成的训练集

test_ct_path = r'D:\CXJ_code\Liver\LiTS_Tumor\data_prepare\3D_data\test_datasets\CT'    # 模型评估CT
test_seg_path = r'D:\CXJ_code\Liver\LiTS_Tumor\data_prepare\3D_data\test_datasets\seg'  # 模型评估mask
SUYU_path = r'D:\CXJ_code\Liver\LiTS_Tumor\data_prepare\3D_data\test_datasets\testCT'   # 测试数据

val_path = r'D:\CXJ_code\Liver\LiTS_Tumor\result\3D_result\test'   # 网络评估结果保存路径
test_path = r'D:\CXJ_code\Liver\LiTS_Tumor\result\3D_result\suyu'  # 网络测试结果保存路径
crf_path = r'./crf'               # CRF优化结果保存路径

module_path = r'D:\CXJ_code\Liver\LiTS_Tumor\weight\3D\best_model.pth'  # 模型权重

# ---------------------训练数据获取相关参数-----------------------------------
size = 48                 # 使用48张连续切片作为网络的输入
down_scale = 0.5          # 横断面降采样因子
expand_slice = 20         # 仅使用包含肝脏以及肝脏上下20张切片作为训练样本
slice_thickness = 1       # 将所有数据在z轴的spacing归一化到1mm
upper, lower = 200, -200  # CT数据灰度截断窗口
drop_rate = 0.3           # dropout随机丢弃概率
gpu = '0'                 # 使用的显卡序号
Epoch = 100
learning_rate = 1e-4
learning_rate_decay = [500, 750]
alpha = 0.33              # 深度监督衰减系数
batch_size = 1
num_workers = 3
patience = 20             # 早停法的参数
pin_memory = True
cudnn_benchmark = True
threshold = 0.5           # 阈值度阈值
stride = 12               # 滑动取样步长
maximum_hole = 5e4        # 最大的空洞面积
z_expand, x_expand, y_expand = 10, 30, 30  # 根据预测结果在三个方向上的扩展数量
max_iter = 20             # CRF迭代次数
s1, s2, s3 = 1, 10, 10    # CRF高斯核参数
