# LYSO闪烁体探测器分辨率优化系统

基于1D ResNet的深度学习模型，用于将低分辨率LYSO能谱增强到高分辨率HPGe能谱质量。

## 项目特点

- **1D ResNet架构**：专门针对一维能谱数据设计
- **混合精度训练**：支持FP16训练，节省显存
- **梯度累积**：支持在有限显存下模拟大批次训练
- **自定义复合损失函数**：重点关注峰区域，提升分辨率
- **GPU加速指标计算**：高效的训练和验证
- **完整的评估指标**：FWHM、峰质心、峰康比等
- **灵活的配置系统**：YAML配置文件支持

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 创建示例数据集

```bash
python create_sample_data.py --num_samples 1000
```

### 3. 训练模型

```bash
# 使用默认配置
python train_improved.py --config configs/default_config.yaml

# 使用RTX 4060优化配置（推荐）
python train_improved.py --config configs/rtx4060_config.yaml

# 命令行覆盖配置参数
python train_improved.py --config configs/rtx4060_config.yaml --batch_size 32 --learning_rate 0.0001
```

### 4. 推理测试

```bash
# 单个文件
python inference.py --checkpoint checkpoints/best_model.pth --input test_spectrum.h5 --output enhanced.npz --visualize

# 批量处理
python inference.py --checkpoint checkpoints/best_model.pth --input dataset/test --output results --batch
```

## 项目结构

```
CNN/
├── train_improved.py            # 主训练脚本（支持混合精度和梯度累积）
├── inference.py                 # 推理脚本
├── create_sample_data.py        # 创建示例数据
├── models/
│   ├── resnet1d.py             # 基础1D ResNet模型
│   └── resnet1d_improved.py    # 改进版模型（可选）
├── utils/                       # 工具模块
│   ├── dataset.py              # 数据加载器
│   ├── losses.py               # 损失函数
│   ├── metrics.py              # 评估指标
│   ├── config_manager.py       # 配置管理
│   ├── checkpoint_manager.py   # 检查点管理
│   └── data_preprocessor.py    # 数据预处理
├── configs/
│   ├── default_config.yaml     # 默认配置
│   └── rtx4060_config.yaml     # RTX 4060优化配置
└── README_RTX4060.md           # RTX 4060专用指南
```

## 配置说明

配置文件使用YAML格式，主要参数包括：

```yaml
model:
  name: SpectralResNet1D        # 模型名称
  num_blocks: 12                # 残差块数量
  channels: 64                  # 通道数

training:
  num_epochs: 100               # 训练轮数
  batch_size: 64                # 批次大小
  learning_rate: 0.001          # 学习率
  mixed_precision: true         # 混合精度训练
  gradient_accumulation_steps: 4 # 梯度累积步数

loss:
  type: optimized               # 损失函数类型
  peak_weight: 10.0             # 峰权重
  compton_weight: 1.0           # 康普顿权重
  smoothness_weight: 0.1        # 平滑度权重

data:
  format: lmdb                  # 数据格式（推荐lmdb）
  train_path: dataset/train     # 训练数据路径
  val_path: dataset/val         # 验证数据路径
```

## 数据格式

支持的输入格式：
- HDF5文件（.h5）：包含'lyso'和'hpge'数据集
- NumPy文件（.npy/.npz）：包含'lyso'和'hpge'数组
- LMDB格式：用于大规模数据集（推荐）

每个能谱应为长度4096的一维数组。

## 大规模数据训练

对于百万级数据，建议先转换为LMDB格式：

```bash
# 转换为LMDB格式
python utils/data_preprocessor.py \
    --source_dir /path/to/raw/data \
    --output_dir /path/to/lmdb \
    --format lmdb \
    --num_workers 8
```

## 性能监控

```bash
# TensorBoard监控
tensorboard --logdir logs

# GPU使用监控
nvidia-smi -l 1
```

## 恢复训练

```bash
# 从检查点恢复
python train_improved.py --config configs/rtx4060_config.yaml --resume checkpoints/checkpoint_epoch_50.pth
```

## 故障排除

### 显存不足
- 减小batch_size
- 减少model.channels
- 增加gradient_accumulation_steps
- 确保mixed_precision: true

### 数据加载慢
- 使用LMDB格式
- 数据存储在SSD上
- 调整num_workers

详细的RTX 4060优化指南请参考 [README_RTX4060.md](README_RTX4060.md)

## 许可证

本项目采用 MIT 许可证。