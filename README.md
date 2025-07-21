# LYSO闪烁体探测器分辨率优化系统

基于1D ResNet的深度学习模型，用于将低分辨率LYSO能谱增强到高分辨率HPGe能谱质量。

## 项目特点

- **1D ResNet架构**：专门针对一维能谱数据设计
- **自定义复合损失函数**：重点关注峰区域，提升分辨率
- **GPU加速支持**：充分利用CUDA进行高效训练
- **分布式训练**：支持多GPU并行训练
- **大规模数据支持**：优化的数据加载器，支持百万级数据
- **完整的评估指标**：FWHM、峰质心、峰康比等
- **可视化工具**：实时监控训练过程和结果分析

## 目录

- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [基础使用](#基础使用)
- [大规模数据训练](#大规模数据训练)
- [性能优化](#性能优化)
- [故障排除](#故障排除)

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
# 单GPU训练
python train.py

# 多GPU分布式训练
python train_distributed.py --gpus 4
```

### 4. 推理测试

```bash
# 单个文件
python inference.py --input test_spectrum.h5 --output enhanced.npz --visualize

# 批量处理
python inference.py --input dataset/test --output results --batch
```

## 项目结构

```
CNN/
├── models/
│   ├── __init__.py
│   └── resnet1d.py              # 1D ResNet模型定义
├── utils/
│   ├── __init__.py
│   ├── dataset.py               # 基础数据加载器
│   ├── dataset_large.py        # 大规模数据加载器
│   ├── data_preprocessor.py    # 数据预处理工具
│   ├── losses.py                # 自定义损失函数
│   ├── metrics.py               # 性能评估指标
│   └── visualization.py         # 可视化工具
├── configs/
│   ├── default_config.yaml     # 默认配置
│   ├── distributed_config.yaml # 分布式训练配置
│   └── large_scale_config.yaml # 大规模数据配置
├── train.py                     # 单GPU训练脚本
├── train_distributed.py         # 分布式训练脚本
├── inference.py                 # 推理脚本
├── create_sample_data.py        # 创建示例数据
└── requirements.txt             # 依赖列表
```

## 配置说明

使用YAML配置文件管理所有参数。默认配置位于 `configs/default_config.yaml`。

### 基础配置示例

```yaml
# configs/default_config.yaml
model:
  name: SpectralResNet1D
  num_blocks: 12
  channels: 64
  input_channels: 1

training:
  num_epochs: 100
  batch_size: 16
  learning_rate: 0.001
  weight_decay: 0.00001
  optimizer: adam
  scheduler:
    type: CosineAnnealingWarmRestarts
    T_0: 10
    T_mult: 2

loss:
  peak_weight: 10.0
  compton_weight: 1.0
  smoothness_weight: 0.1

data:
  train_path: D:/mechine-learning/CNN/dataset/train
  val_path: D:/mechine-learning/CNN/dataset/val
  num_workers: 4
  pin_memory: true
  normalize: true

logging:
  log_dir: logs
  checkpoint_dir: checkpoints
  save_interval: 5
```

## 基础使用

### 数据格式

支持的输入格式：
- HDF5文件（.h5）：包含'lyso'和'hpge'数据集
- NumPy文件（.npy/.npz）：包含'lyso'和'hpge'数组

每个能谱应为长度4096的一维数组。

### 训练模型

```bash
# 使用默认配置
python train.py

# 使用自定义配置
python train.py --config configs/my_config.yaml

# 覆盖特定参数
python train.py --batch_size 32 --learning_rate 0.0001
```

### 性能监控

```bash
# TensorBoard监控
tensorboard --logdir logs
```

## 大规模数据训练

### 数据预处理

对于百万级数据，首先转换为高效格式：

```bash
# 转换为LMDB格式（推荐）
python utils/data_preprocessor.py \
    --source_dir /path/to/raw/data \
    --output_dir /path/to/lmdb \
    --format lmdb \
    --num_workers 32

# 创建内存映射缓存
python utils/data_preprocessor.py \
    --source_dir /path/to/raw/data \
    --output_dir /path/to/mmap \
    --format mmap

# 创建分片数据集
python utils/data_preprocessor.py \
    --source_dir /path/to/raw/data \
    --output_dir /path/to/shards \
    --format shard \
    --shard_size 10000
```

### 分布式训练

```bash
# 使用所有可用GPU
python train_distributed.py

# 指定GPU数量
python train_distributed.py --gpus 8 --config configs/distributed_config.yaml
```

### 大规模数据配置示例

```yaml
# configs/large_scale_config.yaml
model:
  name: SpectralResNet1D
  num_blocks: 12
  channels: 64

training:
  num_epochs: 100
  batch_size: 256  # 总批次大小，会自动分配到各GPU
  learning_rate: 0.001
  gradient_accumulation_steps: 4
  mixed_precision: true

data:
  format: lmdb  # 使用LMDB格式
  train_path: /path/to/lmdb/train
  val_path: /path/to/lmdb/val
  num_workers: 32
  pin_memory: true
  prefetch_factor: 2
  persistent_workers: true

distributed:
  backend: nccl
  find_unused_parameters: false
```

## 性能优化

### 数据加载优化

1. **使用LMDB格式**：速度提升3-5倍
2. **增加num_workers**：充分利用CPU
3. **启用pin_memory**：加速GPU传输
4. **使用persistent_workers**：减少进程启动开销

### 训练优化

1. **混合精度训练**：减少内存使用，加速计算
2. **梯度累积**：模拟更大批次
3. **分布式训练**：多GPU线性加速

### 性能基准

| 数据规模 | 格式 | GPU数量 | 批次大小 | 训练速度 |
|---------|------|---------|---------|----------|
| 10万 | HDF5 | 1 | 32 | ~1000 样本/秒 |
| 10万 | LMDB | 1 | 32 | ~3000 样本/秒 |
| 100万 | LMDB | 4 | 128 | ~10000 样本/秒 |
| 100万 | LMDB | 8 | 256 | ~18000 样本/秒 |

## 故障排除

### 内存不足

```yaml
# 减少内存使用的配置
training:
  batch_size: 8
  gradient_accumulation_steps: 8
  mixed_precision: true
data:
  num_workers: 2
```

### I/O瓶颈

1. 使用LMDB格式
2. 数据存储在SSD上
3. 增加prefetch_factor
4. 检查CPU使用率

### GPU利用率低

1. 增加batch_size
2. 增加num_workers
3. 使用更快的数据格式
4. 启用mixed_precision

## 高级功能

### 自定义模型

```python
from models.resnet1d import ResidualBlock1D
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 自定义架构
```

### 自定义损失函数

```python
from utils.losses import SpectralCompositeLoss

class CustomLoss(SpectralCompositeLoss):
    def forward(self, pred, target):
        # 自定义损失计算
```

## 许可证

本项目采用 MIT 许可证。

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。