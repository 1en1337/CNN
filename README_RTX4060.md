# RTX 4060 单GPU训练指南

本指南专门针对使用 RTX 4060 (8GB显存) 训练百万级光谱数据集。

## 核心文件

- `train_improved.py` - 主训练脚本（支持混合精度和梯度累积）
- `configs/rtx4060_config.yaml` - 优化配置文件
- `inference.py` - 推理脚本
- `utils/data_preprocessor.py` - 数据预处理工具

## 快速开始

### 1. 数据预处理（必须！）

将原始数据转换为LMDB格式以提高训练效率：

```bash
python utils/data_preprocessor.py \
    --source_dir /你的原始数据路径 \
    --output_dir dataset/lmdb \
    --format lmdb \
    --num_workers 8
```

### 2. 开始训练

```bash
# 使用优化配置
python train_improved.py --config configs/rtx4060_config.yaml

# 如果显存不足，减小batch_size
python train_improved.py --config configs/rtx4060_config.yaml --batch_size 32

# 恢复中断的训练
python train_improved.py --config configs/rtx4060_config.yaml --resume checkpoints_4060/checkpoint_epoch_50.pth
```

### 3. 监控训练

```bash
# 查看GPU使用情况
nvidia-smi -l 1

# TensorBoard可视化
tensorboard --logdir logs_4060
```

### 4. 推理测试

```bash
# 单个文件
python inference.py \
    --checkpoint checkpoints_4060/best_model.pth \
    --input test.h5 \
    --output result.npz \
    --visualize

# 批量处理
python inference.py \
    --checkpoint checkpoints_4060/best_model.pth \
    --input dataset/test \
    --output results \
    --batch
```

## 性能优化建议

### 显存不足时的调整

1. **减少模型大小**
   - 修改 `channels: 64` → `channels: 48` 或 `32`
   - 修改 `num_blocks: 12` → `num_blocks: 8`

2. **减少批次大小**
   - 修改 `batch_size: 64` → `batch_size: 32` 或 `16`
   - 增加 `gradient_accumulation_steps: 8` 保持有效批次大小

3. **数据加载优化**
   - 如果CPU瓶颈：减少 `num_workers: 4` → `num_workers: 2`
   - 如果内存不足：减少 `cache_size: 200` → `cache_size: 100`

### 预期性能

- **训练速度**: 2000-3000 样本/秒
- **显存使用**: 6-7GB (混合精度)
- **总训练时间**: 约20-30小时（百万数据集）

## 常见问题

### 1. CUDA Out of Memory
```yaml
# 修改配置
batch_size: 32  # 或16
channels: 48    # 或32
gradient_accumulation_steps: 8
```

### 2. 数据加载慢
- 确保使用LMDB格式
- 数据存储在SSD上
- 调整num_workers

### 3. 训练不稳定
- 减小learning_rate
- 增加gradient_clip
- 使用更多的gradient_accumulation_steps

## 推荐训练流程

1. **第一阶段**：使用较小的模型快速验证
   ```bash
   python train_improved.py --config configs/rtx4060_config.yaml --channels 32 --num_epochs 20
   ```

2. **第二阶段**：使用完整模型精调
   ```bash
   python train_improved.py --config configs/rtx4060_config.yaml --resume checkpoints_4060/checkpoint_epoch_20.pth
   ```

3. **评估最佳模型**
   ```bash
   python inference.py --checkpoint checkpoints_4060/best_model.pth --input dataset/test --batch
   ```