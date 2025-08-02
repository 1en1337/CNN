# 配置化运行指南

## 使用方法

### 1. 修改配置文件
编辑 `project_config.yaml` 来设置所有参数：

```yaml
# 例如，修改生成样本数量
data_generation:
  num_samples: 2000  # 改为2000个样本

# 修改训练参数
full_training:
  num_epochs: 50     # 改为50个epoch
  batch_size: 32     # 改为32的批次
```

### 2. 运行任务

在命令行中使用 `run_task.py`：

```bash
# 生成数据（读取配置中的num_samples）
python run_task.py data

# 快速测试训练（读取配置中的quick_test设置）
python run_task.py train-quick

# 完整训练（读取配置中的full_training设置）
python run_task.py train-full

# 运行推理
python run_task.py inference

# 数据预处理
python run_task.py preprocess

# 启动TensorBoard
python run_task.py tensorboard

# 监控GPU
python run_task.py gpu

# 运行完整测试流程（数据生成→快速训练→推理）
python run_task.py all
```

## 配置说明

### project_config.yaml 结构

1. **data_generation** - 数据生成
   - `num_samples`: 生成样本总数
   - `train_ratio`: 训练集比例
   - `val_ratio`: 验证集比例
   - `test_ratio`: 测试集比例

2. **quick_test** - 快速测试
   - `enabled`: 是否启用
   - `num_epochs`: epoch数量
   - `batch_size`: 批次大小

3. **full_training** - 完整训练
   - `config_file`: 使用的配置文件
   - `num_epochs`: epoch数量
   - `batch_size`: 批次大小
   - `learning_rate`: 学习率

4. **inference** - 推理设置
   - `checkpoint_path`: 模型文件路径
   - `test_file`: 测试文件路径
   - `output_path`: 输出路径
   - `visualize`: 是否可视化

5. **preprocessing** - 数据预处理
   - `source_dir`: 源数据目录
   - `output_dir`: 输出目录
   - `num_workers`: 处理进程数

6. **system** - 系统设置
   - `gpu_id`: GPU编号
   - `mixed_precision`: 是否使用混合精度
   - `tensorboard_port`: TensorBoard端口

## 工作流程示例

### 场景1：小数据集快速测试
```yaml
# 修改 project_config.yaml
data_generation:
  num_samples: 100  # 小数据集
quick_test:
  enabled: true
  num_epochs: 5     # 少量epoch
```
然后运行：`python run_task.py all`

### 场景2：大数据集完整训练
```yaml
# 修改 project_config.yaml
data_generation:
  num_samples: 10000  # 大数据集
full_training:
  num_epochs: 200     # 充分训练
  batch_size: 64
```
然后运行：
```bash
python run_task.py data        # 生成数据
python run_task.py train-full  # 完整训练
```

### 场景3：使用真实数据
```yaml
# 修改 project_config.yaml
preprocessing:
  source_dir: "path/to/your/real/data"
  output_dir: "dataset/lmdb"
```
然后运行：
```bash
python run_task.py preprocess  # 预处理真实数据
python run_task.py train-full  # 训练
```

## 优势

1. **集中管理**：所有参数在一个文件中
2. **易于修改**：改配置文件即可，无需改代码
3. **批量实验**：可以保存多个配置文件进行对比
4. **版本控制**：配置文件可以纳入Git管理