#!/usr/bin/env python3
"""
统一任务运行脚本 / Unified task runner script
读取project_config.yaml并执行相应任务 / Read project_config.yaml and execute corresponding tasks
"""

import yaml
import argparse
import subprocess
import sys
import os
from pathlib import Path

def load_config():
    """加载项目配置 / Load project configuration"""
    config_path = Path(__file__).parent / "project_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_command(cmd):
    """运行命令并显示输出 / Run command and display output"""
    print(f"\n执行命令 / Executing command: {' '.join(cmd)}")
    print("-" * 50)
    result = subprocess.run(cmd, shell=False)
    return result.returncode

def prepare_data(config):
    """准备数据集 - 自动分割 / Prepare dataset - automatic split"""
    data_config = config.get('data_config', {})
    source_dir = data_config.get('source_dir', 'data')
    output_dir = data_config.get('output_dir', 'dataset_split')
    
    cmd = [
        sys.executable, "prepare_data.py",
        "--source", source_dir,
        "--output", output_dir,
        "--train_ratio", str(data_config.get('train_ratio', 0.7)),
        "--val_ratio", str(data_config.get('val_ratio', 0.15)),
        "--test_ratio", str(data_config.get('test_ratio', 0.15))
    ]
    return run_command(cmd)

def train_quick(config):
    """快速测试训练 / Quick test training"""
    if not config['training']['quick_test']['enabled']:
        print("快速测试已禁用 / Quick test is disabled")
        return 0
    
    cmd = [
        sys.executable, "train_improved.py",
        "--config", config['training']['config_file'],
        "--num_epochs", str(config['training']['quick_test']['num_epochs'])
    ]
    return run_command(cmd)

def train_full(config):
    """完整训练 / Full training"""
    cmd = [
        sys.executable, "train_improved.py",
        "--config", config['training']['config_file']
    ]
    
    # 如果启用恢复训练 / If resume training is enabled
    if config['training']['resume']['enabled']:
        cmd.extend(["--resume", config['training']['resume']['checkpoint_path']])
    
    return run_command(cmd)

def run_inference(config):
    """运行推理 / Run inference"""
    checkpoint_path = os.path.join(config['inference']['checkpoint_dir'], 
                                   config['inference']['checkpoint_name'])
    output_path = os.path.join(config['inference']['output_dir'], "result.npz")
    
    # 创建输出目录 / Create output directory
    os.makedirs(config['inference']['output_dir'], exist_ok=True)
    
    cmd = [
        sys.executable, "inference.py",
        "--checkpoint", checkpoint_path,
        "--input", config['inference']['test_input'],
        "--output", output_path
    ]
    if config['inference']['visualize']:
        cmd.append("--visualize")
    return run_command(cmd)

def preprocess_data(config):
    """数据预处理 / Data preprocessing"""
    cmd = [
        sys.executable, "utils/data_preprocessor.py",
        "--source_dir", config['preprocessing']['source_dir'],
        "--output_dir", config['preprocessing']['output_dir'],
        "--format", "lmdb",
        "--num_workers", str(config['preprocessing']['num_workers'])
    ]
    return run_command(cmd)

def start_tensorboard(config):
    """启动TensorBoard / Start TensorBoard"""
    logdir = config['monitoring']['tensorboard_logdir']
    port = config['monitoring']['tensorboard_port']
    cmd = ["tensorboard", "--logdir", logdir, "--port", str(port)]
    print(f"\nTensorBoard启动在 / TensorBoard started at: http://localhost:{port}")
    print("按 Ctrl+C 停止 / Press Ctrl+C to stop")
    subprocess.run(cmd)

def monitor_gpu():
    """监控GPU / Monitor GPU"""
    cmd = ["nvidia-smi", "-l", "1"]
    print("\nGPU监控 / GPU monitoring (按 Ctrl+C 停止 / Press Ctrl+C to stop)")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description='CNN项目任务运行器 / CNN Project Task Runner')
    parser.add_argument('task', 
                       choices=['prepare', 'train-quick', 'train-full', 'inference', 
                               'preprocess', 'tensorboard', 'gpu'],
                       help='要执行的任务 / Task to execute')
    parser.add_argument('--config', default='project_config.yaml',
                       help='配置文件路径 / Configuration file path')
    
    args = parser.parse_args()
    
    # 加载配置 / Load configuration
    config = load_config()
    
    # 执行任务 / Execute task
    if args.task == 'prepare':
        prepare_data(config)
    elif args.task == 'train-quick':
        train_quick(config)
    elif args.task == 'train-full':
        train_full(config)
    elif args.task == 'inference':
        run_inference(config)
    elif args.task == 'preprocess':
        preprocess_data(config)
    elif args.task == 'tensorboard':
        start_tensorboard(config)
    elif args.task == 'gpu':
        monitor_gpu()

if __name__ == '__main__':
    main()