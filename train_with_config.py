#!/usr/bin/env python3
"""
使用统一配置系统的训练脚本
支持命令行参数覆盖配置文件设置
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

from models.resnet1d import SpectralResNet1D
from utils.dataset import create_data_loaders
from utils.dataset_large import create_large_data_loaders
from utils.losses import SpectralCompositeLoss
from utils.metrics import SpectralMetrics
from utils.config_manager import ConfigManager


class ConfigurableTrainer:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.device = self._setup_device()
        
        # 创建模型
        self.model = self._create_model()
        
        # 创建损失函数
        self.criterion = self._create_criterion()
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 创建学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 创建数据加载器
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # 设置日志
        self.writer = self._setup_logging()
        
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # 创建保存目录
        self._create_directories()
    
    def _setup_device(self):
        """设置计算设备"""
        if self.config.get('device.use_cuda', True) and torch.cuda.is_available():
            device_id = self.config.get('device.device_id', 0)
            if device_id >= 0:
                device = torch.device(f'cuda:{device_id}')
            else:
                device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        print(f"Using device: {device}")
        return device
    
    def _create_model(self):
        """创建模型"""
        model_config = self.config.get_model_config()
        
        model = SpectralResNet1D(
            input_channels=model_config.get('input_channels', 1),
            num_blocks=model_config.get('num_blocks', 12),
            channels=model_config.get('channels', 64)
        )
        
        model = model.to(self.device)
        
        # 模型编译（PyTorch 2.0+）
        if self.config.get('optimization.compile_model', False):
            try:
                model = torch.compile(model)
                print("Model compiled successfully")
            except Exception as e:
                print(f"Model compilation failed: {e}")
        
        return model
    
    def _create_criterion(self):
        """创建损失函数"""
        loss_config = self.config.get_loss_config()
        
        return SpectralCompositeLoss(
            peak_weight=loss_config.get('peak_weight', 10.0),
            compton_weight=loss_config.get('compton_weight', 1.0),
            smoothness_weight=loss_config.get('smoothness_weight', 0.1)
        )
    
    def _create_optimizer(self):
        """创建优化器"""
        training_config = self.config.get_training_config()
        
        optimizer_type = training_config.get('optimizer', 'adam').lower()
        lr = training_config.get('learning_rate', 1e-3)
        weight_decay = training_config.get('weight_decay', 1e-5)
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        return optimizer
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        training_config = self.config.get_training_config()
        scheduler_config = training_config.get('scheduler', {})
        
        scheduler_type = scheduler_config.get('type', 'CosineAnnealingWarmRestarts')
        
        if scheduler_type == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=scheduler_config.get('T_0', 10),
                T_mult=scheduler_config.get('T_mult', 2)
            )
        elif scheduler_type == 'StepLR':
            scheduler = StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10)
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _create_data_loaders(self):
        """创建数据加载器"""
        data_config = self.config.get_data_config()
        data_format = data_config.get('format', 'h5')
        
        if data_format in ['lmdb', 'mmap'] or self.config.get('large_scale.enable_sharding', False):
            # 使用大规模数据加载器
            return create_large_data_loaders(
                train_path=data_config['train_path'],
                val_path=data_config['val_path'],
                batch_size=self.config.get('training.batch_size', 16),
                num_workers=data_config.get('num_workers', 4),
                use_lmdb=(data_format == 'lmdb')
            )
        else:
            # 使用标准数据加载器
            return create_data_loaders(
                train_path=data_config['train_path'],
                val_path=data_config['val_path'],
                batch_size=self.config.get('training.batch_size', 16),
                num_workers=data_config.get('num_workers', 4)
            )
    
    def _setup_logging(self):
        """设置日志"""
        logging_config = self.config.get_logging_config()
        
        if logging_config.get('tensorboard', True):
            log_dir = logging_config.get('log_dir', 'logs')
            return SummaryWriter(log_dir)
        else:
            return None
    
    def _create_directories(self):
        """创建必要的目录"""
        logging_config = self.config.get_logging_config()
        
        Path(logging_config.get('log_dir', 'logs')).mkdir(parents=True, exist_ok=True)
        Path(logging_config.get('checkpoint_dir', 'checkpoints')).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        loss_components = {'peak_loss': 0, 'compton_loss': 0, 'smoothness_loss': 0}
        
        # 混合精度训练
        use_amp = self.config.get('training.mixed_precision', False)
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        # 梯度累积
        accumulation_steps = self.config.get('training.gradient_accumulation_steps', 1)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (lyso, hpge) in enumerate(pbar):
            lyso = lyso.to(self.device, non_blocking=True)
            hpge = hpge.to(self.device, non_blocking=True)
            
            # 前向传播
            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = self.model(lyso)
                    loss, components = self.criterion(pred, hpge)
                    loss = loss / accumulation_steps
            else:
                pred = self.model(lyso)
                loss, components = self.criterion(pred, hpge)
                loss = loss / accumulation_steps
            
            # 反向传播
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度更新
            if (batch_idx + 1) % accumulation_steps == 0:
                # 梯度裁剪
                grad_clip_norm = self.config.get('training.gradient_clip_norm')
                if grad_clip_norm:
                    if use_amp:
                        scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                
                if use_amp:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            for key in loss_components:
                loss_components[key] += components[key]
            
            # 日志记录
            log_interval = self.config.get('logging.log_interval', 10)
            if batch_idx % log_interval == 0:
                pbar.set_postfix({'loss': loss.item() * accumulation_steps})
                
                if self.writer:
                    self.writer.add_scalar('Loss/train', loss.item() * accumulation_steps, self.global_step)
                    for key, value in components.items():
                        self.writer.add_scalar(f'Loss/{key}', value, self.global_step)
            
            self.global_step += 1
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, loss_components
    
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_metrics = []
        
        with torch.no_grad():
            for lyso, hpge in tqdm(self.val_loader, desc='Validation'):
                lyso = lyso.to(self.device, non_blocking=True)
                hpge = hpge.to(self.device, non_blocking=True)
                
                pred = self.model(lyso)
                loss, _ = self.criterion(pred, hpge)
                total_loss += loss.item()
                
                # 计算性能指标（仅部分样本以加速）
                if len(all_metrics) < 20:
                    metrics = SpectralMetrics.compute_all_metrics(pred[0], hpge[0], lyso[0])
                    all_metrics.append(metrics)
        
        avg_loss = total_loss / len(self.val_loader)
        
        # 计算平均指标
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m]
                if values:
                    avg_metrics[key] = np.mean(values)
        
        # 记录到TensorBoard
        if self.writer:
            self.writer.add_scalar('Loss/val', avg_loss, epoch)
            for key, value in avg_metrics.items():
                self.writer.add_scalar(f'Metrics/{key}', value, epoch)
        
        return avg_loss, avg_metrics
    
    def save_checkpoint(self, epoch, val_loss):
        """保存检查点"""
        logging_config = self.config.get_logging_config()
        checkpoint_dir = logging_config.get('checkpoint_dir', 'checkpoints')
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存当前检查点
        torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pth")
        
        # 保存最佳模型
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, f"{checkpoint_dir}/best_model.pth")
    
    def train(self):
        """主训练循环"""
        num_epochs = self.config.get('training.num_epochs', 100)
        save_interval = self.config.get('logging.save_interval', 5)
        val_interval = self.config.get('logging.val_interval', 1)
        
        for epoch in range(num_epochs):
            # 训练
            train_loss, train_components = self.train_epoch(epoch)
            
            # 验证
            if epoch % val_interval == 0:
                val_loss, val_metrics = self.validate(epoch)
            else:
                val_loss, val_metrics = None, {}
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    if val_loss is not None:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 打印信息
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"           Val Loss: {val_loss:.4f}")
            if val_metrics:
                print(f"           Val Metrics: {val_metrics}")
            
            # 保存检查点
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch, val_loss or train_loss)
        
        if self.writer:
            self.writer.close()


def main():
    # 创建配置管理器
    config_manager = ConfigManager()
    
    # 解析命令行参数并更新配置
    args = config_manager.parse_args_and_update()
    
    # 验证配置
    config_manager.validate_config()
    
    # 打印配置（调试模式）
    if args.debug:
        config_manager.print_config()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建训练器并开始训练
    trainer = ConfigurableTrainer(config_manager)
    trainer.train()


if __name__ == '__main__':
    main()