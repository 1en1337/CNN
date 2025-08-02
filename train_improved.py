import torch
import torch.nn as nn
import torch.optim as optim
from utils.scheduler_utils import create_scheduler, step_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import time

from models.resnet1d import SpectralResNet1D
from models.resnet1d_improved import ImprovedSpectralResNet1D
from utils.dataset import create_data_loaders, create_large_data_loaders
from utils.losses import SpectralCompositeLoss, OptimizedSpectralLoss, AdaptivePeakWeightedLoss
from utils.metrics import SpectralMetrics
from utils.efficient_metrics import EfficientSpectralMetrics, FastTrainingMetrics
from utils.config_manager import ConfigManager
from utils.loss_adapter import LossComponentAdapter
from utils.checkpoint_manager import CheckpointManager


class ImprovedTrainer:
    """改进的训练器，确保使用验证损失保存最佳模型
    Improved trainer that ensures saving the best model based on validation loss"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 模型配置 / Model configuration
        model_config = self.config_manager.get_model_config()
        model_name = model_config.get('name', 'SpectralResNet1D')
        
        if model_name == 'ImprovedSpectralResNet1D':
            self.model = ImprovedSpectralResNet1D(
                input_channels=model_config.get('input_channels', 1),
                num_blocks=model_config.get('num_blocks', 12),
                base_channels=model_config.get('channels', 64)
            ).to(self.device)
        else:
            self.model = SpectralResNet1D(
                input_channels=model_config.get('input_channels', 1),
                num_blocks=model_config.get('num_blocks', 12),
                channels=model_config.get('channels', 64)
            ).to(self.device)
        
        # 损失函数配置 / Loss function configuration
        loss_config = self.config_manager.get_loss_config()
        loss_type = loss_config.get('type', 'original')
        
        if loss_type == 'optimized':
            self.criterion = OptimizedSpectralLoss(
                peak_weight=loss_config.get('peak_weight', 10.0),
                base_weight=loss_config.get('compton_weight', 1.0),
                smoothness_weight=loss_config.get('smoothness_weight', 0.1),
                frequency_weight=loss_config.get('frequency_weight', 0.1)
            )
        elif loss_type == 'adaptive':
            self.criterion = AdaptivePeakWeightedLoss(
                base_weight=loss_config.get('compton_weight', 1.0),
                max_peak_weight=loss_config.get('peak_weight', 10.0),
                smoothness_weight=loss_config.get('smoothness_weight', 0.1)
            )
        else:
            self.criterion = SpectralCompositeLoss(
                peak_weight=loss_config.get('peak_weight', 10.0),
                compton_weight=loss_config.get('compton_weight', 1.0),
                smoothness_weight=loss_config.get('smoothness_weight', 0.1)
            )
        
        # 训练配置 / Training configuration
        training_config = self.config_manager.get_training_config()
        
        # 优化器 / Optimizer
        if training_config.get('optimizer', 'adam') == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=training_config.get('learning_rate', 1e-3),
                weight_decay=training_config.get('weight_decay', 1e-5)
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=training_config.get('learning_rate', 1e-3),
                momentum=0.9,
                weight_decay=training_config.get('weight_decay', 1e-5)
            )
        
        # 学习率调度器 / Learning rate scheduler
        scheduler_config = self.config_manager.get('scheduler', {})
        self.scheduler = create_scheduler(self.optimizer, scheduler_config)
        
        # 混合精度训练 / Mixed precision training
        self.use_amp = self.config_manager.get('device.mixed_precision', False)
        if self.use_amp:
            self.scaler = GradScaler()
            print("Using mixed precision training")
        
        # 数据加载器 / Data loaders
        self.setup_data_loaders()
        
        # 日志配置 / Logging configuration
        logging_config = self.config_manager.get_logging_config()
        self.writer = SummaryWriter(logging_config.get('log_dir', 'logs'))
        self.checkpoint_dir = Path(logging_config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练状态 / Training state
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.global_step = 0
        self.save_interval = logging_config.get('save_interval', 5)
        self.log_interval = logging_config.get('log_interval', 10)
        
        # 指标系统 / Metrics system
        metrics_config = self.config_manager.get('metrics', {})
        self.use_efficient_metrics = metrics_config.get('use_efficient_metrics', True)
        if self.use_efficient_metrics:
            self.val_metrics = EfficientSpectralMetrics().to(self.device)
            self.train_metrics = FastTrainingMetrics().to(self.device)
            print("Using efficient GPU metrics system")
        
        # 检查点管理器 / Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=logging_config.get('checkpoint_dir', 'checkpoints')
        )
    
    def setup_data_loaders(self):
        """设置数据加载器 / Setup data loaders"""
        data_config = self.config_manager.get_data_config()
        training_config = self.config_manager.get_training_config()
        
        dataset_kwargs = {
            'cache_size': data_config.get('cache_size', 100),
            'use_mmap': data_config.get('use_mmap', True),
            'normalize': data_config.get('normalize', True)
        }
        
        # 选择合适的数据加载器 / Choose appropriate data loader
        if data_config.get('use_streaming', False):
            self.train_loader, self.val_loader = create_data_loaders(
                data_config['train_path'],
                data_config['val_path'],
                training_config['batch_size'],
                data_config.get('num_workers', 4),
                use_streaming=True,
                **dataset_kwargs
            )
        elif data_config.get('format') == 'lmdb':
            self.train_loader, self.val_loader = create_large_data_loaders(
                data_config['train_path'],
                data_config['val_path'],
                training_config['batch_size'],
                data_config.get('num_workers', 4),
                **dataset_kwargs
            )
        else:
            # 使用修复版的数据加载器，避免NumPy DLL问题 / Use fixed data loader to avoid NumPy DLL issues
            self.train_loader, self.val_loader = create_data_loaders(
                data_config['train_path'],
                data_config['val_path'],
                training_config['batch_size'],
                data_config.get('num_workers', 0),  # Windows下使用0 / Use 0 for Windows
                **dataset_kwargs
            )
    
    def train_epoch(self, epoch):
        """训练一个epoch / Train one epoch"""
        self.model.train()
        total_loss = 0
        # 使用标准键初始化 / Initialize with standard keys
        loss_components = {key: 0 for key in LossComponentAdapter.STANDARD_KEYS}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (lyso, hpge) in enumerate(pbar):
            lyso = lyso.to(self.device)
            hpge = hpge.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 混合精度训练 / Mixed precision training
            if self.use_amp:
                with autocast():
                    pred = self.model(lyso)
                    loss, components = self.criterion(pred, hpge)
                
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪 / Gradient clipping
                if self.config_manager.get('training.gradient_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config_manager.get('training.gradient_clip')
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(lyso)
                loss, components = self.criterion(pred, hpge)
                loss.backward()
                
                # 梯度裁剪 / Gradient clipping
                if self.config_manager.get('training.gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config_manager.get('training.gradient_clip')
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            # 适配并累加损失组件 / Adapt and accumulate loss components
            adapted_components = LossComponentAdapter.adapt_components(components)
            for key in loss_components:
                if key in adapted_components:
                    loss_components[key] += adapted_components[key]
            
            # 更新训练指标 / Update training metrics
            if self.use_efficient_metrics:
                self.train_metrics.update(pred, hpge)
            
            # 日志记录 / Logging
            if batch_idx % self.log_interval == 0:
                pbar_info = {'loss': loss.item()}
                
                if self.use_efficient_metrics:
                    train_metrics_info = self.train_metrics.compute()
                    pbar_info.update({
                        'corr': f"{train_metrics_info['running_correlation']:.3f}",
                        'pk_ratio': f"{train_metrics_info['running_peak_ratio']:.3f}"
                    })
                
                pbar.set_postfix(pbar_info)
                
                # TensorBoard记录 / TensorBoard logging
                self.writer.add_scalar('Loss/train', loss.item(), self.global_step)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                for key, value in components.items():
                    self.writer.add_scalar(f'Loss/{key}', value, self.global_step)
                
                if self.use_efficient_metrics:
                    for key, value in train_metrics_info.items():
                        self.writer.add_scalar(f'Train_Metrics/{key}', value, self.global_step)
            
            self.global_step += 1
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, loss_components
    
    def validate(self, epoch):
        """验证模型 / Validate model"""
        self.model.eval()
        total_loss = 0
        
        # 重置验证指标 / Reset validation metrics
        if self.use_efficient_metrics:
            self.val_metrics.reset()
        else:
            all_metrics = []
        
        with torch.no_grad():
            for lyso, hpge in tqdm(self.val_loader, desc='Validation'):
                lyso = lyso.to(self.device, non_blocking=True)
                hpge = hpge.to(self.device, non_blocking=True)
                
                # 混合精度推理 / Mixed precision inference
                if self.use_amp:
                    with autocast():
                        pred = self.model(lyso)
                        loss, _ = self.criterion(pred, hpge)
                else:
                    pred = self.model(lyso)
                    loss, _ = self.criterion(pred, hpge)
                
                total_loss += loss.item()
                
                # 更新指标 / Update metrics
                if self.use_efficient_metrics:
                    self.val_metrics.update(pred, hpge)
                else:
                    # 采样计算指标 / Sample calculation of metrics
                    if len(all_metrics) * 10 < len(self.val_loader):
                        if np.random.random() < 0.1:
                            metrics = SpectralMetrics.compute_all_metrics(pred[0], hpge[0], lyso[0])
                            all_metrics.append(metrics)
        
        avg_loss = total_loss / len(self.val_loader)
        
        # 计算最终指标 / Calculate final metrics
        if self.use_efficient_metrics:
            final_metrics = self.val_metrics.compute()
            
            # 记录到TensorBoard / Record to TensorBoard
            self.writer.add_scalar('Loss/val', avg_loss, epoch)
            for key, value in final_metrics.items():
                self.writer.add_scalar(f'Val_Metrics/{key}', value, epoch)
            
            # 打印主要指标 / Print main metrics
            print(f"\nValidation Results:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  FWHM Improvement: {final_metrics.get('fwhm_improvement_pct', 0):.2f}%")
            print(f"  Spectral Correlation: {final_metrics.get('spectral_correlation', 0):.4f}")
            print(f"  Peak Intensity Ratio: {final_metrics.get('peak_intensity_ratio', 0):.4f}")
            
            return avg_loss, final_metrics
        else:
            # 原有方法 / Original method
            avg_metrics = {}
            if all_metrics:
                for key in all_metrics[0].keys():
                    values = [m[key] for m in all_metrics if key in m]
                    if values:
                        avg_metrics[key] = np.mean(values)
            
            self.writer.add_scalar('Loss/val', avg_loss, epoch)
            for key, value in avg_metrics.items():
                self.writer.add_scalar(f'Metrics/{key}', value, epoch)
            
            return avg_loss, avg_metrics
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """保存检查点，使用统一的CheckpointManager / Save checkpoint using unified CheckpointManager"""
        scaler = self.scaler if self.use_amp else None
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            val_loss=val_loss,
            config=self.config_manager.config,
            scaler=scaler,
            is_best=is_best
        )
        
        if is_best:
            print(f"Saved new best model (val_loss: {val_loss:.4f}) at epoch {epoch}")
            
            # 保存模型信息 / Save model information
            info_path = self.checkpoint_dir / 'best_model_info.txt'
            with open(info_path, 'w') as f:
                f.write(f"Best Model Information\n")
                f.write(f"=====================\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Validation Loss: {val_loss:.6f}\n")
                f.write(f"Training Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {self.config_manager.get('model.name', 'SpectralResNet1D')}\n")
                f.write(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}\n")
        
        return checkpoint_path
    
    def train(self):
        """训练主循环 / Main training loop"""
        training_config = self.config_manager.get_training_config()
        num_epochs = training_config.get('num_epochs', 100)
        
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Model: {self.config_manager.get('model.name', 'SpectralResNet1D')}")
        print(f"Loss Type: {self.config_manager.get('loss.type', 'original')}")
        print(f"Batch Size: {training_config.get('batch_size', 16)}")
        print(f"Learning Rate: {training_config.get('learning_rate', 1e-3)}")
        print("-" * 50)
        
        for epoch in range(num_epochs):
            # 训练 / Training
            train_loss, train_components = self.train_epoch(epoch)
            
            # 验证 / Validation
            val_loss, val_metrics = self.validate(epoch)
            
            # 更新学习率 - 正确处理调度器步进 / Update learning rate - properly handle scheduler stepping
            step_scheduler(self.scheduler, epoch, val_loss)
            
            # 检查是否是最佳模型（基于验证损失） / Check if it's the best model (based on validation loss)
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
            
            # 打印epoch总结 / Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f} {'(BEST)' if is_best else ''}")
            print(f"  Best Val Loss: {self.best_val_loss:.4f} (epoch {self.best_epoch})")
            
            # 保存检查点 / Save checkpoint
            if (epoch + 1) % self.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
        
        # 训练结束 / Training complete
        print("\n" + "=" * 50)
        print("Training Complete!")
        print(f"Best Validation Loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        print("=" * 50)
        
        self.writer.close()
    
    def resume_from_checkpoint(self, checkpoint_path):
        """从检查点恢复训练，使用统一的CheckpointManager / Resume training from checkpoint using unified CheckpointManager"""
        scaler = self.scaler if self.use_amp else None
        
        resume_info = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=scaler,
            strict_loading=False,
            device=self.device
        )
        
        # 处理AMP状态不一致 - 根据检查点调整当前配置 / Handle AMP state inconsistency - adjust current config based on checkpoint
        checkpoint_config = resume_info.get('checkpoint_config', {})
        checkpoint_has_amp = checkpoint_config.get('device', {}).get('mixed_precision', False)
        
        if checkpoint_has_amp and not self.use_amp:
            print("Warning: Checkpoint was trained with mixed precision but current config has it disabled.")
            print("Enabling mixed precision to match checkpoint...")
            self.use_amp = True
            self.scaler = GradScaler()
            # 重新加载以获取scaler状态 / Reload to get scaler state
            self.checkpoint_manager.load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                strict_loading=False,
                device=self.device
            )
        elif not checkpoint_has_amp and self.use_amp:
            print("Warning: Checkpoint was trained without mixed precision but current config has it enabled.")
            print("Disabling mixed precision to match checkpoint...")
            self.use_amp = False
            self.scaler = None
        
        self.best_val_loss = resume_info.get('val_loss', float('inf'))
        self.best_epoch = resume_info.get('epoch', -1)
        
        start_epoch = resume_info['start_epoch']
        print(f"Resumed from epoch {start_epoch}")
        
        return start_epoch


def main():
    parser = argparse.ArgumentParser(description='Improved CNN Training with Config Management')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # 允许命令行覆盖配置 / Allow command line override of configuration
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--gpus', type=int, default=None)
    
    args = parser.parse_args()
    
    # 创建配置管理器 / Create configuration manager
    config_manager = ConfigManager()
    
    # 解析参数并更新配置 / Parse arguments and update configuration
    parsed_args = config_manager.parse_args_and_update()
    
    # 验证配置 / Validate configuration
    try:
        config_manager.validate_config()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Creating default configuration files...")
        from utils.config_manager import create_default_configs
        create_default_configs()
        print("Please check the configs directory and update the configuration files.")
        return
    
    # 打印配置 / Print configuration
    config_manager.print_config()
    
    # 创建训练器 / Create trainer
    trainer = ImprovedTrainer(config_manager)
    
    # 如果指定了恢复检查点 / If resume checkpoint is specified
    if parsed_args.resume:
        start_epoch = trainer.resume_from_checkpoint(parsed_args.resume)
        # 更新epoch数量 / Update number of epochs
        config_manager.set('training.num_epochs', 
                          config_manager.get('training.num_epochs', 100) - start_epoch)
    
    # 开始训练 / Start training
    trainer.train()


if __name__ == '__main__':
    main()