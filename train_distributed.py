import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import argparse
import json
import os
from tqdm import tqdm

from models.resnet1d import SpectralResNet1D
from utils.dataset_large import create_large_data_loaders, LargeSpectralDataset
from utils.losses import SpectralCompositeLoss
from utils.metrics import SpectralMetrics


class DistributedTrainer:
    def __init__(self, config, rank, world_size):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(self.device)
        
        # 初始化模型
        self.model = SpectralResNet1D(
            input_channels=1,
            num_blocks=config['num_blocks'],
            channels=config['channels']
        ).to(self.device)
        
        # 包装为DDP模型
        self.model = DDP(self.model, device_ids=[rank])
        
        # 损失函数
        self.criterion = SpectralCompositeLoss(
            peak_weight=config['peak_weight'],
            compton_weight=config['compton_weight'],
            smoothness_weight=config['smoothness_weight']
        )
        
        # 优化器
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config['learning_rate'],
                momentum=0.9,
                weight_decay=config['weight_decay']
            )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config['T_0'],
            T_mult=config['T_mult']
        )
        
        # 数据加载器
        self.setup_data_loaders()
        
        # TensorBoard（仅主进程）
        if rank == 0:
            self.writer = SummaryWriter(config['log_dir'])
        
        self.best_val_loss = float('inf')
        self.global_step = 0
        
    def setup_data_loaders(self):
        # 创建数据集
        train_dataset = LargeSpectralDataset(
            self.config['train_path'],
            use_mmap=True
        )
        val_dataset = LargeSpectralDataset(
            self.config['val_path'],
            use_mmap=True
        )
        
        # 分布式采样器
        self.train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        self.val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        )
        
        # 数据加载器
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'] // self.world_size,
            sampler=self.train_sampler,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'] // self.world_size,
            sampler=self.val_sampler,
            num_workers=self.config['num_workers'] // 2,
            pin_memory=True
        )
    
    def train_epoch(self, epoch):
        self.model.train()
        self.train_sampler.set_epoch(epoch)  # 重要：确保每个epoch的数据打乱不同
        
        total_loss = 0
        loss_components = {'peak_loss': 0, 'compton_loss': 0, 'smoothness_loss': 0}
        
        if self.rank == 0:
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        else:
            pbar = self.train_loader
        
        for batch_idx, (lyso, hpge) in enumerate(pbar):
            lyso = lyso.to(self.device, non_blocking=True)
            hpge = hpge.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            pred = self.model(lyso)
            loss, components = self.criterion(pred, hpge)
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += components[key]
            
            if self.rank == 0 and batch_idx % 10 == 0:
                pbar.set_postfix({'loss': loss.item()})
                self.writer.add_scalar('Loss/train', loss.item(), self.global_step)
                for key, value in components.items():
                    self.writer.add_scalar(f'Loss/{key}', value, self.global_step)
            
            self.global_step += 1
        
        # 同步所有进程的损失
        avg_loss = total_loss / len(self.train_loader)
        avg_loss_tensor = torch.tensor(avg_loss).to(self.device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / self.world_size
        
        return avg_loss, loss_components
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        all_metrics = []
        
        with torch.no_grad():
            for lyso, hpge in tqdm(self.val_loader, desc='Validation', disable=self.rank != 0):
                lyso = lyso.to(self.device, non_blocking=True)
                hpge = hpge.to(self.device, non_blocking=True)
                
                pred = self.model(lyso)
                loss, _ = self.criterion(pred, hpge)
                total_loss += loss.item()
                
                # 仅计算部分metrics以加速
                if len(all_metrics) < 10:
                    metrics = SpectralMetrics.compute_all_metrics(pred[0], hpge[0], lyso[0])
                    all_metrics.append(metrics)
        
        # 同步验证损失
        avg_loss = total_loss / len(self.val_loader)
        avg_loss_tensor = torch.tensor(avg_loss).to(self.device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / self.world_size
        
        # 计算平均metrics
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m]
                if values:
                    avg_metrics[key] = np.mean(values)
        
        if self.rank == 0:
            self.writer.add_scalar('Loss/val', avg_loss, epoch)
            for key, value in avg_metrics.items():
                self.writer.add_scalar(f'Metrics/{key}', value, epoch)
        
        return avg_loss, avg_metrics
    
    def save_checkpoint(self, epoch, val_loss):
        if self.rank != 0:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),  # 注意：使用.module访问原始模型
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        Path(self.config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, f"{self.config['checkpoint_dir']}/checkpoint_epoch_{epoch}.pth")
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, f"{self.config['checkpoint_dir']}/best_model.pth")
    
    def train(self):
        for epoch in range(self.config['num_epochs']):
            train_loss, train_components = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate(epoch)
            
            self.scheduler.step()
            
            if self.rank == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                print(f"Val Metrics: {val_metrics}")
                
                if (epoch + 1) % self.config['save_interval'] == 0:
                    self.save_checkpoint(epoch, val_loss)
        
        if self.rank == 0:
            self.writer.close()
    
    def cleanup(self):
        dist.destroy_process_group()


def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def train_worker(rank, world_size, config):
    """每个进程的训练函数"""
    setup(rank, world_size)
    
    trainer = DistributedTrainer(config, rank, world_size)
    trainer.train()
    trainer.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_distributed.json')
    parser.add_argument('--gpus', type=int, default=None, help='Number of GPUs to use')
    args = parser.parse_args()
    
    # 默认配置
    default_config = {
        'train_path': 'D:/mechine-learning/CNN/dataset/train',
        'val_path': 'D:/mechine-learning/CNN/dataset/val',
        'log_dir': 'logs_distributed',
        'checkpoint_dir': 'checkpoints_distributed',
        'num_epochs': 100,
        'batch_size': 128,  # 总批次大小，会被分配到各GPU
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'num_blocks': 12,
        'channels': 64,
        'peak_weight': 10.0,
        'compton_weight': 1.0,
        'smoothness_weight': 0.1,
        'optimizer': 'adam',
        'T_0': 10,
        'T_mult': 2,
        'num_workers': 16,  # 增加worker数量
        'save_interval': 5
    }
    
    # 加载配置
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
        default_config.update(config)
    else:
        with open(args.config, 'w') as f:
            json.dump(default_config, f, indent=4)
        print(f"Created default config file: {args.config}")
    
    # 确定GPU数量
    if args.gpus is None:
        world_size = torch.cuda.device_count()
    else:
        world_size = args.gpus
    
    print(f"Starting distributed training on {world_size} GPUs")
    
    # 启动多进程训练
    torch.multiprocessing.spawn(
        train_worker,
        args=(world_size, default_config),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()