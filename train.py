import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json

from models.resnet1d import SpectralResNet1D
from utils.dataset import create_data_loaders
from utils.losses import SpectralCompositeLoss
from utils.metrics import SpectralMetrics


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = SpectralResNet1D(
            input_channels=1,
            num_blocks=config['num_blocks'],
            channels=config['channels']
        ).to(self.device)
        
        self.criterion = SpectralCompositeLoss(
            peak_weight=config['peak_weight'],
            compton_weight=config['compton_weight'],
            smoothness_weight=config['smoothness_weight']
        )
        
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
        
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config['T_0'],
            T_mult=config['T_mult']
        )
        
        self.train_loader, self.val_loader = create_data_loaders(
            config['train_path'],
            config['val_path'],
            config['batch_size'],
            config['num_workers']
        )
        
        self.writer = SummaryWriter(config['log_dir'])
        self.best_val_loss = float('inf')
        self.global_step = 0
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loss_components = {'peak_loss': 0, 'compton_loss': 0, 'smoothness_loss': 0}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (lyso, hpge) in enumerate(pbar):
            lyso = lyso.to(self.device)
            hpge = hpge.to(self.device)
            
            self.optimizer.zero_grad()
            
            pred = self.model(lyso)
            loss, components = self.criterion(pred, hpge)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += components[key]
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({'loss': loss.item()})
                self.writer.add_scalar('Loss/train', loss.item(), self.global_step)
                for key, value in components.items():
                    self.writer.add_scalar(f'Loss/{key}', value, self.global_step)
                
            self.global_step += 1
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, loss_components
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        all_metrics = []
        
        with torch.no_grad():
            for lyso, hpge in tqdm(self.val_loader, desc='Validation'):
                lyso = lyso.to(self.device)
                hpge = hpge.to(self.device)
                
                pred = self.model(lyso)
                loss, _ = self.criterion(pred, hpge)
                total_loss += loss.item()
                
                metrics = SpectralMetrics.compute_all_metrics(pred[0], hpge[0], lyso[0])
                all_metrics.append(metrics)
        
        avg_loss = total_loss / len(self.val_loader)
        
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            if values:
                avg_metrics[key] = np.mean(values)
        
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        for key, value in avg_metrics.items():
            self.writer.add_scalar(f'Metrics/{key}', value, epoch)
        
        return avg_loss, avg_metrics
    
    def save_checkpoint(self, epoch, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
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
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val Metrics: {val_metrics}")
            
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch, val_loss)
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    args = parser.parse_args()
    
    default_config = {
        'train_path': 'D:/mechine-learning/CNN/dataset/train',
        'val_path': 'D:/mechine-learning/CNN/dataset/val',
        'log_dir': 'logs',
        'checkpoint_dir': 'checkpoints',
        'num_epochs': 100,
        'batch_size': 16,
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
        'num_workers': 4,
        'save_interval': 5
    }
    
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
        default_config.update(config)
    else:
        with open('config.json', 'w') as f:
            json.dump(default_config, f, indent=4)
        print(f"Created default config file: config.json")
    
    trainer = Trainer(default_config)
    trainer.train()


if __name__ == '__main__':
    main()