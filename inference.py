import torch
import numpy as np
import argparse
from pathlib import Path
import json
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.resnet1d import SpectralResNet1D
from utils.dataset import SpectralDataset
from utils.metrics import SpectralMetrics, PreciseSpectralMetrics
from utils.efficient_metrics import EfficientSpectralMetrics
from utils.visualization import SpectralVisualizer


class SpectralEnhancer:
    def __init__(self, checkpoint_path, device=None, use_efficient_metrics=False):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_efficient_metrics = use_efficient_metrics
        print(f"Using device: {self.device}")
        print(f"Using efficient metrics: {use_efficient_metrics}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint['config']
        
        self.model = SpectralResNet1D(
            input_channels=1,
            num_blocks=self.config.get('num_blocks', 12),
            channels=self.config.get('channels', 64)
        ).to(self.device)
        
        # 处理 DDP 模型的兼容性
        state_dict = checkpoint['model_state_dict']
        # 如果是 DDP 模型保存的，移除 'module.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # 移除 'module.' 前缀
            else:
                new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        
        self.visualizer = SpectralVisualizer()
        
        # 初始化高效指标（如果启用）
        if self.use_efficient_metrics:
            self.efficient_metrics = EfficientSpectralMetrics().to(self.device)
        
    def enhance_spectrum(self, lyso_spectrum):
        if isinstance(lyso_spectrum, np.ndarray):
            lyso_spectrum = torch.from_numpy(lyso_spectrum.astype(np.float32))
        
        if len(lyso_spectrum.shape) == 1:
            lyso_spectrum = lyso_spectrum.unsqueeze(0).unsqueeze(0)
        elif len(lyso_spectrum.shape) == 2:
            lyso_spectrum = lyso_spectrum.unsqueeze(0)
        
        lyso_spectrum = lyso_spectrum / (torch.max(lyso_spectrum) + 1e-8)
        
        lyso_spectrum = lyso_spectrum.to(self.device)
        
        with torch.no_grad():
            enhanced_spectrum = self.model(lyso_spectrum)
        
        return enhanced_spectrum
    
    def process_file(self, input_path, output_path=None, visualize=True, target_spectrum=None):
        if input_path.endswith('.h5'):
            with h5py.File(input_path, 'r') as f:
                lyso_spectrum = f['lyso'][:]
                if 'hpge' in f:
                    target_spectrum = f['hpge'][:]
        elif input_path.endswith('.npy'):
            data = np.load(input_path)
            if isinstance(data, np.ndarray):
                lyso_spectrum = data
            else:
                lyso_spectrum = data['lyso']
                if 'hpge' in data:
                    target_spectrum = data['hpge']
        else:
            raise ValueError(f"Unsupported file format: {input_path}")
        
        enhanced_spectrum = self.enhance_spectrum(lyso_spectrum)
        
        if target_spectrum is not None:
            # 使用精确指标进行推理评估
            target_tensor = torch.from_numpy(target_spectrum.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            original_tensor = torch.from_numpy(lyso_spectrum.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            
            if self.use_efficient_metrics:
                # 使用高效GPU指标
                self.efficient_metrics.reset()
                self.efficient_metrics.update(enhanced_spectrum, target_tensor)
                metrics = self.efficient_metrics.compute()
                
                # 获取当前批次详细指标
                batch_metrics = self.efficient_metrics.get_current_batch_metrics(enhanced_spectrum, target_tensor)
                metrics.update(batch_metrics)
                
                print(f"\n🚀 高效GPU指标评估结果:")
            else:
                # 使用精确CPU指标
                # 基础指标
                metrics = SpectralMetrics.compute_all_metrics(
                    enhanced_spectrum, target_tensor, original_tensor
                )
                
                # 详细分析指标
                detailed_metrics = PreciseSpectralMetrics.compute_detailed_metrics(
                    enhanced_spectrum, target_tensor, original_tensor
                )
                
                # 合并指标
                metrics.update(detailed_metrics)
                
                print(f"\n🔬 精确CPU指标评估结果:")
            
            print("=" * 40)
            
            # 核心指标
            fwhm_key = 'fwhm_improvement_pct' if self.use_efficient_metrics else 'fwhm_improvement'
            if fwhm_key in metrics:
                print(f"📊 FWHM改善: {metrics[fwhm_key]:.2f}%")
            if 'precise_fwhm_improvement' in metrics:
                print(f"   精确FWHM改善: {metrics['precise_fwhm_improvement']:.2f}%")
            
            # 光谱质量
            if 'spectral_correlation' in metrics:
                print(f"📈 光谱相关性: {metrics['spectral_correlation']:.4f}")
            if 'frequency_correlation' in metrics:
                print(f"   频域相关性: {metrics['frequency_correlation']:.4f}")
            
            # 峰特性
            if 'peak_intensity_ratio' in metrics:
                print(f"🎯 峰强度比: {metrics['peak_intensity_ratio']:.4f}")
            if 'centroid_shift' in metrics:
                print(f"   质心偏移: {metrics['centroid_shift']:.2f}")
            elif 'centroid_error' in metrics:
                print(f"   质心误差: {metrics['centroid_error']:.2f}")
            
            # 噪声抑制
            if 'noise_reduction_pct' in metrics:
                print(f"🔇 噪声减少: {metrics['noise_reduction_pct']:.2f}%")
            elif 'snr_improvement_pct' in metrics:
                print(f"🔇 信噪比改善: {metrics['snr_improvement_pct']:.2f}%")
            
            print("=" * 40)
            
            # 详细指标（可选显示）
            excluded_keys = {'fwhm_improvement', 'fwhm_improvement_pct', 'precise_fwhm_improvement', 
                           'spectral_correlation', 'frequency_correlation', 'peak_intensity_ratio', 
                           'centroid_shift', 'centroid_error', 'noise_reduction_pct', 'snr_improvement_pct'}
            
            other_metrics = {k: v for k, v in metrics.items() if k not in excluded_keys}
            if other_metrics:
                print("详细指标:")
                for key, value in other_metrics.items():
                    print(f"  {key}: {value:.4f}")
        
        if visualize:
            if target_spectrum is not None:
                self.visualizer.plot_spectrum_comparison(
                    lyso_spectrum, target_spectrum, enhanced_spectrum,
                    title=f"Enhancement Results - {Path(input_path).stem}",
                    save_name=f"comparison_{Path(input_path).stem}.png"
                )
                self.visualizer.plot_metrics_comparison(
                    lyso_spectrum, target_spectrum, enhanced_spectrum,
                    save_name=f"metrics_{Path(input_path).stem}.png"
                )
                self.visualizer.plot_residuals(
                    target_spectrum, enhanced_spectrum,
                    save_name=f"residuals_{Path(input_path).stem}.png"
                )
            else:
                plt.figure(figsize=(12, 6))
                channels = np.arange(4096)
                plt.plot(channels, lyso_spectrum.squeeze(), 'b-', label='LYSO (Input)', alpha=0.7)
                plt.plot(channels, enhanced_spectrum.cpu().numpy().squeeze(), 'r-', label='CNN Enhanced', alpha=0.8)
                plt.xlabel('Channel')
                plt.ylabel('Counts (normalized)')
                plt.title('Spectrum Enhancement')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()
        
        if output_path:
            output_data = {
                'lyso_original': lyso_spectrum,
                'enhanced': enhanced_spectrum.cpu().numpy().squeeze(),
            }
            if target_spectrum is not None:
                output_data['hpge_target'] = target_spectrum
                output_data['metrics'] = metrics
            
            np.savez(output_path, **output_data)
            print(f"Results saved to: {output_path}")
        
        return enhanced_spectrum.cpu().numpy().squeeze()
    
    def batch_process(self, input_dir, output_dir=None, file_pattern='*.h5'):
        input_path = Path(input_dir)
        files = list(input_path.glob(file_pattern))
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        all_metrics = []
        
        for file in tqdm(files, desc="Processing files"):
            try:
                if output_dir:
                    out_file = output_path / f"{file.stem}_enhanced.npz"
                else:
                    out_file = None
                
                self.process_file(str(file), out_file, visualize=False)
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
        
        print(f"\nProcessed {len(files)} files")


def main():
    parser = argparse.ArgumentParser(description='Spectral Enhancement Inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input file or directory path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file or directory path')
    parser.add_argument('--batch', action='store_true',
                       help='Process multiple files in a directory')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize results')
    parser.add_argument('--pattern', type=str, default='*.h5',
                       help='File pattern for batch processing')
    parser.add_argument('--use-efficient-metrics', action='store_true',
                       help='Use efficient GPU metrics for evaluation')
    
    args = parser.parse_args()
    
    enhancer = SpectralEnhancer(args.checkpoint, use_efficient_metrics=args.use_efficient_metrics)
    
    if args.batch:
        enhancer.batch_process(args.input, args.output, args.pattern)
    else:
        enhancer.process_file(args.input, args.output, args.visualize)


if __name__ == '__main__':
    main()