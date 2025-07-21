import numpy as np
import h5py
from pathlib import Path
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def create_synthetic_spectrum(num_channels=4096, num_peaks=5, resolution_factor=1.0, noise_level=0.05):
    spectrum = np.zeros(num_channels)
    
    peak_positions = np.random.randint(500, num_channels-500, num_peaks)
    peak_heights = np.random.uniform(0.3, 1.0, num_peaks)
    peak_widths = np.random.uniform(5, 20, num_peaks) * resolution_factor
    
    for pos, height, width in zip(peak_positions, peak_heights, peak_widths):
        x = np.arange(num_channels)
        spectrum += height * np.exp(-(x - pos)**2 / (2 * width**2))
    
    background = 0.1 * np.exp(-np.arange(num_channels) / 1000) + 0.05
    spectrum += background
    
    compton_edges = peak_positions * 0.7
    for edge in compton_edges:
        edge_int = int(edge)
        if edge_int < num_channels - 100:
            compton = 0.2 * np.exp(-np.arange(num_channels - edge_int) / 200)
            spectrum[edge_int:] += compton[:num_channels - edge_int]
    
    noise = np.random.normal(0, noise_level, num_channels)
    spectrum += noise
    
    spectrum = np.maximum(spectrum, 0)
    spectrum = spectrum / np.max(spectrum)
    
    return spectrum


def create_sample_dataset(output_dir, num_samples=100):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_path = output_path / 'train'
    val_path = output_path / 'val'
    test_path = output_path / 'test'
    
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)
    test_path.mkdir(exist_ok=True)
    
    train_samples = int(num_samples * 0.7)
    val_samples = int(num_samples * 0.15)
    test_samples = num_samples - train_samples - val_samples
    
    print(f"Creating {train_samples} training samples...")
    for i in range(train_samples):
        lyso_spectrum = create_synthetic_spectrum(resolution_factor=2.0, noise_level=0.08)
        hpge_spectrum = create_synthetic_spectrum(resolution_factor=1.0, noise_level=0.02)
        
        with h5py.File(train_path / f'sample_{i:04d}.h5', 'w') as f:
            f.create_dataset('lyso', data=lyso_spectrum.astype(np.float32))
            f.create_dataset('hpge', data=hpge_spectrum.astype(np.float32))
    
    print(f"Creating {val_samples} validation samples...")
    for i in range(val_samples):
        lyso_spectrum = create_synthetic_spectrum(resolution_factor=2.0, noise_level=0.08)
        hpge_spectrum = create_synthetic_spectrum(resolution_factor=1.0, noise_level=0.02)
        
        with h5py.File(val_path / f'sample_{i:04d}.h5', 'w') as f:
            f.create_dataset('lyso', data=lyso_spectrum.astype(np.float32))
            f.create_dataset('hpge', data=hpge_spectrum.astype(np.float32))
    
    print(f"Creating {test_samples} test samples...")
    for i in range(test_samples):
        lyso_spectrum = create_synthetic_spectrum(resolution_factor=2.0, noise_level=0.08)
        hpge_spectrum = create_synthetic_spectrum(resolution_factor=1.0, noise_level=0.02)
        
        with h5py.File(test_path / f'sample_{i:04d}.h5', 'w') as f:
            f.create_dataset('lyso', data=lyso_spectrum.astype(np.float32))
            f.create_dataset('hpge', data=hpge_spectrum.astype(np.float32))
    
    print(f"\nDataset created successfully!")
    print(f"Total samples: {num_samples}")
    print(f"Training: {train_samples}")
    print(f"Validation: {val_samples}")
    print(f"Test: {test_samples}")
    print(f"Location: {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create synthetic spectral dataset')
    parser.add_argument('--output_dir', type=str, default='D:/mechine-learning/CNN/dataset',
                       help='Output directory for the dataset')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Total number of samples to generate')
    
    args = parser.parse_args()
    
    create_sample_dataset(args.output_dir, args.num_samples)