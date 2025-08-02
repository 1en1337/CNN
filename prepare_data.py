"""
数据准备脚本 / Data preparation script
自动将data目录中的数据分割为训练集、验证集和测试集
Automatically split data in data directory into training, validation and test sets
"""
import os
import shutil
from pathlib import Path
import random
import argparse


def prepare_dataset(source_dir='data', output_dir='dataset_split', 
                   train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                   seed=42):
    """
    将源目录中的数据文件分割为训练集、验证集和测试集
    Split data files from source directory into training, validation and test sets
    
    Args:
        source_dir: 源数据目录 / Source data directory
        output_dir: 输出目录 / Output directory
        train_ratio: 训练集比例 / Training set ratio
        val_ratio: 验证集比例 / Validation set ratio
        test_ratio: 测试集比例 / Test set ratio
        seed: 随机种子 / Random seed
    """
    # 设置随机种子 / Set random seed
    random.seed(seed)
    
    # 创建路径 / Create paths
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # 检查源目录 / Check source directory
    if not source_path.exists():
        print(f"错误：源目录 '{source_dir}' 不存在！ / Error: Source directory '{source_dir}' does not exist!")
        print(f"请将你的数据文件放在 {source_path.absolute()} 目录中 / Please put your data files in {source_path.absolute()} directory")
        return False
    
    # 获取所有数据文件 / Get all data files
    data_files = list(source_path.glob('*.h5')) + list(source_path.glob('*.hdf5'))
    
    if not data_files:
        print(f"错误：在 '{source_dir}' 中没有找到 .h5 或 .hdf5 文件！ / Error: No .h5 or .hdf5 files found in '{source_dir}'!")
        return False
    
    print(f"找到 {len(data_files)} 个数据文件 / Found {len(data_files)} data files")
    
    # 随机打乱文件列表 / Shuffle file list randomly
    random.shuffle(data_files)
    
    # 计算分割点 / Calculate split points
    total = len(data_files)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    
    # 分割数据 / Split data
    train_files = data_files[:train_size]
    val_files = data_files[train_size:train_size + val_size]
    test_files = data_files[train_size + val_size:]
    
    # 创建输出目录 / Create output directories
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    test_dir = output_path / 'test'
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 复制文件到相应目录 / Copy files to corresponding directories
    print("\n分割数据集... / Splitting dataset...")
    print(f"训练集: {len(train_files)} 个文件 / Training set: {len(train_files)} files")
    for f in train_files:
        shutil.copy2(f, train_dir / f.name)
    
    print(f"验证集: {len(val_files)} 个文件 / Validation set: {len(val_files)} files")
    for f in val_files:
        shutil.copy2(f, val_dir / f.name)
    
    print(f"测试集: {len(test_files)} 个文件 / Test set: {len(test_files)} files")
    for f in test_files:
        shutil.copy2(f, test_dir / f.name)
    
    print(f"\n数据集准备完成！ / Dataset preparation complete!")
    print(f"输出目录 / Output directory: {output_path.absolute()}")
    print(f"├── train/ ({len(train_files)} 个文件 / {len(train_files)} files)")
    print(f"├── val/   ({len(val_files)} 个文件 / {len(val_files)} files)")
    print(f"└── test/  ({len(test_files)} 个文件 / {len(test_files)} files)")
    
    # 验证第一个文件 / Validate the first file
    try:
        import h5py
        sample_file = train_files[0]
        print(f"\n验证数据格式 / Validating data format (使用文件 / using file: {sample_file.name})...")
        with h5py.File(str(train_dir / sample_file.name), 'r') as f:
            keys = list(f.keys())
            print(f"数据集包含 / Dataset contains: {keys}")
            
            if 'lyso' in keys and 'hpge' in keys:
                lyso_shape = f['lyso'].shape
                hpge_shape = f['hpge'].shape
                print(f"LYSO 形状 / LYSO shape: {lyso_shape}")
                print(f"HPGe 形状 / HPGe shape: {hpge_shape}")
                print("✓ 数据格式正确！ / Data format is correct!")
            else:
                print("警告：数据文件应包含 'lyso' 和 'hpge' 数据集 / Warning: Data files should contain 'lyso' and 'hpge' datasets")
    except Exception as e:
        print(f"验证数据时出错 / Error validating data: {e}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='准备数据集 / Prepare dataset')
    parser.add_argument('--source', type=str, default='data',
                       help='源数据目录 (默认: data) / Source data directory (default: data)')
    parser.add_argument('--output', type=str, default='dataset_split',
                       help='输出目录 (默认: dataset_split) / Output directory (default: dataset_split)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='训练集比例 (默认: 0.7) / Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='验证集比例 (默认: 0.15) / Validation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='测试集比例 (默认: 0.15) / Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42) / Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # 验证比例 / Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"错误：比例之和必须为1.0，当前为 {total_ratio} / Error: Sum of ratios must be 1.0, current is {total_ratio}")
        return
    
    prepare_dataset(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()