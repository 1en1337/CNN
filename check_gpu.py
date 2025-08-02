"""
检查GPU信息的脚本
"""
import torch
import subprocess
import sys

def check_gpu_nvidia_smi():
    """使用nvidia-smi检查GPU"""
    print("=== NVIDIA-SMI GPU信息 ===")
    try:
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("nvidia-smi命令执行失败")
    except FileNotFoundError:
        print("找不到nvidia-smi，请确保已安装NVIDIA驱动")

def check_gpu_pytorch():
    """使用PyTorch检查GPU"""
    print("\n=== PyTorch GPU信息 ===")
    
    if not torch.cuda.is_available():
        print("CUDA不可用！")
        return
    
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  名称: {torch.cuda.get_device_name(i)}")
        print(f"  显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # 获取当前使用的显存
        if torch.cuda.is_available():
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  已分配显存: {allocated:.2f} GB")
            print(f"  已预留显存: {reserved:.2f} GB")
            print(f"  可用显存: {(torch.cuda.get_device_properties(i).total_memory / 1024**3 - reserved):.2f} GB")

def check_current_device():
    """检查当前默认设备"""
    print("\n=== 当前设备信息 ===")
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"当前默认GPU ID: {current_device}")
        print(f"当前默认GPU名称: {torch.cuda.get_device_name(current_device)}")
    else:
        print("当前使用CPU")

def main():
    print("GPU设备检查工具")
    print("=" * 50)
    
    # 检查nvidia-smi
    check_gpu_nvidia_smi()
    
    # 检查PyTorch GPU
    check_gpu_pytorch()
    
    # 检查当前设备
    check_current_device()
    
    print("\n" + "=" * 50)
    print("提示：")
    print("- GPU ID 从 0 开始编号")
    print("- 如果只有一块GPU，device_id 就是 0")
    print("- RTX 4060 应该显示约 8GB 显存")

if __name__ == "__main__":
    main()