import torch

def check_cuda_availability():
    """检查CUDA是否可用"""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"CUDA可用，当前设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("CUDA不可用，将使用CPU")
    return cuda_available

if __name__ == "__main__":
    check_cuda_availability()