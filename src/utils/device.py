import torch

def get_device():
    """사용 가능한 디바이스 반환 (CUDA, MPS, CPU 순서로 우선순위)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device
