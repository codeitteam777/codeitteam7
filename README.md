# 환경 세팅 가이드

## 레포지토리 최신화:
- git pull origin main

## 환경 생성 (처음이라면)
- conda env create -f environment.yml
- conda activate codeit_project_env

## 이미 환경이 있다면 업데이트만:
conda env update -f environment.yml --prune

## GPU 팀원 
### CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

### CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

### Mac 팀원
pip install torch torchvision torchaudio

## 장치 확인 코드
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
