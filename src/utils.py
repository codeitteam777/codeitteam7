import platform
from matplotlib import font_manager, rc
from matplotlib import pyplot as plt
import torch

def setup_matplotlib_korean_font():
    """matplotlib 한글 폰트 설정"""
    if platform.system() == 'Windows':
        font_name = font_manager.FontProperties(
            fname="c:/Windows/Fonts/malgun.ttf"
        ).get_name()
        rc('font', family=font_name)
    elif platform.system() == 'Darwin':
        rc('font', family='AppleGothic')
    else:
        try:
            font_name = font_manager.FontProperties(
                fname='/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
            ).get_name()
            rc('font', family=font_name)
        except:
            rc('font', family='DejaVu Sans')

    plt.rcParams['axes.unicode_minus'] = False


def get_device():
    """사용 가능한 디바이스 반환 (CUDA, MPS, CPU 순서로 우선순위)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device
