# models/__init__.py
import os
import sys

# 自动将models目录及子目录加入sys.path，方便内部模块导入
root_dir = os.path.dirname(__file__)
for root, dirs, files in os.walk(root_dir):
    if root not in sys.path:
        sys.path.append(root)

# 导入 ResNet18/ResNet 系列
try:
    from .resnet import ResNet18, ResNet34, ResNet50
except ImportError:
    pass  # 文件中可能未定义某些模型，忽略

# 导入 ViT 系列
try:
    from .vit import ViT, ViT_small, ViT_base
except ImportError:
    pass

# 导入 ConvMixer
try:
    from .convmixer import ConvMixer
except ImportError:
    pass

# 导入 U-Net 系列
try:
    from .U_Net_Zoo import UNet, UNet3D
except ImportError:
    pass

# 导入自定义 models.py 里面的模型（如果有）
try:
    from .models_factory import AllCNN, SomeOtherModel  # 替换 SomeOtherModel 为你实际需要的类
except ImportError:
    pass

# 导入工具函数
try:
    from .model_utils import *
except ImportError:
    pass

# 这里可以再手动添加其他模型类到顶级命名空间
__all__ = [
    'ResNet18', 'ResNet34', 'ResNet50',
    'ViT', 'ViT_small', 'ViT_base',
    'ConvMixer', 'UNet', 'UNet3D',
    'AllCNN',
]
