#!/bin/bash

# 视频超分辨率处理程序安装脚本
# 自动检测系统环境并安装必要的依赖

set -e  # 遇到错误时退出

echo "🚀 开始安装视频超分辨率处理程序..."

# 检查Python版本
echo "📋 检查Python版本..."
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.7"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 7) else 1)"; then
    echo "✅ Python版本检查通过: $python_version"
else
    echo "❌ 需要Python 3.7或更高版本，当前版本: $python_version"
    exit 1
fi

# 检查pip
echo "📋 检查pip..."
if command -v pip3 &> /dev/null; then
    echo "✅ pip3已安装"
else
    echo "❌ pip3未找到，请先安装pip"
    exit 1
fi

# 检测操作系统
OS="Unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    OS="Windows"
fi

echo "🖥️  检测到操作系统: $OS"

# 安装FFmpeg
echo "📦 安装FFmpeg..."
case $OS in
    "Linux")
        if command -v apt-get &> /dev/null; then
            echo "使用apt-get安装FFmpeg..."
            sudo apt-get update
            sudo apt-get install -y ffmpeg
        elif command -v yum &> /dev/null; then
            echo "使用yum安装FFmpeg..."
            sudo yum install -y ffmpeg
        elif command -v dnf &> /dev/null; then
            echo "使用dnf安装FFmpeg..."
            sudo dnf install -y ffmpeg
        else
            echo "⚠️  无法自动安装FFmpeg，请手动安装"
        fi
        ;;
    "macOS")
        if command -v brew &> /dev/null; then
            echo "使用Homebrew安装FFmpeg..."
            brew install ffmpeg
        else
            echo "⚠️  请先安装Homebrew，然后运行: brew install ffmpeg"
        fi
        ;;
    "Windows")
        echo "⚠️  Windows用户请手动安装FFmpeg并添加到PATH"
        ;;
    *)
        echo "⚠️  未知操作系统，请手动安装FFmpeg"
        ;;
esac

# 检查CUDA
echo "🔍 检查CUDA支持..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ 检测到NVIDIA GPU"
    nvidia-smi --query-gpu=name --format=csv,noheader,nounits
    CUDA_AVAILABLE=true
else
    echo "⚠️  未检测到NVIDIA GPU，将使用CPU模式"
    CUDA_AVAILABLE=false
fi

# 创建必要的目录
echo "📁 创建目录结构..."
mkdir -p weights
mkdir -p inputs
mkdir -p outputs

# 安装Python依赖
echo "📦 安装Python依赖包..."

# 升级pip
pip3 install --upgrade pip

# 根据CUDA支持情况安装PyTorch
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "🔥 安装CUDA版本的PyTorch..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 安装CPU版本的PyTorch..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 安装其他依赖
echo "📦 安装其他依赖包..."
pip3 install opencv-python
pip3 install numpy
pip3 install basicsr
pip3 install realesrgan
pip3 install gfpgan
pip3 install facexlib
pip3 install tqdm
pip3 install Pillow

# 测试安装
echo "🧪 测试安装..."
python3 -c "
try:
    import torch
    import cv2
    import numpy as np
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    print('✅ 所有依赖包安装成功')
    print(f'PyTorch版本: {torch.__version__}')
    print(f'CUDA可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU数量: {torch.cuda.device_count()}')
        print(f'当前GPU: {torch.cuda.get_device_name(0)}')
except ImportError as e:
    print(f'❌ 依赖包导入失败: {e}')
    exit(1)
"

# 显示使用说明
echo ""
echo "🎉 安装完成！"
echo ""
echo "📖 使用方法："
echo "  1. 将视频文件放入 inputs/ 目录"
echo "  2. 运行命令："
echo "     python3 video_super_resolution.py -i inputs/your_video.mp4 -o outputs/enhanced_video.mp4"
echo ""
echo "📚 更多用法请参考 README.md"
echo ""

# 询问是否下载模型
echo "❓ 是否现在下载默认模型文件? (RealESRGAN_x4plus, ~65MB) [y/N]"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "📥 下载模型文件..."
    
    model_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    model_path="weights/RealESRGAN_x4plus.pth"
    
    if command -v wget &> /dev/null; then
        wget -O "$model_path" "$model_url"
    elif command -v curl &> /dev/null; then
        curl -L -o "$model_path" "$model_url"
    else
        echo "⚠️  请手动下载模型文件："
        echo "   $model_url"
        echo "   保存到: $model_path"
    fi
    
    if [ -f "$model_path" ]; then
        echo "✅ 模型文件下载成功"
    else
        echo "❌ 模型文件下载失败"
    fi
fi

echo ""
echo "🚀 安装脚本执行完成！" 