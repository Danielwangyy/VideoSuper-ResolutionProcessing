# 视频超分辨率处理程序

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于Real-ESRGAN技术的视频清晰度提升工具，支持多种模型和自定义配置，可将低分辨率视频提升到高分辨率。

## 📋 功能特点

- 🚀 **多种模型支持**：支持Real-ESRGAN系列模型，包括通用模型和动漫专用模型
- 🎥 **视频处理**：支持批量视频处理，保留音频轨道
- 🔧 **灵活配置**：可自定义缩放比例、质量、帧率等参数
- 👤 **人脸增强**：集成GFPGAN进行人脸修复和增强
- 💻 **GPU加速**：支持CUDA加速，提升处理速度
- 📊 **进度显示**：实时显示处理进度和状态

## 📦 安装

### 1. 克隆项目

```bash
git clone <repository-url>
cd video-super-resolution
```

### 2. 安装依赖

#### 使用pip安装（推荐）

```bash
pip install -r requirements.txt
```

#### 或者使用conda安装

```bash
conda create -n video-sr python=3.8
conda activate video-sr
pip install -r requirements.txt
```

### 3. 安装FFmpeg（用于音频处理）

**Windows:**
```bash
# 使用chocolatey
choco install ffmpeg

# 或手动下载并添加到PATH
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

### 4. 下载模型文件

模型文件会在首次运行时自动下载，或者您可以手动下载：

- [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) - 通用4倍超分辨率模型
- [RealESRGAN_x2plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth) - 通用2倍超分辨率模型
- [RealESRGAN_x4plus_anime_6B.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth) - 动漫专用模型
- [realesr-animevideov3.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth) - 动漫视频专用模型

下载后请将模型文件放置在 `weights/` 目录下。

## 🚀 使用方法

### 基本用法

```bash
# 处理单个视频文件（4倍超分辨率）
python video_super_resolution.py -i input_video.mp4 -o output_video.mp4

# 使用动漫专用模型
python video_super_resolution.py -i anime_video.mp4 -o enhanced_anime.mp4 -m RealESRGAN_x4plus_anime_6B

# 启用人脸增强
python video_super_resolution.py -i face_video.mp4 -o enhanced_face.mp4 --face-enhance
```

### 高级配置

```bash
# 自定义缩放比例和帧率
python video_super_resolution.py -i input.mp4 -o output.mp4 -s 3.5 --fps 30

# 批量处理目录中的所有视频
python video_super_resolution.py -i input_dir/ -o output_dir/ --batch

# 使用CPU处理（内存不足时）
python video_super_resolution.py -i input.mp4 -o output.mp4 --device cpu

# 启用分块处理（显存不足时）
python video_super_resolution.py -i input.mp4 -o output.mp4 --tile 400

# 不保留音频
python video_super_resolution.py -i input.mp4 -o output.mp4 --no-audio
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-i, --input` | 输入视频文件或目录 | 必需 |
| `-o, --output` | 输出视频文件或目录 | 必需 |
| `-m, --model` | 模型名称 | RealESRGAN_x4plus |
| `-s, --scale` | 输出缩放比例 | 模型默认 |
| `--face-enhance` | 启用人脸增强 | False |
| `--fps` | 输出帧率 | 保持原始 |
| `--quality` | 视频质量 (0-100) | 90 |
| `--no-audio` | 不保留音频 | False |
| `--device` | 计算设备 (auto/cpu/cuda) | auto |
| `--tile` | 分块大小 | 0 |
| `--fp32` | 使用fp32精度 | False |
| `--batch` | 批量处理模式 | False |

## 🎯 支持的模型

| 模型名称 | 缩放倍数 | 适用场景 | 模型大小 |
|----------|----------|----------|----------|
| RealESRGAN_x4plus | 4x | 通用图像/视频 | ~65MB |
| RealESRGAN_x2plus | 2x | 通用图像/视频 | ~65MB |
| RealESRGAN_x4plus_anime_6B | 4x | 动漫插画 | ~17MB |
| realesr-animevideov3 | 4x | 动漫视频 | ~8MB |

## 📊 性能优化

### GPU内存优化

如果遇到GPU内存不足的问题，可以尝试以下方法：

1. **启用分块处理**：
   ```bash
   python video_super_resolution.py -i input.mp4 -o output.mp4 --tile 400
   ```

2. **使用半精度**（默认启用）：
   ```bash
   python video_super_resolution.py -i input.mp4 -o output.mp4
   ```

3. **使用CPU处理**：
   ```bash
   python video_super_resolution.py -i input.mp4 -o output.mp4 --device cpu
   ```

### 处理速度优化

1. **选择合适的模型**：动漫视频模型通常更小更快
2. **调整分块大小**：根据GPU内存调整tile参数
3. **使用GPU加速**：确保安装了CUDA版本的PyTorch

## 📁 项目结构

```
video-super-resolution/
├── video_super_resolution.py  # 主程序
├── requirements.txt           # 依赖文件
├── setup.py                  # 安装脚本
├── README.md                 # 说明文档
├── weights/                  # 模型文件目录
├── inputs/                   # 输入视频目录
└── outputs/                  # 输出视频目录
```

## 🛠️ 技术原理

本项目基于以下技术：

- **Real-ESRGAN**：腾讯ARC实验室开发的实用图像超分辨率算法
- **GFPGAN**：生成对抗网络用于人脸修复
- **BasicSR**：图像视频复原工具箱
- **PyTorch**：深度学习框架

### 算法流程

1. **视频解码**：使用OpenCV读取视频帧
2. **帧增强**：对每一帧应用超分辨率算法
3. **后处理**：调整输出尺寸和质量
4. **视频编码**：重新编码为视频文件
5. **音频合并**：使用FFmpeg合并音频轨道

## 🔧 故障排除

### 常见问题

1. **模型下载失败**
   - 手动下载模型文件到 `weights/` 目录
   - 检查网络连接

2. **CUDA内存不足**
   ```bash
   # 使用分块处理
   python video_super_resolution.py -i input.mp4 -o output.mp4 --tile 200
   ```

3. **FFmpeg未找到**
   - 安装FFmpeg并添加到系统PATH
   - 或使用 `--no-audio` 参数

4. **处理速度慢**
   - 确保使用GPU加速
   - 选择合适的模型
   - 调整分块大小

### 系统要求

- **最低要求**：
  - Python 3.7+
  - 4GB RAM
  - CPU处理

- **推荐配置**：
  - Python 3.8+
  - 8GB+ RAM
  - NVIDIA GPU (4GB+ VRAM)
  - CUDA 11.0+

## 📄 许可证

本项目采用MIT许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - 腾讯ARC实验室
- [BasicSR](https://github.com/XPixelGroup/BasicSR) - XPixel团队
- [GFPGAN](https://github.com/TencentARC/GFPGAN) - 腾讯ARC实验室

## 📞 支持

如果您遇到问题或有改进建议，请：

1. 查看FAQ部分
2. 搜索已有的Issues
3. 创建新的Issue描述问题

---

⭐ 如果这个项目对您有帮助，请给个Star支持一下！ 