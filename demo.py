#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频超分辨率演示脚本
展示如何使用主程序进行图像和视频处理
"""

import os
import cv2
import numpy as np
from video_super_resolution import VideoSuperResolution
import argparse

def create_test_video(output_path="test_video.mp4", duration=5, fps=30, resolution=(320, 240)):
    """
    创建一个测试视频文件
    
    Args:
        output_path: 输出视频路径
        duration: 视频时长（秒）
        fps: 帧率
        resolution: 分辨率 (width, height)
    """
    print(f"创建测试视频: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    
    total_frames = duration * fps
    
    for i in range(total_frames):
        # 创建彩色渐变背景
        frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        
        # 添加时间变化的颜色
        r = int(127 + 127 * np.sin(2 * np.pi * i / total_frames))
        g = int(127 + 127 * np.sin(2 * np.pi * i / total_frames + np.pi/3))
        b = int(127 + 127 * np.sin(2 * np.pi * i / total_frames + 2*np.pi/3))
        
        frame[:, :] = [b, g, r]
        
        # 添加移动的圆形
        center_x = int(resolution[0] * (0.2 + 0.6 * (i / total_frames)))
        center_y = int(resolution[1] * 0.5)
        cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
        
        # 添加文字
        text = f"Frame {i+1}/{total_frames}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    print(f"测试视频创建完成: {output_path}")

def demo_image_enhancement():
    """演示图像增强功能"""
    print("\n🖼️  演示图像增强...")
    
    # 创建一个测试图像
    test_image = np.zeros((240, 320, 3), dtype=np.uint8)
    
    # 添加渐变背景
    for y in range(240):
        for x in range(320):
            test_image[y, x] = [
                int(255 * x / 320),
                int(255 * y / 240),
                int(255 * (x + y) / (320 + 240))
            ]
    
    # 添加一些图案
    cv2.circle(test_image, (160, 120), 50, (255, 255, 255), -1)
    cv2.rectangle(test_image, (50, 50), (150, 150), (0, 255, 0), 3)
    cv2.putText(test_image, "TEST", (180, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # 保存测试图像
    os.makedirs("inputs", exist_ok=True)
    test_image_path = "inputs/test_image.jpg"
    cv2.imwrite(test_image_path, test_image)
    print(f"创建测试图像: {test_image_path}")
    
    try:
        # 初始化处理器
        processor = VideoSuperResolution(model_name='RealESRGAN_x4plus', device='cpu')
        
        # 注意：这里只是演示代码结构，实际运行需要模型文件
        print("✅ 处理器初始化成功")
        print("⚠️  注意：实际运行需要下载模型文件")
        
    except Exception as e:
        print(f"⚠️  处理器初始化失败: {e}")
        print("这是正常的，因为演示环境中没有安装所有依赖")

def demo_video_processing():
    """演示视频处理功能"""
    print("\n🎥 演示视频处理...")
    
    # 创建测试视频
    os.makedirs("inputs", exist_ok=True)
    test_video_path = "inputs/test_video.mp4"
    create_test_video(test_video_path, duration=3, fps=24, resolution=(240, 180))
    
    print("\n📋 视频处理示例命令：")
    print("python video_super_resolution.py -i inputs/test_video.mp4 -o outputs/enhanced_video.mp4")
    print("python video_super_resolution.py -i inputs/test_video.mp4 -o outputs/enhanced_video.mp4 -m RealESRGAN_x2plus")
    print("python video_super_resolution.py -i inputs/test_video.mp4 -o outputs/enhanced_video.mp4 --face-enhance")

def demo_batch_processing():
    """演示批量处理功能"""
    print("\n📁 演示批量处理...")
    
    os.makedirs("inputs", exist_ok=True)
    
    # 创建多个测试视频
    test_videos = [
        ("inputs/video1.mp4", (160, 120)),
        ("inputs/video2.mp4", (240, 180)),
        ("inputs/video3.mp4", (320, 240))
    ]
    
    for video_path, resolution in test_videos:
        create_test_video(video_path, duration=2, fps=20, resolution=resolution)
    
    print("\n📋 批量处理示例命令：")
    print("python video_super_resolution.py -i inputs/ -o outputs/ --batch")
    print("python video_super_resolution.py -i inputs/ -o outputs/ --batch -m realesr-animevideov3")

def show_model_info():
    """显示模型信息"""
    print("\n🎯 支持的模型:")
    
    models = {
        'RealESRGAN_x4plus': {
            'scale': '4x',
            'size': '~65MB',
            'description': '通用4倍超分辨率模型，适用于各种图像和视频',
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        },
        'RealESRGAN_x2plus': {
            'scale': '2x', 
            'size': '~65MB',
            'description': '通用2倍超分辨率模型，处理速度更快',
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
        },
        'RealESRGAN_x4plus_anime_6B': {
            'scale': '4x',
            'size': '~17MB', 
            'description': '动漫插画专用模型，体积小速度快',
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
        },
        'realesr-animevideov3': {
            'scale': '4x',
            'size': '~8MB',
            'description': '动漫视频专用模型，最小体积最快速度',
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth'
        }
    }
    
    for model_name, info in models.items():
        print(f"\n📦 {model_name}")
        print(f"   缩放倍数: {info['scale']}")
        print(f"   模型大小: {info['size']}")
        print(f"   描述: {info['description']}")
        print(f"   下载链接: {info['url']}")

def show_usage_examples():
    """显示使用示例"""
    print("\n💡 使用示例:")
    
    examples = [
        {
            'title': '基本视频增强',
            'command': 'python video_super_resolution.py -i input.mp4 -o output.mp4',
            'description': '使用默认模型将视频放大4倍'
        },
        {
            'title': '动漫视频增强',
            'command': 'python video_super_resolution.py -i anime.mp4 -o enhanced_anime.mp4 -m realesr-animevideov3',
            'description': '使用动漫专用模型处理动漫视频'
        },
        {
            'title': '人脸视频增强',
            'command': 'python video_super_resolution.py -i face_video.mp4 -o enhanced_face.mp4 --face-enhance',
            'description': '启用人脸增强功能'
        },
        {
            'title': '自定义缩放',
            'command': 'python video_super_resolution.py -i input.mp4 -o output.mp4 -s 2.5',
            'description': '自定义2.5倍缩放'
        },
        {
            'title': '批量处理',
            'command': 'python video_super_resolution.py -i videos/ -o enhanced/ --batch',
            'description': '批量处理整个目录的视频'
        },
        {
            'title': 'CPU模式',
            'command': 'python video_super_resolution.py -i input.mp4 -o output.mp4 --device cpu',
            'description': '使用CPU进行处理（GPU内存不足时）'
        },
        {
            'title': '分块处理',
            'command': 'python video_super_resolution.py -i input.mp4 -o output.mp4 --tile 400',
            'description': '启用分块处理以节省GPU内存'
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print(f"   命令: {example['command']}")
        print(f"   说明: {example['description']}")

def show_installation_guide():
    """显示安装指南"""
    print("\n🔧 安装指南:")
    print("\n1. 自动安装（推荐）:")
    print("   chmod +x install.sh")
    print("   ./install.sh")
    
    print("\n2. 手动安装:")
    print("   pip install -r requirements.txt")
    
    print("\n3. 下载模型文件:")
    print("   将模型文件放置在 weights/ 目录下")
    
    print("\n4. 安装FFmpeg:")
    print("   - Linux: sudo apt install ffmpeg")
    print("   - macOS: brew install ffmpeg") 
    print("   - Windows: 下载并添加到PATH")

def main():
    parser = argparse.ArgumentParser(description='视频超分辨率演示程序')
    parser.add_argument('--demo', choices=['image', 'video', 'batch', 'all'], 
                       default='all', help='演示类型')
    parser.add_argument('--info', action='store_true', help='显示模型信息')
    parser.add_argument('--examples', action='store_true', help='显示使用示例')
    parser.add_argument('--install', action='store_true', help='显示安装指南')
    
    args = parser.parse_args()
    
    print("🚀 视频超分辨率处理程序演示")
    print("=" * 50)
    
    if args.info:
        show_model_info()
    
    if args.examples:
        show_usage_examples()
    
    if args.install:
        show_installation_guide()
    
    if args.demo == 'all':
        demo_image_enhancement()
        demo_video_processing()
        demo_batch_processing()
    elif args.demo == 'image':
        demo_image_enhancement()
    elif args.demo == 'video':
        demo_video_processing()
    elif args.demo == 'batch':
        demo_batch_processing()
    
    print("\n" + "=" * 50)
    print("📖 更多信息请查看 README.md")
    print("🔗 GitHub项目: https://github.com/xinntao/Real-ESRGAN")
    print("📞 如有问题请查看文档或提交Issue")

if __name__ == '__main__':
    main() 