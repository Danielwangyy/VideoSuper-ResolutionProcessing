#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速视频超分辨率测试程序
简化版本，用于快速验证功能
"""

import os
import cv2
import numpy as np
import argparse
import tempfile
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_upscale_test(input_path, output_path, scale=2):
    """
    简单的视频放大测试（不使用AI模型）
    """
    logger.info(f"开始简单放大测试: {input_path} -> {output_path}")
    
    # 打开输入视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {input_path}")
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
    
    # 输出视频设置
    new_width = int(width * scale)
    new_height = int(height * scale)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    # 处理每一帧
    with tqdm(total=total_frames, desc="处理视频帧") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 简单放大
            enhanced_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            out.write(enhanced_frame)
            pbar.update(1)
    
    # 清理
    cap.release()
    out.release()
    logger.info(f"简单放大完成: {output_path}")

def ai_upscale_test(input_path, output_path, device='cpu', max_frames=None):
    """
    AI超分辨率测试
    """
    logger.info(f"开始AI超分辨率测试: {input_path}")
    
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        
        # 设置模型
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = 'weights/RealESRGAN_x4plus.pth'
        
        # 确保模型目录和文件存在
        os.makedirs('weights', exist_ok=True)
        if not os.path.exists(model_path):
            logger.info("模型文件不存在，将自动下载...")
            import urllib.request
            url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
            urllib.request.urlretrieve(url, model_path)
            logger.info("模型下载完成")
        
        # 初始化
        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=400,  # 使用较大的分块
            tile_pad=10,
            pre_pad=0,
            half=False,  # 强制使用fp32
            device=device
        )
        
        # 处理视频
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 确定要处理的帧数
        frames_to_process = min(max_frames, total_frames) if max_frames else total_frames
        logger.info(f"将处理 {frames_to_process} / {total_frames} 帧")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width*4, height*4))
        
        with tqdm(total=frames_to_process, desc="AI处理进度") as pbar:
            for i in range(frames_to_process):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                try:
                    enhanced_frame, _ = upsampler.enhance(frame, outscale=4)
                    out.write(enhanced_frame)
                    pbar.set_description(f"AI处理进度 (第{i+1}帧)")
                except Exception as e:
                    logger.error(f"第 {i+1} 帧处理失败: {e}")
                    # fallback到简单放大
                    enhanced_frame = cv2.resize(frame, (width*4, height*4), interpolation=cv2.INTER_CUBIC)
                    out.write(enhanced_frame)
                
                pbar.update(1)
        
        cap.release()
        out.release()
        logger.info("AI超分辨率测试完成")
        
    except Exception as e:
        logger.error(f"AI超分辨率失败: {e}")
        logger.info("回退到简单放大方法")
        simple_upscale_test(input_path, output_path, scale=4)

def main():
    parser = argparse.ArgumentParser(description='快速视频超分辨率测试')
    parser.add_argument('-i', '--input', required=True, help='输入视频文件')
    parser.add_argument('-o', '--output', required=True, help='输出视频文件')
    parser.add_argument('--mode', choices=['simple', 'ai'], default='simple', 
                       help='测试模式: simple=简单放大, ai=AI超分辨率')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='计算设备')
    parser.add_argument('--scale', type=float, default=2, help='简单放大倍数')
    parser.add_argument('--max-frames', type=int, help='AI模式最大处理帧数（不指定则处理全部）')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        if args.mode == 'simple':
            simple_upscale_test(args.input, args.output, args.scale)
        else:
            ai_upscale_test(args.input, args.output, args.device, args.max_frames)
            
        logger.info("测试完成！")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 