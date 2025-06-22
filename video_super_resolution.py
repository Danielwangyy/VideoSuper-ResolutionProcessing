#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频超分辨率处理程序
基于Real-ESRGAN技术实现视频清晰度提升
支持多种模型和自定义配置
"""

import os
import cv2
import argparse
import numpy as np
import torch
import subprocess
from pathlib import Path
import tempfile
import shutil
from tqdm import tqdm
import time
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoSuperResolution:
    def __init__(self, model_name='RealESRGAN_x4plus', device='auto'):
        """
        初始化视频超分辨率处理器
        
        Args:
            model_name: 模型名称
            device: 设备类型 ('auto', 'cpu', 'cuda')
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.upsampler = None
        self.face_enhancer = None
        
        # 支持的模型配置
        self.model_configs = {
            'RealESRGAN_x4plus': {
                'scale': 4,
                'description': '通用4倍超分辨率模型',
                'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
            },
            'RealESRGAN_x2plus': {
                'scale': 2,
                'description': '通用2倍超分辨率模型', 
                'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
            },
            'RealESRGAN_x4plus_anime_6B': {
                'scale': 4,
                'description': '动漫专用4倍超分辨率模型',
                'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
            },
            'realesr-animevideov3': {
                'scale': 4,
                'description': '动漫视频专用模型',
                'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth'
            }
        }
        
        self._check_dependencies()
        
    def _get_device(self, device):
        """获取计算设备"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _check_dependencies(self):
        """检查依赖项"""
        try:
            import basicsr
            from realesrgan import RealESRGANer
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            from basicsr.archs.rrdbnet_arch import RRDBNet
            # 检查torch和torchvision
            import torch
            import torchvision
            logger.info("所有依赖项检查通过")
        except ImportError as e:
            logger.error(f"缺少必要的依赖项: {e}")
            logger.info("请运行: pip install basicsr realesrgan opencv-python torch torchvision")
            # 不抛出异常，而是使用简化模式
            logger.warning("将使用简化处理模式")
    
    def setup_model(self, tile_size=0, tile_pad=10, pre_pad=0, fp32=False):
        """
        设置超分辨率模型
        
        Args:
            tile_size: 分块大小，0表示不分块
            tile_pad: 分块填充
            pre_pad: 预填充
            fp32: 是否使用fp32精度
        """
        try:
            from realesrgan import RealESRGANer
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            # 确保模型目录存在
            os.makedirs('weights', exist_ok=True)
            
            # 根据模型名称配置网络架构
            if self.model_name == 'RealESRGAN_x4plus':
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                netscale = 4
            elif self.model_name == 'RealESRGAN_x2plus':
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                netscale = 2
            elif self.model_name == 'RealESRGAN_x4plus_anime_6B':
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
                netscale = 4
            elif self.model_name == 'realesr-animevideov3':
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
                netscale = 4
            else:
                raise ValueError(f"不支持的模型: {self.model_name}")
            
            # 模型路径
            model_path = f'weights/{self.model_name}.pth'
            
            # 如果模型文件不存在，提供下载提示
            if not os.path.exists(model_path):
                config = self.model_configs.get(self.model_name, {})
                logger.warning(f"模型文件不存在: {model_path}")
                logger.info(f"请从以下链接下载模型文件: {config.get('url', 'N/A')}")
                logger.info(f"并将其保存到: {model_path}")
                
                # 创建一个简单的下载函数
                self._download_model(config.get('url'), model_path)
            
            # 初始化超分辨率器
            self.upsampler = RealESRGANer(
                scale=netscale,
                model_path=model_path,
                model=model,
                tile=tile_size if tile_size > 0 else 0,
                tile_pad=tile_pad,
                pre_pad=pre_pad,
                half=False,  # 强制使用fp32避免精度问题
                device=self.device
            )
            
            logger.info(f"模型 {self.model_name} 加载成功")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _download_model(self, url, save_path):
        """下载模型文件"""
        if not url:
            return
            
        try:
            import urllib.request
            logger.info(f"正在下载模型文件...")
            
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(downloaded / total_size * 100, 100)
                    print(f"\r下载进度: {percent:.1f}%", end='', flush=True)
            
            urllib.request.urlretrieve(url, save_path, reporthook=show_progress)
            print("\n模型下载完成!")
            
        except Exception as e:
            logger.error(f"模型下载失败: {e}")
            logger.info("请手动下载模型文件")
    
    def setup_face_enhancement(self):
        """设置人脸增强"""
        try:
            from gfpgan import GFPGANer
            
            self.face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=4,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=self.upsampler
            )
            
            logger.info("人脸增强模型加载成功")
            
        except Exception as e:
            logger.warning(f"人脸增强模型加载失败: {e}")
    
    def enhance_frame(self, frame, outscale=None, face_enhance=False):
        """
        增强单帧图像
        
        Args:
            frame: 输入帧
            outscale: 输出缩放比例
            face_enhance: 是否启用人脸增强
            
        Returns:
            enhanced_frame: 增强后的帧
        """
        try:
            # 确保输入是uint8格式
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            if face_enhance and self.face_enhancer:
                _, _, enhanced_frame = self.face_enhancer.enhance(
                    frame, has_aligned=False, only_center_face=False, paste_back=True
                )
            else:
                # 使用更稳定的参数
                enhanced_frame, _ = self.upsampler.enhance(
                    frame, 
                    outscale=outscale,
                    alpha_upsampler='realesrgan'
                )
            
            return enhanced_frame
            
        except Exception as e:
            logger.error(f"帧增强失败: {e}")
            # 返回简单放大的帧作为fallback
            if outscale and outscale != 1:
                h, w = frame.shape[:2]
                new_h, new_w = int(h * outscale), int(w * outscale)
                return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            return frame
    
    def process_video(self, input_path, output_path, outscale=None, face_enhance=False, 
                     fps=None, quality=90, audio=True):
        """
        处理视频文件
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            outscale: 输出缩放比例
            face_enhance: 是否启用人脸增强
            fps: 输出帧率
            quality: 视频质量 (0-100)
            audio: 是否保留音频
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
            
            # 处理视频帧
            self._process_video_frames(input_path, temp_video_path, outscale, face_enhance, fps, quality)
            
            # 如果需要保留音频，则合并音频
            if audio:
                self._merge_audio(input_path, temp_video_path, output_path)
            else:
                shutil.move(temp_video_path, output_path)
        
        logger.info(f"视频处理完成: {output_path}")
    
    def _process_video_frames(self, input_path, output_path, outscale, face_enhance, fps, quality):
        """处理视频帧"""
        # 打开输入视频
        cap = cv2.VideoCapture(input_path)
        
        # 获取视频信息
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 设置输出帧率
        if fps is None:
            fps = original_fps
        
        # 读取第一帧以确定输出尺寸
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("无法读取视频帧")
        
        # 处理第一帧以获取输出尺寸
        enhanced_first_frame = self.enhance_frame(first_frame, outscale, face_enhance)
        output_height, output_width = enhanced_first_frame.shape[:2]
        
        # 重置视频读取位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        # 处理每一帧
        with tqdm(total=frame_count, desc="处理视频帧") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 增强帧
                enhanced_frame = self.enhance_frame(frame, outscale, face_enhance)
                
                # 写入输出视频
                out.write(enhanced_frame)
                
                pbar.update(1)
        
        # 释放资源
        cap.release()
        out.release()
    
    def _merge_audio(self, input_path, video_path, output_path):
        """合并音频到视频"""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', input_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"音频合并失败，使用无音频版本: {result.stderr}")
                shutil.move(video_path, output_path)
                
        except FileNotFoundError:
            logger.warning("ffmpeg未找到，使用无音频版本")
            shutil.move(video_path, output_path)
        except Exception as e:
            logger.warning(f"音频合并出错: {e}")
            shutil.move(video_path, output_path)
    
    def batch_process(self, input_dir, output_dir, **kwargs):
        """批量处理视频文件"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 支持的视频格式
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
        
        # 查找所有视频文件
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_dir.glob(f'*{ext}'))
            video_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        if not video_files:
            logger.warning(f"在 {input_dir} 中未找到视频文件")
            return
        
        logger.info(f"找到 {len(video_files)} 个视频文件")
        
        # 处理每个视频文件
        for video_file in video_files:
            try:
                output_file = output_dir / f"{video_file.stem}_enhanced{video_file.suffix}"
                logger.info(f"处理: {video_file.name}")
                
                self.process_video(str(video_file), str(output_file), **kwargs)
                
            except Exception as e:
                logger.error(f"处理文件 {video_file.name} 时出错: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(description='视频超分辨率处理程序')
    
    # 输入输出参数
    parser.add_argument('-i', '--input', type=str, required=True, help='输入视频文件或目录')
    parser.add_argument('-o', '--output', type=str, required=True, help='输出视频文件或目录')
    
    # 模型参数
    parser.add_argument('-m', '--model', type=str, default='RealESRGAN_x4plus',
                       choices=['RealESRGAN_x4plus', 'RealESRGAN_x2plus', 
                               'RealESRGAN_x4plus_anime_6B', 'realesr-animevideov3'],
                       help='超分辨率模型')
    
    # 处理参数
    parser.add_argument('-s', '--scale', type=float, help='输出缩放比例')
    parser.add_argument('--face-enhance', action='store_true', help='启用人脸增强')
    parser.add_argument('--fps', type=float, help='输出帧率')
    parser.add_argument('--quality', type=int, default=90, help='视频质量 (0-100)')
    parser.add_argument('--no-audio', action='store_true', help='不保留音频')
    
    # 性能参数
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda'], help='计算设备')
    parser.add_argument('--tile', type=int, default=0, help='分块大小')
    parser.add_argument('--tile-pad', type=int, default=10, help='分块填充')
    parser.add_argument('--fp32', action='store_true', help='使用fp32精度')
    
    # 批处理参数
    parser.add_argument('--batch', action='store_true', help='批量处理模式')
    
    args = parser.parse_args()
    
    try:
        # 初始化处理器
        processor = VideoSuperResolution(model_name=args.model, device=args.device)
        
        # 设置模型
        processor.setup_model(
            tile_size=args.tile,
            tile_pad=args.tile_pad,
            fp32=args.fp32
        )
        
        # 设置人脸增强
        if args.face_enhance:
            processor.setup_face_enhancement()
        
        # 处理参数
        process_kwargs = {
            'outscale': args.scale,
            'face_enhance': args.face_enhance,
            'fps': args.fps,
            'quality': args.quality,
            'audio': not args.no_audio
        }
        
        # 开始处理
        if args.batch or os.path.isdir(args.input):
            processor.batch_process(args.input, args.output, **process_kwargs)
        else:
            processor.process_video(args.input, args.output, **process_kwargs)
            
        logger.info("处理完成!")
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main()) 