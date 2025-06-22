#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU优化的视频超分辨率处理程序
基于Real-ESRGAN技术，支持Apple MPS和NVIDIA CUDA GPU加速
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

class GPUVideoSuperResolution:
    def __init__(self, model_name='RealESRGAN_x4plus', device='auto'):
        """
        初始化GPU优化的视频超分辨率处理器
        
        Args:
            model_name: 模型名称
            device: 设备类型 ('auto', 'cpu', 'cuda', 'mps')
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
        
        # 设置GPU优化环境
        self._setup_gpu_environment()
        self._check_dependencies()
        
    def _get_device(self, device):
        """获取计算设备 - 支持Apple MPS和CUDA GPU加速"""
        if device == 'auto':
            if torch.backends.mps.is_available():
                logger.info("🚀 检测到Apple Silicon，使用MPS GPU加速")
                return 'mps'
            elif torch.cuda.is_available():
                logger.info("🚀 检测到NVIDIA GPU，使用CUDA加速") 
                return 'cuda'
            else:
                logger.info("⚠️ 未检测到GPU，使用CPU处理")
                return 'cpu'
        
        # 验证手动指定的设备
        if device == 'mps':
            if torch.backends.mps.is_available():
                logger.info("🚀 手动指定MPS GPU加速")
                return 'mps'
            else:
                logger.warning("❌ MPS不可用，回退到CPU")
                return 'cpu'
        elif device == 'cuda':
            if torch.cuda.is_available():
                logger.info("🚀 手动指定CUDA GPU加速")
                return 'cuda'
            else:
                logger.warning("❌ CUDA不可用，回退到CPU")
                return 'cpu'
            
        return device
    
    def _setup_gpu_environment(self):
        """设置GPU优化环境变量"""
        if self.device == 'mps':
            # Apple MPS优化设置
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            logger.info("🔧 Apple MPS环境变量已优化")
            
        elif self.device == 'cuda':
            # CUDA优化设置
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            logger.info("🔧 CUDA环境变量已优化")
    
    def _check_dependencies(self):
        """检查依赖项和GPU支持"""
        try:
            import basicsr
            from realesrgan import RealESRGANer
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            logger.info("✅ 所有依赖项检查通过")
            
            # 详细的GPU信息
            self._log_gpu_info()
            
        except ImportError as e:
            logger.error(f"❌ 缺少必要的依赖项: {e}")
            logger.info("请运行: pip install basicsr realesrgan opencv-python torch torchvision")
            logger.warning("将使用简化处理模式")
    
    def _log_gpu_info(self):
        """记录GPU详细信息"""
        logger.info("=" * 50)
        logger.info("🖥️ 系统硬件信息:")
        logger.info(f"PyTorch版本: {torch.__version__}")
        
        if self.device == 'mps':
            logger.info(f"设备类型: Apple MPS")
            logger.info(f"MPS可用: {torch.backends.mps.is_available()}")
            logger.info(f"MPS已构建: {torch.backends.mps.is_built()}")
            
            # 测试MPS性能
            try:
                start_time = time.time()
                x = torch.randn(1000, 1000).to('mps')
                y = torch.matmul(x, x)
                mps_time = time.time() - start_time
                logger.info(f"MPS测试耗时: {mps_time:.3f}秒")
            except Exception as e:
                logger.warning(f"MPS测试失败: {e}")
                
        elif self.device == 'cuda':
            logger.info(f"设备类型: NVIDIA CUDA")
            logger.info(f"CUDA版本: {torch.version.cuda}")
            logger.info(f"GPU数量: {torch.cuda.device_count()}")
            if torch.cuda.is_available():
                logger.info(f"当前GPU: {torch.cuda.get_device_name()}")
                
        logger.info("=" * 50)
    
    def get_optimal_tile_size(self):
        """根据设备类型获取最优分块大小"""
        if self.device == 'mps':
            # Apple Silicon优化：较小的分块避免内存问题
            return 256
        elif self.device == 'cuda':
            # CUDA：可以使用更大的分块
            return 512
        else:
            # CPU：使用小分块
            return 128
    
    def setup_model(self, tile_size=None, tile_pad=10, pre_pad=0, fp32=None):
        """
        设置GPU优化的超分辨率模型
        
        Args:
            tile_size: 分块大小，None表示自动选择最优值
            tile_pad: 分块填充
            pre_pad: 预填充
            fp32: 是否使用fp32精度，None表示自动选择
        """
        try:
            from realesrgan import RealESRGANer
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            # 确保模型目录存在
            os.makedirs('weights', exist_ok=True)
            
            # GPU优化参数
            if tile_size is None:
                tile_size = self.get_optimal_tile_size()
                logger.info(f"🔧 自动选择最优分块大小: {tile_size}")
            
            if fp32 is None:
                # MPS建议使用fp32，CUDA可以使用fp16
                fp32 = (self.device == 'mps') or (self.device == 'cpu')
                logger.info(f"🔧 自动选择精度: {'fp32' if fp32 else 'fp16'}")
            
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
            
            # 如果模型文件不存在，下载
            if not os.path.exists(model_path):
                config = self.model_configs.get(self.model_name, {})
                logger.warning(f"模型文件不存在: {model_path}")
                self._download_model(config.get('url'), model_path)
            
            # 🚀 GPU优化的关键：正确初始化RealESRGANer
            logger.info(f"🚀 在{self.device.upper()}上初始化模型...")
            self.upsampler = RealESRGANer(
                scale=netscale,
                model_path=model_path,
                model=model,
                tile=tile_size,
                tile_pad=tile_pad,
                pre_pad=pre_pad,
                half=not fp32,  # GPU加速的关键参数
                device=self.device  # 🔥 这里决定GPU加速是否生效！
            )
            
            logger.info(f"✅ 模型 {self.model_name} 在 {self.device.upper()} 上加载成功")
            
            # 清理GPU缓存
            self._clear_gpu_cache()
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
    
    def _clear_gpu_cache(self):
        """清理GPU缓存"""
        if self.device == 'mps':
            torch.mps.empty_cache()
            logger.info("🧹 MPS缓存已清理")
        elif self.device == 'cuda':
            torch.cuda.empty_cache()
            logger.info("🧹 CUDA缓存已清理")
    
    def _download_model(self, url, save_path):
        """下载模型文件"""
        if not url:
            logger.error("模型下载URL不可用")
            return
            
        try:
            import urllib.request
            logger.info(f"📥 正在下载模型文件: {os.path.basename(save_path)}")
            
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(downloaded / total_size * 100, 100)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\r📥 下载进度: {percent:.1f}% ({mb_downloaded:.1f}MB/{mb_total:.1f}MB)", 
                          end='', flush=True)
            
            urllib.request.urlretrieve(url, save_path, reporthook=show_progress)
            print("\n✅ 模型下载完成!")
            
        except Exception as e:
            logger.error(f"❌ 模型下载失败: {e}")
            logger.info(f"请手动下载模型文件到: {save_path}")
            raise
    
    def enhance_frame(self, frame, outscale=None, face_enhance=False):
        """
        GPU加速的帧增强
        
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
            
            # 🚀 GPU加速处理在这里发生
            if face_enhance and self.face_enhancer:
                _, _, enhanced_frame = self.face_enhancer.enhance(
                    frame, has_aligned=False, only_center_face=False, paste_back=True
                )
            else:
                # RealESRGANer会自动在指定的GPU设备上处理
                enhanced_frame, _ = self.upsampler.enhance(
                    frame, 
                    outscale=outscale
                )
            
            return enhanced_frame
            
        except Exception as e:
            logger.error(f"❌ GPU帧增强失败: {e}")
            # Fallback到简单放大
            if outscale and outscale != 1:
                h, w = frame.shape[:2]
                new_h, new_w = int(h * outscale), int(w * outscale)
                return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            return frame
    
    def process_video(self, input_path, output_path, outscale=None, face_enhance=False, 
                     fps=None, quality=90, audio=True):
        """
        GPU加速的视频处理
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        logger.info(f"🎬 开始处理视频: {input_path}")
        logger.info(f"🚀 使用设备: {self.device.upper()}")
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
            
            # GPU加速处理视频帧
            self._process_video_frames_gpu(input_path, temp_video_path, outscale, face_enhance, fps, quality)
            
            # 合并音频
            if audio:
                self._merge_audio(input_path, temp_video_path, output_path)
            else:
                shutil.move(temp_video_path, output_path)
        
        logger.info(f"✅ 视频处理完成: {output_path}")
    
    def _process_video_frames_gpu(self, input_path, output_path, outscale, face_enhance, fps, quality):
        """GPU加速的视频帧处理"""
        # 打开输入视频
        cap = cv2.VideoCapture(input_path)
        
        # 获取视频信息
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"📹 原始视频: {original_width}x{original_height}, {original_fps:.1f}fps, {frame_count}帧")
        
        # 设置输出帧率
        if fps is None:
            fps = original_fps
        
        # 读取第一帧以确定输出尺寸
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("无法读取视频帧")
        
        # GPU处理第一帧
        start_time = time.time()
        enhanced_first_frame = self.enhance_frame(first_frame, outscale, face_enhance)
        first_frame_time = time.time() - start_time
        
        output_height, output_width = enhanced_first_frame.shape[:2]
        logger.info(f"📈 输出视频: {output_width}x{output_height}, {fps:.1f}fps")
        logger.info(f"⚡ 首帧处理时间: {first_frame_time:.2f}秒")
        
        # 重置视频读取位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        # GPU加速处理每一帧
        total_processing_time = 0
        with tqdm(total=frame_count, desc=f"🚀 GPU处理({self.device.upper()})") as pbar:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # GPU增强帧
                frame_start = time.time()
                enhanced_frame = self.enhance_frame(frame, outscale, face_enhance)
                frame_time = time.time() - frame_start
                total_processing_time += frame_time
                
                # 写入输出视频
                out.write(enhanced_frame)
                
                frame_idx += 1
                avg_time = total_processing_time / frame_idx
                pbar.set_postfix({
                    '当前帧': f'{frame_time:.2f}s',
                    '平均': f'{avg_time:.2f}s/帧',
                    '设备': self.device.upper()
                })
                pbar.update(1)
                
                # 每10帧清理一次GPU缓存
                if frame_idx % 10 == 0:
                    self._clear_gpu_cache()
        
        # 释放资源
        cap.release()
        out.release()
        
        # 最终统计
        avg_time_per_frame = total_processing_time / frame_count
        total_time = total_processing_time
        logger.info(f"📊 处理统计:")
        logger.info(f"   - 总帧数: {frame_count}")
        logger.info(f"   - 总耗时: {total_time:.1f}秒")
        logger.info(f"   - 平均: {avg_time_per_frame:.2f}秒/帧")
        logger.info(f"   - 设备: {self.device.upper()}")
    
    def _merge_audio(self, input_path, video_path, output_path):
        """合并音频到视频"""
        try:
            cmd = [
                'ffmpeg', '-y', '-loglevel', 'quiet',
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
                logger.warning(f"⚠️ 音频合并失败，使用无音频版本")
                shutil.move(video_path, output_path)
            else:
                logger.info("🎵 音频合并成功")
                
        except FileNotFoundError:
            logger.warning("⚠️ ffmpeg未找到，使用无音频版本")
            shutil.move(video_path, output_path)
        except Exception as e:
            logger.warning(f"⚠️ 音频合并出错: {e}")
            shutil.move(video_path, output_path)


def main():
    parser = argparse.ArgumentParser(description='GPU优化的视频超分辨率处理程序')
    
    # 输入输出参数
    parser.add_argument('-i', '--input', type=str, required=True, help='输入视频文件')
    parser.add_argument('-o', '--output', type=str, required=True, help='输出视频文件')
    
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
    
    # GPU参数
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda', 'mps'], help='计算设备')
    parser.add_argument('--tile', type=int, help='分块大小（自动选择最优值）')
    parser.add_argument('--fp32', action='store_true', help='强制使用fp32精度')
    
    args = parser.parse_args()
    
    try:
        # 初始化GPU优化处理器
        processor = GPUVideoSuperResolution(model_name=args.model, device=args.device)
        
        # 设置GPU优化模型
        processor.setup_model(
            tile_size=args.tile,
            fp32=args.fp32
        )
        
        # 处理参数
        process_kwargs = {
            'outscale': args.scale,
            'face_enhance': args.face_enhance,
            'fps': args.fps,
            'quality': args.quality,
            'audio': not args.no_audio
        }
        
        # 开始GPU加速处理
        processor.process_video(args.input, args.output, **process_kwargs)
        
        logger.info("🎉 GPU加速处理完成!")
        
    except Exception as e:
        logger.error(f"❌ 处理失败: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main()) 