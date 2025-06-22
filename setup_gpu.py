#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU加速环境配置脚本
检测系统环境并优化配置，支持Apple MPS和NVIDIA CUDA
"""

import sys
import os
import platform
import subprocess
import importlib
import time
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUSetup:
    def __init__(self):
        self.system_info = self._get_system_info()
        self.gpu_type = None
        self.recommendations = []
    
    def _get_system_info(self):
        """获取系统信息"""
        info = {
            'platform': platform.system(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
            'processor': platform.processor()
        }
        
        # 检测Apple Silicon
        if info['platform'] == 'Darwin' and info['machine'] == 'arm64':
            info['is_apple_silicon'] = True
        else:
            info['is_apple_silicon'] = False
            
        return info
    
    def check_pytorch_installation(self):
        """检查PyTorch安装情况"""
        logger.info("🔍 检查PyTorch安装情况...")
        
        try:
            import torch
            import torchvision
            
            logger.info(f"✅ PyTorch版本: {torch.__version__}")
            logger.info(f"✅ TorchVision版本: {torchvision.__version__}")
            
            # 检查编译版本信息
            logger.info(f"📊 PyTorch编译信息:")
            logger.info(f"   - CUDA编译支持: {torch.version.cuda is not None}")
            if torch.version.cuda:
                logger.info(f"   - CUDA版本: {torch.version.cuda}")
            
            return True
            
        except ImportError as e:
            logger.error(f"❌ PyTorch未安装: {e}")
            self.recommendations.append("安装PyTorch: pip install torch torchvision")
            return False
    
    def check_gpu_support(self):
        """检查GPU支持情况"""
        logger.info("🚀 检查GPU支持情况...")
        
        try:
            import torch
            
            # 检查CUDA支持
            cuda_available = torch.cuda.is_available()
            logger.info(f"NVIDIA CUDA可用: {cuda_available}")
            
            if cuda_available:
                self.gpu_type = 'cuda'
                logger.info(f"   - CUDA设备数量: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    logger.info(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
                    
                # CUDA性能测试
                self._test_cuda_performance()
            
            # 检查MPS支持（Apple Silicon）
            if hasattr(torch.backends, 'mps'):
                mps_available = torch.backends.mps.is_available()
                mps_built = torch.backends.mps.is_built()
                
                logger.info(f"Apple MPS可用: {mps_available}")
                logger.info(f"Apple MPS已构建: {mps_built}")
                
                if mps_available and not cuda_available:
                    self.gpu_type = 'mps'
                    # MPS性能测试
                    self._test_mps_performance()
            
            if not cuda_available and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                self.gpu_type = 'cpu'
                logger.warning("⚠️ 未检测到GPU加速支持，将使用CPU")
                
        except Exception as e:
            logger.error(f"❌ GPU检查失败: {e}")
            self.gpu_type = 'cpu'
    
    def _test_cuda_performance(self):
        """测试CUDA性能"""
        try:
            import torch
            
            logger.info("🧪 CUDA性能测试...")
            device = torch.device('cuda')
            
            # 矩阵乘法测试
            size = 2000
            start_time = time.time()
            
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            c = torch.matmul(a, b)
            torch.cuda.synchronize()  # 确保计算完成
            
            cuda_time = time.time() - start_time
            logger.info(f"   - CUDA矩阵乘法({size}x{size}): {cuda_time:.3f}秒")
            
            # 内存信息
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"   - GPU内存使用: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
            
        except Exception as e:
            logger.warning(f"⚠️ CUDA性能测试失败: {e}")
    
    def _test_mps_performance(self):
        """测试MPS性能"""
        try:
            import torch
            
            logger.info("🧪 MPS性能测试...")
            device = torch.device('mps')
            
            # 矩阵乘法测试
            size = 2000
            start_time = time.time()
            
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            c = torch.matmul(a, b)
            
            mps_time = time.time() - start_time
            logger.info(f"   - MPS矩阵乘法({size}x{size}): {mps_time:.3f}秒")
            
            # CPU对比测试
            start_time = time.time()
            a_cpu = torch.randn(size, size)
            b_cpu = torch.randn(size, size)
            c_cpu = torch.matmul(a_cpu, b_cpu)
            cpu_time = time.time() - start_time
            
            speedup = cpu_time / mps_time if mps_time > 0 else 1
            logger.info(f"   - CPU矩阵乘法({size}x{size}): {cpu_time:.3f}秒")
            logger.info(f"   - MPS加速比: {speedup:.1f}x")
            
        except Exception as e:
            logger.warning(f"⚠️ MPS性能测试失败: {e}")
    
    def check_realesrgan_compatibility(self):
        """检查Real-ESRGAN兼容性"""
        logger.info("🔍 检查Real-ESRGAN兼容性...")
        
        try:
            # 检查basicsr
            import basicsr
            logger.info(f"✅ BasicSR版本: {basicsr.__version__}")
            
            # 检查realesrgan
            import realesrgan
            logger.info(f"✅ RealESRGAN版本: {realesrgan.__version__}")
            
            # 尝试导入关键模块
            from realesrgan import RealESRGANer
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            logger.info("✅ Real-ESRGAN模块导入成功")
            
            # 检查known issues
            self._check_known_issues()
            
            return True
            
        except ImportError as e:
            logger.error(f"❌ Real-ESRGAN依赖缺失: {e}")
            self.recommendations.append("安装Real-ESRGAN: pip install realesrgan basicsr")
            return False
        except Exception as e:
            logger.error(f"❌ Real-ESRGAN兼容性问题: {e}")
            return False
    
    def _check_known_issues(self):
        """检查已知兼容性问题"""
        try:
            import torch
            import torchvision
            
            # 检查torchvision版本兼容性
            tv_version = torchvision.__version__
            major, minor = map(int, tv_version.split('.')[:2])
            
            if major > 0 or (major == 0 and minor >= 13):
                # 新版本可能有functional_tensor问题
                try:
                    from torchvision.transforms.functional_tensor import rgb_to_grayscale
                    logger.warning("⚠️ 检测到可能的functional_tensor兼容性问题")
                    self.recommendations.append("可能需要修复BasicSR中的functional_tensor导入问题")
                except ImportError:
                    # 这是预期的，新版本已经移除了functional_tensor模块
                    logger.info("✅ TorchVision版本兼容性检查通过")
                    
        except Exception as e:
            logger.warning(f"⚠️ 兼容性检查失败: {e}")
    
    def optimize_environment(self):
        """优化环境设置"""
        logger.info("🔧 优化环境设置...")
        
        if self.gpu_type == 'mps':
            # Apple MPS优化
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            logger.info("✅ Apple MPS环境变量已设置")
            
        elif self.gpu_type == 'cuda':
            # CUDA优化
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 异步执行
            logger.info("✅ CUDA环境变量已设置")
        
        # 通用优化
        os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
        logger.info(f"✅ OpenMP线程数设置为: {os.cpu_count()}")
    
    def generate_config_file(self):
        """生成配置文件"""
        config_content = f"""# GPU加速配置文件
# 自动生成于: {time.strftime('%Y-%m-%d %H:%M:%S')}

[system]
platform = {self.system_info['platform']}
machine = {self.system_info['machine']}
is_apple_silicon = {self.system_info['is_apple_silicon']}

[gpu]
type = {self.gpu_type}
recommended_device = {self.gpu_type}

[optimization]
"""
        
        if self.gpu_type == 'mps':
            config_content += """tile_size = 256
precision = fp32
batch_size = 1
"""
        elif self.gpu_type == 'cuda':
            config_content += """tile_size = 512
precision = fp16
batch_size = 2
"""
        else:
            config_content += """tile_size = 128
precision = fp32
batch_size = 1
"""
        
        with open('gpu_config.ini', 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        logger.info("✅ 配置文件已生成: gpu_config.ini")
    
    def print_summary(self):
        """打印总结报告"""
        logger.info("=" * 60)
        logger.info("📊 GPU加速环境配置总结")
        logger.info("=" * 60)
        
        # 系统信息
        logger.info("🖥️ 系统信息:")
        logger.info(f"   - 操作系统: {self.system_info['platform']}")
        logger.info(f"   - 架构: {self.system_info['machine']}")
        logger.info(f"   - Python版本: {self.system_info['python_version']}")
        if self.system_info['is_apple_silicon']:
            logger.info("   - Apple Silicon: ✅")
        
        # GPU信息
        logger.info(f"🚀 推荐GPU设备: {self.gpu_type.upper()}")
        
        if self.gpu_type == 'mps':
            logger.info("🍎 Apple MPS配置:")
            logger.info("   - 分块大小: 256")
            logger.info("   - 精度: FP32")
            logger.info("   - 预期加速比: 2-5x")
        elif self.gpu_type == 'cuda':
            logger.info("🔥 NVIDIA CUDA配置:")
            logger.info("   - 分块大小: 512")
            logger.info("   - 精度: FP16")
            logger.info("   - 预期加速比: 5-20x")
        else:
            logger.info("💻 CPU配置:")
            logger.info("   - 分块大小: 128")
            logger.info("   - 精度: FP32")
        
        # 推荐操作
        if self.recommendations:
            logger.info("💡 建议操作:")
            for rec in self.recommendations:
                logger.info(f"   - {rec}")
        
        # 使用示例
        logger.info("🎯 使用示例:")
        logger.info(f"   python3 video_super_resolution_gpu.py -i input.mp4 -o output.mp4 --device {self.gpu_type}")
        
        logger.info("=" * 60)


def main():
    logger.info("🚀 开始GPU加速环境配置...")
    
    setup = GPUSetup()
    
    # 步骤1: 检查PyTorch
    pytorch_ok = setup.check_pytorch_installation()
    
    # 步骤2: 检查GPU支持
    if pytorch_ok:
        setup.check_gpu_support()
    
    # 步骤3: 检查Real-ESRGAN兼容性
    if pytorch_ok:
        setup.check_realesrgan_compatibility()
    
    # 步骤4: 优化环境
    setup.optimize_environment()
    
    # 步骤5: 生成配置文件
    setup.generate_config_file()
    
    # 步骤6: 打印总结
    setup.print_summary()
    
    logger.info("✅ GPU加速环境配置完成!")
    
    return 0


if __name__ == '__main__':
    exit(main()) 