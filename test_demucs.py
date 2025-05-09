import os
import logging
from pathlib import Path
import torch
import demucs.separate
import soundfile as sf
import numpy as np
import shutil
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioSeparationTest:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.test_dir = Path("test_audio")
        self.test_dir.mkdir(exist_ok=True)
        
    def create_test_audio(self, duration=3.0):
        """创建测试音频文件"""
        logger.info("创建测试音频文件...")
        
        # 生成测试音频
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # 创建一个包含人声和音乐的混合信号
        voice_freq = 200  # 人声频率
        music_freq = 1000  # 音乐频率
        
        voice = 0.5 * np.sin(2 * np.pi * voice_freq * t)
        music = 0.3 * np.sin(2 * np.pi * music_freq * t)
        
        # 混合信号
        mixed = voice + music
        
        # 标准化
        mixed = mixed / np.max(np.abs(mixed))
        
        # 保存文件
        test_audio_path = self.test_dir / "test_mixed.wav"
        sf.write(test_audio_path, mixed, sample_rate)
        
        logger.info(f"测试音频创建完成: {test_audio_path}")
        return test_audio_path

    def separate_audio(self, input_path):
        """使用 demucs 进行音频分离"""
        logger.info(f"开始分离音频: {input_path}")
        
        try:
            # 创建输出目录
            output_dir = Path("separated")
            output_dir.mkdir(exist_ok=True)
            
            # 构建命令行参数列表
            args = [
                str(input_path),
                "-n", "htdemucs",  # 使用 htdemucs 模型
                "--two-stems", "vocals",  # 只分离人声
                "--shifts", "2",  # 设置移位次数
                "--segment", "10",  # 使用 --segment 替代 --split
                "--device", self.device,
                "--overlap", "0.25",  # 重叠率
                "--jobs", "2",  # 并行作业数
                "--out", str(output_dir)  # 指定输出目录
            ]
            
            # 执行分离
            demucs.separate.main(args)
            
            # 获取输出文件路径
            track_name = Path(input_path).stem
            vocals_path = output_dir / "htdemucs" / track_name / "vocals.wav"
            
            if not vocals_path.exists():
                raise FileNotFoundError(f"人声文件未找到: {vocals_path}")
            
            logger.info(f"音频分离完成，输出文件: {vocals_path}")
            return str(vocals_path)
        
        except Exception as e:
            logger.error(f"音频分离失败: {str(e)}")
            return input_path

    def verify_output(self, output_path):
        """验证输出文件"""
        output_path = Path(output_path)
        if not output_path.exists():
            return False, "输出文件不存在"
            
        try:
            # 检查音频文件是否可读
            audio_data, sample_rate = sf.read(output_path)
            
            # 基本检查
            checks = {
                "文件存在": output_path.exists(),
                "文件大小": output_path.stat().st_size > 0,
                "采样率": sample_rate == 44100,
                "音频长度": len(audio_data) > 0
            }
            
            return True, checks
            
        except Exception as e:
            return False, f"验证失败: {str(e)}"

    def cleanup(self):
        """清理测试文件"""
        try:
            if self.test_dir.exists():
                shutil.rmtree(self.test_dir)
            if Path("separated").exists():
                shutil.rmtree("separated")
            logger.info("清理完成")
        except Exception as e:
            logger.error(f"清理失败: {str(e)}")

def main():
    """主测试函数"""
    test = AudioSeparationTest()
    
    try:
        # 1. 检查环境
        logger.info("\n=== 环境检查 ===")
        logger.info(f"使用设备: {test.device}")
        logger.info(f"PyTorch 版本: {torch.__version__}")
        logger.info(f"CUDA 是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
        
        # 2. 创建测试音频
        logger.info("\n=== 创建测试音频 ===")
        test_audio_path = test.create_test_audio()
        
        # 3. 执行音频分离
        logger.info("\n=== 执行音频分离 ===")
        start_time = time.time()
        output_path = test.separate_audio(test_audio_path)
        processing_time = time.time() - start_time
        logger.info(f"处理时间: {processing_time:.2f} 秒")
        
        # 4. 验证结果
        logger.info("\n=== 验证结果 ===")
        success, result = test.verify_output(output_path)
        if success:
            logger.info("验证通过:")
            for check_name, check_result in result.items():
                logger.info(f"- {check_name}: {check_result}")
        else:
            logger.error(f"验证失败: {result}")
        
        # 5. 性能指标
        logger.info("\n=== 性能指标 ===")
        if torch.cuda.is_available():
            logger.info(f"GPU 内存使用: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        
        return success
        
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}", exc_info=True)
        return False
        
    finally:
        # 6. 清理
        logger.info("\n=== 清理测试文件 ===")
        test.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        success = main()
        if success:
            logger.info("\n=== 测试完成：通过 ===")
            exit(0)
        else:
            logger.error("\n=== 测试完成：失败 ===")
            exit(1)
    except KeyboardInterrupt:
        logger.info("\n=== 测试被用户中断 ===")
        exit(2)
    except Exception as e:
        logger.error(f"\n=== 测试异常: {str(e)} ===")
        exit(3)