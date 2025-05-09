import logging

def test_dependencies():
    """测试所需依赖是否正确安装"""
    dependencies = {
        'torch': 'PyTorch',
        'torchaudio': 'TorchAudio',
        'demucs': 'Demucs',
        'numpy': 'NumPy',
        'pathlib': 'PathLib',
        'logging': 'Logging'
    }
    
    missing_deps = []
    installed_versions = {}
    
    for module, name in dependencies.items():
        try:
            imported_module = __import__(module)
            version = getattr(imported_module, '__version__', 'Unknown version')
            installed_versions[name] = version
        except ImportError:
            missing_deps.append(name)
    
    return installed_versions, missing_deps

def test_cuda_availability():
    """测试 CUDA 是否可用"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            return True, {
                "device_count": device_count,
                "device_name": device_name,
                "cuda_version": torch.version.cuda
            }
        return False, None
    except Exception as e:
        return False, str(e)

def create_test_audio():
    """创建测试用的音频文件"""
    try:
        import numpy as np
        import soundfile as sf
        
        # 创建一个简单的音频信号（3秒，44.1kHz采样率）
        sample_rate = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # 生成一个包含人声频率范围的信号
        voice_freq = 200  # 典型人声基频
        audio_signal = np.sin(2 * np.pi * voice_freq * t)
        
        # 添加一些噪声
        noise = np.random.normal(0, 0.1, len(t))
        audio_signal = audio_signal + noise
        
        # 标准化音频
        audio_signal = audio_signal / np.max(np.abs(audio_signal))
        
        # 保存测试音频文件
        test_file = "test_audio.wav"
        sf.write(test_file, audio_signal, sample_rate)
        
        return True, test_file
    except Exception as e:
        return False, str(e)

def test_audio_separation(audio_file):
    """测试音频分离功能"""
    try:
        from pathlib import Path
        import time
        
        start_time = time.time()
        
        # 初始化 VideoTranslator
        translator = VideoTranslator()
        
        # 执行音频分离
        result_path = translator.separate_audio(audio_file)
        
        duration = time.time() - start_time
        
        # 检查结果
        result = {
            "success": Path(result_path).exists(),
            "input_path": audio_file,
            "output_path": result_path,
            "duration": f"{duration:.2f} seconds"
        }
        
        return True, result
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("开始依赖检查和功能测试...")
    
    # 1. 检查依赖
    logger.info("\n=== 检查依赖 ===")
    installed_versions, missing_deps = test_dependencies()
    
    if missing_deps:
        logger.error(f"缺少以下依赖: {', '.join(missing_deps)}")
        logger.error("请使用 pip install 安装缺少的依赖")
    else:
        logger.info("已安装的依赖版本:")
        for name, version in installed_versions.items():
            logger.info(f"- {name}: {version}")
    
    # 2. 检查 CUDA
    logger.info("\n=== 检查 CUDA ===")
    cuda_available, cuda_info = test_cuda_availability()
    if cuda_available:
        logger.info("CUDA 可用:")
        for key, value in cuda_info.items():
            logger.info(f"- {key}: {value}")
    else:
        logger.warning(f"CUDA 不可用: {cuda_info}")
    
    # 3. 创建测试音频
    logger.info("\n=== 创建测试音频 ===")
    audio_created, audio_result = create_test_audio()
    if audio_created:
        logger.info(f"测试音频创建成功: {audio_result}")
    else:
        logger.error(f"创建测试音频失败: {audio_result}")
        sys.exit(1)
    
    # 4. 测试音频分离
    logger.info("\n=== 测试音频分离 ===")
    separation_success, separation_result = test_audio_separation(audio_result)
    if separation_success:
        logger.info("音频分离测试结果:")
        for key, value in separation_result.items():
            logger.info(f"- {key}: {value}")
    else:
        logger.error(f"音频分离测试失败: {separation_result}")
    
    # 5. 清理测试文件
    logger.info("\n=== 清理测试文件 ===")
    try:
        import os
        if os.path.exists(audio_result):
            os.remove(audio_result)
        if separation_success:
            output_path = separation_result.get("output_path")
            if output_path and os.path.exists(output_path):
                os.remove(output_path)
        logger.info("测试文件清理完成")
    except Exception as e:
        logger.warning(f"清理测试文件时出错: {e}")
    
    logger.info("\n=== 测试完成 ===")