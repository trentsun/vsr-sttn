import os
import torch
import whisper
import logging
from moviepy.editor import VideoFileClip
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TranscriptionTester:
    def __init__(self):
        logger.info("初始化转录测试...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {self.device}")
        
        # 加载Whisper模型
        logger.info("加载 Whisper 模型...")
        self.whisper_model = whisper.load_model("large").to(self.device)
        logger.info("Whisper 模型加载完成")

    def extract_audio(self, video_path):
        """从视频中提取音频"""
        logger.info(f"开始从视频提取音频: {video_path}")
        
        # 确保输出目录存在
        os.makedirs("temp", exist_ok=True)
        audio_path = "temp/temp_audio.wav"
        
        try:
            video = VideoFileClip(video_path)
            audio = video.audio
            logger.info("正在写入音频文件...")
            audio.write_audiofile(audio_path)
            video.close()
            
            logger.info(f"音频提取完成: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"音频提取失败: {str(e)}")
            raise

    def transcribe_audio(self, audio_path):
        """转录音频文件"""
        logger.info(f"开始转录音频: {audio_path}")
        
        try:
            # 使用Whisper进行转录
            result = self.whisper_model.transcribe(
                audio_path,
                language="pt",  # 指定源语言为葡萄牙语
                task="transcribe",
                temperature=0.2,
                best_of=5,
                beam_size=5,
                word_timestamps=True,
                condition_on_previous_text=True
            )
            
            # 获取转录片段
            segments = result["segments"]
            
            # 输出转录结果
            logger.info(f"转录完成，共 {len(segments)} 个片段")
            for i, segment in enumerate(segments):
                logger.info(f"片段 {i+1}:")
                logger.info(f"时间: {segment['start']:.2f}s - {segment['end']:.2f}s")
                logger.info(f"文本: {segment['text']}\n")
            
            return segments
            
        except Exception as e:
            logger.error(f"转录失败: {str(e)}")
            raise

def main():
    try:
        # 初始化测试器
        tester = TranscriptionTester()
        
        # 测试视频路径
        video_path = "video-translate/demo-teste.mp4"
        
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 1. 提取音频
        audio_path = tester.extract_audio(video_path)
        
        # 2. 转录音频
        segments = tester.transcribe_audio(audio_path)
        
        # 3. 保存转录结果到文件
        output_file = "transcription_results.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("转录结果:\n\n")
            for i, segment in enumerate(segments):
                f.write(f"片段 {i+1}:\n")
                f.write(f"时间: {segment['start']:.2f}s - {segment['end']:.2f}s\n")
                f.write(f"文本: {segment['text']}\n\n")
        
        logger.info(f"转录结果已保存到: {output_file}")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists("temp"):
            import shutil
            shutil.rmtree("temp")

if __name__ == "__main__":
    main()