import os
import torch
import whisper
import logging
from moviepy.editor import VideoFileClip
from pathlib import Path
import demucs.separate
from pydub import AudioSegment

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
        # 对于中文，建议使用medium或large模型
        # - base: 快速但准确度较低
        # - medium: 平衡速度和准确度
        # - large-v2: 最高准确度但速度较慢
        logger.info("加载 Whisper 模型...")
        self.model_size = "large-v2"  # 可以根据需要调整为 "medium" 或 "base"
        self.whisper_model = whisper.load_model(self.model_size).to(self.device)
        logger.info(f"Whisper {self.model_size}模型加载完成")

    def extract_audio(self, video_path):
        """从视频中提取音频"""
        logger.info(f"开始从视频提取音频: {video_path}")
        
        os.makedirs("temp", exist_ok=True)
        audio_path = "temp/temp_audio.wav"
        
        try:
            video = VideoFileClip(video_path)
            audio = video.audio
            logger.info("正在写入音频文件...")
            # 设置采样率为16kHz，这是Whisper模型推荐的采样率
            audio.write_audiofile(audio_path, fps=16000)
            video.close()
            
            logger.info(f"音频提取完成: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"音频提取失败: {str(e)}")
            raise

    def separate_vocals(self, audio_path):
        """使用 demucs 进行人声分离"""
        logger.info(f"开始进行人声分离: {audio_path}")
        
        try:
            output_dir = Path("separated")
            output_dir.mkdir(exist_ok=True)
            
            # 使用 MDX-Net 模型，它对中文语音分离效果较好
            args = [
                str(audio_path),
                "-n", "mdx_extra",  # 使用 MDX-Net 模型
                "--two-stems", "vocals",
                "--shifts", "2",
                "--segment", "7",
                "--device", self.device,
                "--overlap", "0.25",
                "--jobs", "2",
                "--out", str(output_dir)
            ]
            
            logger.info("执行人声分离...")
            demucs.separate.main(args)
            
            track_name = Path(audio_path).stem
            vocals_path = output_dir / "mdx_extra" / track_name / "vocals.wav"
            
            if not vocals_path.exists():
                raise FileNotFoundError(f"人声文件未找到: {vocals_path}")
            
            # 音频标准化处理
            logger.info("正在标准化音频...")
            audio = AudioSegment.from_wav(str(vocals_path))
            
            # 标准化音量
            target_dBFS = -20.0
            change_in_dBFS = target_dBFS - audio.dBFS
            audio = audio.apply_gain(change_in_dBFS)
            
            processed_path = "temp/processed_vocals.wav"
            audio.export(processed_path, format="wav")
            
            logger.info(f"人声分离完成，输出文件: {processed_path}")
            return processed_path
            
        except Exception as e:
            logger.error(f"人声分离失败: {str(e)}")
            logger.info("使用原始音频继续处理...")
            return audio_path

    def transcribe_audio(self, audio_path):
        """转录音频文件"""
        logger.info(f"开始转录音频: {audio_path}")
        
        try:
            # 针对中文的Whisper配置
            result = self.whisper_model.transcribe(
                audio_path,
                language="zh",  # 指定语言为中文
                task="transcribe",
                temperature=0.2,  # 降低随机性
                best_of=5,  # 生成多个候选并选择最佳
                beam_size=5,  # 使用波束搜索
                word_timestamps=True,  # 获取词级时间戳
                condition_on_previous_text=True,  # 考虑上下文
                initial_prompt="这是一段中文对话"  # 提供中文提示
            )
            
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

    def cleanup(self):
        """清理临时文件"""
        logger.info("清理临时文件...")
        dirs_to_clean = ["temp", "separated"]
        for dir_name in dirs_to_clean:
            if os.path.exists(dir_name):
                import shutil
                shutil.rmtree(dir_name)
        logger.info("清理完成")

def main():
    tester = None
    try:
        tester = TranscriptionTester()
        
        video_path = "video-translate/demo-teste.mp4"
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 1. 提取音频
        audio_path = tester.extract_audio(video_path)
        
        # 2. 人声分离
        vocals_path = tester.separate_vocals(audio_path)
        
        # 3. 转录音频
        segments = tester.transcribe_audio(vocals_path)
        
        # 4. 保存转录结果
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
        if tester:
            tester.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()