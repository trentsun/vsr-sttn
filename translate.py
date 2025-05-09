import os
import time
import glob  # 用于文件模式匹配
import sys   # 用于系统相关操作
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
from transformers import pipeline
import torch
import torchaudio
from TTS.api import TTS
import whisper
import googletrans
from pydub import AudioSegment
import numpy as np
from torch.serialization import add_safe_globals

# 导入必要的配置类
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.configs.xtts_config import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.models.xtts import XttsArgs  # 这个之前漏掉了
from tqdm import tqdm  # 用于显示进度条
from TTS.utils.audio import AudioProcessor

import subprocess
import torch.hub
from pathlib import Path


import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 保存原始的torch.load函数
original_torch_load = torch.load

# 添加安全全局类
add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    BaseDatasetConfig,
    BaseTTSConfig,
    AudioProcessor,
    XttsArgs,
    # 基本数据类型
    dict, 
    list,
    tuple,
    str,
    int,
    float,
    bool,
    type(None)
])

class VideoTranslator:
    def __init__(self):
        logger.info("初始化 VideoTranslator...")
        
        # 添加设备检测
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {self.device}")
        
        from demucs.pretrained import get_model
        from demucs.apply import apply_model


        
        def separate_audio(self, input_path):
            """使用 demucs 进行音频分离"""
            logger.info(f"开始分离音频: {input_path}")
            
            try:
                # 创建输出目录
                output_dir = Path("separated")
                output_dir.mkdir(exist_ok=True)
                
                # 加载模型
                model = get_model('htdemucs')
                model.to(self.device)
                
                # 加载音频
                wav, sr = torchaudio.load(input_path)
                wav = wav.to(self.device)
                
                # 如果需要，重采样到模型所需的采样率
                if sr != model.samplerate:
                    wav = torchaudio.transforms.Resample(sr, model.samplerate)(wav)
                
                # 应用模型
                ref = wav.mean(0)
                wav = (wav - ref.mean()) / ref.std()
                sources = apply_model(model, wav.unsqueeze(0), shifts=2, split=True, overlap=0.25, progress=True)[0]
                
                # 获取人声部分
                vocals = sources[model.sources.index('vocals')]
                
                # 还原音量
                vocals = vocals * ref.std() + ref.mean()
                
                # 保存结果
                track_name = Path(input_path).stem
                vocals_path = output_dir / "htdemucs" / track_name / "vocals.wav"
                vocals_path.parent.mkdir(parents=True, exist_ok=True)
                
                torchaudio.save(
                    str(vocals_path),
                    vocals.cpu(),
                    model.samplerate
                )
                
                logger.info(f"音频分离完成，输出文件: {vocals_path}")
                return str(vocals_path)
                
            except Exception as e:
                logger.error(f"音频分离失败: {str(e)}")
                return input_path
            finally:
                # 清理 GPU 内存
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        
        self.separate_audio = separate_audio

        try:

            logger.info("加载 Whisper 模型...")
            self.whisper_model = whisper.load_model("large").to(self.device)
            logger.info("Whisper 模型加载完成")
            
            logger.info("初始化 Translator...")
            self.translator = googletrans.Translator()
            logger.info("Translator 初始化完成")
            
            logger.info("初始化 TTS 模型...")
            try:
                # 第一次尝试加载模型
                self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                if self.device == "cuda":
                    self.tts.to(self.device)
            except Exception as e:
                logger.warning(f"首次加载失败，尝试使用 weights_only=False: {str(e)}")
                # 使用原始的torch.load，但添加weights_only=False参数
                def load_with_weights(*args, **kwargs):
                    kwargs['weights_only'] = False
                    kwargs['map_location'] = torch.device(self.device)
                    return original_torch_load(*args, **kwargs)
                
                # 临时替换torch.load
                torch.load = load_with_weights
                # 再次尝试加载模型
                self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                if self.device == "cuda":
                    self.tts.to(self.device)
                # 恢复原始的torch.load
                torch.load = original_torch_load
                
            logger.info("TTS 模型加载成功")
            
        except Exception as e:
            logger.error(f"初始化失败: {str(e)}", exc_info=True)
            # 确保恢复原始的torch.load
            torch.load = original_torch_load
            raise

    def cleanup_temp_files(self):
        """清理所有临时文件和目录"""
        try:
            logger.info("开始清理临时文件...")
            
            # 需要清理的临时文件模式
            temp_patterns = [
                "temp_audio*.wav",
                "processed_audio.wav",
                "final_audio.wav",
                "*.mp3",
                "*.wav"
            ]
            
            # 需要清理的临时目录
            temp_dirs = [
                "separated",
                "demucs_quantized",
                "__pycache__"
            ]
            
            # 清理临时文件
            files_removed = 0
            for pattern in temp_patterns:
                for file_path in glob.glob(pattern):
                    try:
                        os.remove(file_path)
                        files_removed += 1
                        logger.debug(f"已删除文件: {file_path}")
                    except Exception as e:
                        logger.warning(f"删除文件 {file_path} 失败: {str(e)}")
            
            # 清理临时目录
            dirs_removed = 0
            for dir_name in temp_dirs:
                dir_path = Path(dir_name)
                if dir_path.exists():
                    try:
                        import shutil
                        shutil.rmtree(dir_path)
                        dirs_removed += 1
                        logger.debug(f"已删除目录: {dir_path}")
                    except Exception as e:
                        logger.warning(f"删除目录 {dir_path} 失败: {str(e)}")
            
            logger.info(f"清理完成: 删除了 {files_removed} 个文件和 {dirs_removed} 个目录")
        except Exception as e:
            logger.error(f"清理临时文件时出错: {str(e)}")

    def extract_audio(self, video_path):
        logger.info(f"开始从视频提取音频: {video_path}")
        start_time = time.time()
        
        video = VideoFileClip(video_path)
        audio = video.audio
        audio_path = "temp_audio.wav"
        logger.info("正在写入音频文件...")
        audio.write_audiofile(audio_path)
        
        duration = time.time() - start_time
        logger.info(f"音频提取完成，用时: {duration:.2f}秒")
        return audio_path

    def separate_vocals(self, audio_path):
        """
        使用 Demucs 进行人声分离
        """
        logger.info("开始进行人声分离...")
        try:
            # 创建输出目录
            output_dir = Path("separated")
            output_dir.mkdir(exist_ok=True)
            
            # 使用 demucs 进行分离
            command = [
                "demucs",
                "--two-stems=vocals",  # 只分离人声
                "-n", "demucs_quantized",  # 使用量化模型
                "--device", self.device,
                audio_path
            ]
            
            logger.info("执行人声分离命令...")
            subprocess.run(command, check=True)
            
            # 获取输出文件路径
            track_name = Path(audio_path).stem
            vocals_path = output_dir / "demucs_quantized" / track_name / "vocals.wav"
            
            if not vocals_path.exists():
                raise FileNotFoundError(f"人声文件未找到: {vocals_path}")
            
            logger.info(f"人声分离完成，输出文件: {vocals_path}")
            return str(vocals_path)
            
        except Exception as e:
            logger.error(f"人声分离失败: {str(e)}")
            return audio_path  # 如果分离失败，返回原始音频

    def process_audio(self, audio_path):
        """
        音频预处理
        """
        try:
            # 1. 人声分离
            vocals_path = self.separate_audio(audio_path)
            
            # 2. 音频标准化
            audio = AudioSegment.from_wav(vocals_path)
            
            # 标准化音量
            target_dBFS = -20.0
            change_in_dBFS = target_dBFS - audio.dBFS
            audio = audio.apply_gain(change_in_dBFS)
            
            # 导出处理后的音频
            processed_path = "processed_audio.wav"
            audio.export(processed_path, format="wav")
            
            return processed_path
            
        except Exception as e:
            logger.error(f"音频处理失败: {str(e)}")
            return audio_path

    def transcribe_audio(self, audio_path):
        logger.info("开始音频转录...")
        start_time = time.time()
        
        try:
            # 处理音频
            processed_audio = self.process_audio(audio_path)
            
            # 使用 Whisper 进行转录
            result = self.whisper_model.transcribe(
                processed_audio,
                language="pt",  # 指定源语言
                task="transcribe",
                temperature=0.2,  # 降低随机性
                best_of=5,  # 生成多个候选结果并选择最佳
                beam_size=5,  # 使用波束搜索
                word_timestamps=True,  # 获取词级时间戳
                condition_on_previous_text=True,  # 考虑上下文
                initial_prompt="这是一段对话视频",  # 提供上下文提示
            )
            
            segments = result["segments"]
            
            # 后处理结果
            processed_segments = self.post_process_segments(segments)
            
            duration = time.time() - start_time
            logger.info(f"音频转录完成，识别出 {len(processed_segments)} 个片段，用时: {duration:.2f}秒")
            return processed_segments
            
        except Exception as e:
            logger.error(f"转录失败: {str(e)}")
            raise

    def post_process_segments(self, segments):
        """对识别结果进行后处理"""
        processed_segments = []
        
        for i, segment in enumerate(segments):
            text = segment["text"]
            
            # 1. 清理文本
            text = text.strip()
            
            # 2. 移除重复内容
            if i > 0 and text in processed_segments[-1]["text"]:
                continue
                
            # 3. 合并短句
            if i > 0 and len(text) < 10:  # 如果当前段落很短
                if len(processed_segments) > 0:
                    # 将短句与前一个段落合并
                    prev_segment = processed_segments[-1]
                    prev_segment["text"] += " " + text
                    prev_segment["end"] = segment["end"]
                    continue
            
            # 4. 标点符号修正
            text = self.fix_punctuation(text)
            
            # 5. 更新段落信息
            segment["text"] = text
            processed_segments.append(segment)
        
        return processed_segments

    def fix_punctuation(self, text):
        """修正标点符号"""
        import re
        # 基本的标点符号修正
        text = re.sub(r'\s+([.,!?])', r'\1', text)  # 移除标点前的空格
        text = re.sub(r'([.,!?])\s+', r'\1 ', text)  # 确保标点后有空格
        text = text.replace('..', '.')  # 修正重复的句号
        return text

    def translate_text(self, text, target_lang='pt'):
        logger.info(f"翻译文本: {text[:50]}...")
        translated = self.translator.translate(text, dest=target_lang)
        logger.info(f"翻译结果: {translated.text[:50]}...")
        return translated.text

    def generate_voice_clone(self, text, speaker_wav, output_path):
        logger.info(f"生成克隆声音，文本长度: {len(text)}")
        start_time = time.time()
        
        try:
            # 移除 gpu 参数，使用 model.to(device) 来代替
            self.tts.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language="pt",
                file_path=output_path
            )
            
            duration = time.time() - start_time
            logger.info(f"声音克隆完成，输出到: {output_path}，用时: {duration:.2f}秒")
        except Exception as e:
            logger.error(f"生成声音克隆时出错: {str(e)}")
            raise

    def create_subtitle_clip(self, text, start_time, end_time):
        """
        创建字幕片段
        
        参数:
            text (str): 字幕文本
            start_time (float): 开始时间（秒）
            end_time (float): 结束时间（秒）
            
        返回:
            TextClip: 字幕片段
        """
        try:
            logger.info(f"创建字幕片段: {text[:30]}...")
            
            # 字幕样式配置
            config = {
                'fontsize': 30,
                'font': 'Arial',  # 或者使用 'Noto Sans CJK SC' 支持中文
                'color': 'white',
                'bg_color': 'rgba(0,0,0,0.7)',
                'size': (720, None),  # 宽度固定，高度自适应
                'method': 'caption',
                'align': 'center',
                'stroke_color': 'black',
                'stroke_width': 1
            }
            
            # 创建文本片段
            txt_clip = TextClip(
                text,
                **config
            )
            
            duration = end_time - start_time
            fade_duration = min(0.5, duration / 4)  # 淡入淡出时间
            
            # 设置位置、持续时间和特效
            final_clip = (txt_clip
                         .set_position(('center', 'bottom'))
                         .set_duration(duration)
                         .set_start(start_time)
                         .crossfadein(fade_duration)
                         .crossfadeout(fade_duration))
            
            logger.info(f"字幕片段创建成功，持续时间: {duration:.2f}秒")
            return final_clip
            
        except Exception as e:
            logger.error(f"创建字幕时出错: {str(e)}")
            # 创建一个简单的后备字幕
            return TextClip(
                text,
                fontsize=30,
                color='white',
                bg_color='rgba(0,0,0,0.5)',
                size=(720, 50)
            ).set_duration(end_time - start_time).set_start(start_time)

    def process_video(self, input_video_path):
        """处理视频的主方法"""
        try:
            # 处理前先清理临时文件
            self.cleanup_temp_files()
            
            # 如果使用 GPU，先清理缓存
            if self.device == "cuda":
                torch.cuda.empty_cache()

            total_start_time = time.time()
            logger.info(f"开始处理视频: {input_video_path}")
            
            # 创建临时目录
            os.makedirs("temp", exist_ok=True)
            
            # 1. 提取音频
            audio_path = self.extract_audio(input_video_path)
            
            # 2. 转录音频
            segments = self.transcribe_audio(audio_path)
            
            # 3. 翻译和生成新音频
            translated_segments = []
            new_audio_segments = []
            subtitle_clips = []
            
            # 使用tqdm显示进度
            for i, segment in enumerate(tqdm(segments, desc="处理语音片段")):
                # 翻译文本
                translated_text = self.translate_text(segment["text"])
                translated_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": translated_text
                })
                
                # 生成新的语音
                temp_audio_path = os.path.join("temp", f"temp_audio_{i}.wav")
                self.generate_voice_clone(
                    translated_text,
                    audio_path,
                    temp_audio_path
                )
                new_audio_segments.append({
                    "path": temp_audio_path,
                    "start": segment["start"],
                    "end": segment["end"]
                })
                
                # 创建字幕
                try:
                    subtitle_clip = self.create_subtitle_clip(
                        translated_text,
                        segment["start"],
                        segment["end"]
                    )
                    subtitle_clips.append(subtitle_clip)
                except Exception as e:
                    logger.error(f"创建字幕失败: {str(e)}")
            
            # 4. 合成最终视频
            logger.info("开始合成最终视频...")
            video = VideoFileClip(input_video_path)
            
            logger.info("合并音频片段...")
            final_audio = AudioSegment.silent(duration=int(video.duration * 1000))
            for segment in new_audio_segments:
                audio_clip = AudioSegment.from_wav(segment["path"])
                final_audio = final_audio.overlay(
                    audio_clip,
                    position=int(segment["start"] * 1000)
                )
            
            final_audio_path = os.path.join("temp", "final_audio.wav")
            final_audio.export(final_audio_path, format="wav")
            
            logger.info("创建最终视频...")
            final_video = CompositeVideoClip([
                video.set_audio(AudioFileClip(final_audio_path)),
                *subtitle_clips
            ])

            # 确保输出目录存在
            os.makedirs("output", exist_ok=True)
            output_path = os.path.join("output", f"translated_{os.path.basename(input_video_path)}")
            
            logger.info(f"导出最终视频到: {output_path}")
            final_video.write_videofile(
                output_path,
                fps=video.fps,
                codec="libx264",
                audio_codec="aac"
            )

            # 清理资源
            video.close()
            if hasattr(final_video, 'close'):
                final_video.close()

            total_duration = time.time() - total_start_time
            logger.info(f"视频处理完成！总用时: {total_duration:.2f}秒")
            logger.info(f"输出文件: {output_path}")
            
            # 处理完成后清理临时文件
            self.cleanup_temp_files()
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"处理视频时出错: {str(e)}")
            if self.device == "cuda":
                torch.cuda.empty_cache()
            # 发生错误时也清理临时文件
            self.cleanup_temp_files()
            raise

if __name__ == "__main__":
    try:
        # 设置输出目录
        os.makedirs("output", exist_ok=True)
        
        translator = VideoTranslator()
        input_video_path = "video-translate/demo-teste.mp4"
        
        # 检查输入文件是否存在
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"输入视频文件不存在: {input_video_path}")
            
        translator.process_video(input_video_path)
        
    except Exception as e:
        logger.error(f"处理过程中出现错误: {str(e)}", exc_info=True)
    finally:
        # 确保程序退出时清理临时文件
        if 'translator' in locals():
            translator.cleanup_temp_files()