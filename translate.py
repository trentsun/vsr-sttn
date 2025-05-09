import os
import time
import glob  # 用于文件模式匹配
import sys   # 用于系统相关操作
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
from transformers import pipeline
import torch
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
        
        try:
            logger.info("加载 Whisper 模型...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model = whisper.load_model("base").to(device)
            
            logger.info("Whisper 模型加载完成")
            
            logger.info("初始化 Translator...")
            self.translator = googletrans.Translator()
            logger.info("Translator 初始化完成")
            
            logger.info("初始化 TTS 模型...")
            try:
                # 第一次尝试加载模型
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            except Exception as e:
                logger.warning(f"首次加载失败，尝试使用 weights_only=False: {str(e)}")
                # 使用原始的torch.load，但添加weights_only=False参数
                def load_with_weights(*args, **kwargs):
                    kwargs['weights_only'] = False
                    kwargs['map_location'] = torch.device('cpu')
                    return original_torch_load(*args, **kwargs)
                
                # 临时替换torch.load
                torch.load = load_with_weights
                # 再次尝试加载模型
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
                # 恢复原始的torch.load
                torch.load = original_torch_load
                
            logger.info("TTS 模型加载成功")
            
        except Exception as e:
            logger.error(f"初始化失败: {str(e)}", exc_info=True)
            # 确保恢复原始的torch.load
            torch.load = original_torch_load
            raise


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

    def transcribe_audio(self, audio_path):
        logger.info("开始音频转录...")
        start_time = time.time()
        
        result = self.whisper_model.transcribe(audio_path)
        segments = result["segments"]
        
        duration = time.time() - start_time
        logger.info(f"音频转录完成，识别出 {len(segments)} 个片段，用时: {duration:.2f}秒")
        return segments

    def translate_text(self, text, target_lang='pt'):
        logger.info(f"翻译文本: {text[:50]}...")
        translated = self.translator.translate(text, dest=target_lang)
        logger.info(f"翻译结果: {translated.text[:50]}...")
        return translated.text

    def generate_voice_clone(self, text, speaker_wav, output_path):
        logger.info(f"生成克隆声音，文本长度: {len(text)}")
        start_time = time.time()
        
        self.tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language="pt",
            file_path=output_path,
            gpu=self.device == "cuda"  # 添加GPU支持
        )
        
        duration = time.time() - start_time
        logger.info(f"声音克隆完成，输出到: {output_path}，用时: {duration:.2f}秒")

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
            total_start_time = time.time()
            logger.info(f"开始处理视频: {input_video_path}")
            
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
                temp_audio_path = f"temp_audio_{i}.wav"
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
            
            final_audio.export("final_audio.wav", format="wav")
            
            logger.info("创建最终视频...")
            final_video = CompositeVideoClip([
                video.set_audio(AudioFileClip("final_audio.wav")),
                *subtitle_clips
            ])

            output_path = "translated_video.mp4"
            logger.info(f"导出最终视频到: {output_path}")
            final_video.write_videofile(
                output_path,
                fps=video.fps,
                codec="libx264",
                audio_codec="aac"
            )

            # 清理临时文件
            # self.cleanup_temp_files(audio_path, new_audio_segments)
            
            total_duration = time.time() - total_start_time
            logger.info(f"视频处理完成！总用时: {total_duration:.2f}秒")
            logger.info(f"输出文件: {output_path}")
            
        except Exception as e:
            logger.error(f"处理视频时出错: {str(e)}")
            raise
        
    #  def cleanup_temp_files(self, audio_path, audio_segments):
    #     """清理临时文件"""
    #     try:
    #         logger.info("清理临时文件...")
    #         files_to_remove = [
    #             audio_path,
    #             "final_audio.wav"
    #         ]
            
    #         # 添加临时音频片段文件
    #         for segment in audio_segments:
    #             files_to_remove.append(segment["path"])
            
    #         # 删除文件
    #         for file_path in files_to_remove:
    #             if os.path.exists(file_path):
    #                 os.remove(file_path)
    #                 logger.debug(f"已删除: {file_path}")
                    
    #         logger.info("临时文件清理完成")
    #     except Exception as e:
    #         logger.warning(f"清理临时文件时出现错误: {str(e)}")


if __name__ == "__main__":
    try:
        translator = VideoTranslator()
        input_video_path = "video-translate/demo-teste.mp4"
        translator.process_video(input_video_path)
    except Exception as e:
        logger.error(f"处理过程中出现错误: {str(e)}", exc_info=True)