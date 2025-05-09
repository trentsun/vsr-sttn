import os
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
from TTS.utils.audio import AudioProcessor

import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加安全全局类
add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    BaseDatasetConfig,
    BaseTTSConfig,
    AudioProcessor,
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
        
        # 设置更安全的torch.load行为
        def safe_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            if 'map_location' not in kwargs:
                kwargs['map_location'] = torch.device('cpu')
            try:
                return torch.load(*args, **kwargs)
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}")
                raise

        # 替换默认的torch.load
        torch.load = safe_load
        
        try:
            logger.info("加载 Whisper 模型...")
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper 模型加载完成")
            
            logger.info("初始化 Translator...")
            self.translator = googletrans.Translator()
            logger.info("Translator 初始化完成")
            
            logger.info("初始化 TTS 模型...")
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            logger.info("TTS 模型加载成功")
            
        except Exception as e:
            logger.error(f"初始化失败: {str(e)}", exc_info=True)
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
            file_path=output_path
        )
        
        duration = time.time() - start_time
        logger.info(f"声音克隆完成，输出到: {output_path}，用时: {duration:.2f}秒")

    def process_video(self, input_video_path):
        logger.info(f"开始处理视频: {input_video_path}")
        total_start_time = time.time()
        
        # 1. 提取音频
        audio_path = self.extract_audio(input_video_path)
        
        # 2. 转录音频
        segments = self.transcribe_audio(audio_path)
        
        # 3. 翻译和生成新音频
        translated_segments = []
        new_audio_segments = []
        subtitle_clips = []

        logger.info(f"开始处理 {len(segments)} 个音频片段...")
        for i, segment in enumerate(segments, 1):
            logger.info(f"处理第 {i}/{len(segments)} 个片段...")
            
            # 翻译文本
            translated_text = self.translate_text(segment["text"])
            translated_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": translated_text
            })

            # 生成克隆声音
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
            logger.info("创建字幕片段...")
            subtitle_clips.append(
                self.create_subtitle_clip(
                    translated_text,
                    segment["start"],
                    segment["end"]
                )
            )

        # 4. 合成最终视频
        logger.info("开始合成最终视频...")
        video = VideoFileClip(input_video_path)
        
        logger.info("合并音频片段...")
        final_audio = AudioSegment.silent(duration=0)
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
        logger.info("清理临时文件...")
        os.remove(audio_path)
        os.remove("final_audio.wav")
        for segment in new_audio_segments:
            os.remove(segment["path"])

        total_duration = time.time() - total_start_time
        logger.info(f"视频处理完成！总用时: {total_duration:.2f}秒")
        logger.info(f"输出文件: {output_path}")

if __name__ == "__main__":
    try:
        translator = VideoTranslator()
        input_video_path = "video-translate/demo-teste.mp4"
        translator.process_video(input_video_path)
    except Exception as e:
        logger.error(f"处理过程中出现错误: {str(e)}", exc_info=True)