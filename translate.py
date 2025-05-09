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
from TTS.tts.configs.xtts_config import XttsConfig

class VideoTranslator:
    def __init__(self):
        # 初始化必要的模型和工具
        self.whisper_model = whisper.load_model("base")
        self.translator = googletrans.Translator()
        
        # 添加XttsConfig到安全全局类列表
        add_safe_globals([XttsConfig])
        
        # 初始化TTS模型
        try:
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        except Exception as e:
            # 如果上述方法失败，尝试使用weights_only=False的方式
            torch.load = lambda f, *args, **kwargs: torch.load(f, *args, **kwargs, weights_only=False)
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        
    def extract_audio(self, video_path):
        """从视频中提取音频"""
        video = VideoFileClip(video_path)
        audio = video.audio
        audio_path = "temp_audio.wav"
        audio.write_audiofile(audio_path)
        return audio_path

    def transcribe_audio(self, audio_path):
        """使用Whisper转录音频为文本"""
        result = self.whisper_model.transcribe(audio_path)
        return result["segments"]

    def translate_text(self, text, target_lang='pt'):
        """将文本翻译成目标语言"""
        translated = self.translator.translate(text, dest=target_lang)
        return translated.text

    def generate_voice_clone(self, text, speaker_wav, output_path):
        """生成克隆声音"""
        self.tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language="pt",
            file_path=output_path
        )

    def create_subtitle_clip(self, text, start_time, end_time):
        """创建字幕片段"""
        return TextClip(
            text,
            fontsize=24,
            color='white',
            bg_color='black',
            size=(720, 50)
        ).set_position(('center', 'bottom')).set_duration(end_time - start_time).set_start(start_time)

    def process_video(self, input_video_path):
        """处理整个视频翻译流程"""
        print("开始处理视频...")
        
        # 1. 提取音频
        audio_path = self.extract_audio(input_video_path)
        print("音频提取完成")

        # 2. 转录音频
        segments = self.transcribe_audio(audio_path)
        print("音频转录完成")

        # 3. 翻译和生成新音频
        translated_segments = []
        new_audio_segments = []
        subtitle_clips = []

        for segment in segments:
            # 翻译文本
            translated_text = self.translate_text(segment["text"])
            translated_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": translated_text
            })

            # 为每个片段生成克隆声音
            temp_audio_path = f"temp_audio_{len(new_audio_segments)}.wav"
            self.generate_voice_clone(
                translated_text,
                audio_path,  # 使用原始音频作为参考
                temp_audio_path
            )
            new_audio_segments.append({
                "path": temp_audio_path,
                "start": segment["start"],
                "end": segment["end"]
            })

            # 创建字幕
            subtitle_clips.append(
                self.create_subtitle_clip(
                    translated_text,
                    segment["start"],
                    segment["end"]
                )
            )

        # 4. 合成最终视频
        video = VideoFileClip(input_video_path)
        
        # 合并所有音频片段
        final_audio = AudioSegment.silent(duration=0)
        for segment in new_audio_segments:
            audio_clip = AudioSegment.from_wav(segment["path"])
            final_audio = final_audio.overlay(
                audio_clip,
                position=int(segment["start"] * 1000)
            )
        
        final_audio.export("final_audio.wav", format="wav")
        
        # 创建最终视频
        final_video = CompositeVideoClip([
            video.set_audio(AudioFileClip("final_audio.wav")),
            *subtitle_clips
        ])

        # 导出最终视频
        output_path = "translated_video.mp4"
        final_video.write_videofile(
            output_path,
            fps=video.fps,
            codec="libx264",
            audio_codec="aac"
        )

        # 清理临时文件
        os.remove(audio_path)
        os.remove("final_audio.wav")
        for segment in new_audio_segments:
            os.remove(segment["path"])

        print("视频处理完成！输出文件：", output_path)

if __name__ == "__main__":
    translator = VideoTranslator()
    input_video_path = "video-translate/demo-teste.mp4"
    translator.process_video(input_video_path)