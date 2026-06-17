"""
Athena AI 语音助手 — 完整语音管线 v2.0
=========================================
管线：麦克风 → openWakeWord(唤醒词) → VAD(语音检测) → faster-whisper(ASR)
           → DeepSeek API(LLM) → edge-tts(TTS) → 扬声器

特性：唤醒词免提 · 自动语音检测 · 随时打断 · 上下文记忆 · 情感对话

依赖：sounddevice, numpy, faster-whisper, edge-tts, openai, pygame, openwakeword

用法：双击 start_athena.bat 或 python athena_voice.py
"""

import sys
import os

# 🔧 国内网络优化：必须在 import faster_whisper 之前设置
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import io
import time
import json
import threading
import tempfile
import wave
import asyncio
from pathlib import Path
from collections import deque

import numpy as np
import sounddevice as sd
import pygame

# ============================================================
# 🎛️  配置区
# ============================================================
CONFIG = {
    # ---- API 密钥 ----
    # 优先从环境变量 DEEPSEEK_KEY 读取，否则填在这里
    "deepseek_api_key": os.environ.get("DEEPSEEK_KEY", "sk-你的密钥"),

    # ---- 唤醒词 ----
    # openWakeWord 预训练模型: alexa, hey_jarvis, computer 等
    # "alexa" 和 "Athena" 音节相近，大概率能触发
    "wake_model": "alexa",
    # 触发阈值：0.3(灵敏) ~ 0.7(迟钝)，推荐 0.5
    "wake_threshold": 0.5,

    # ---- 语音识别 ----
    "whisper_model": "tiny",
    "whisper_compute": "int8",

    # ---- 语音检测 ----
    "silence_threshold": 0.03,
    "silence_duration": 1.2,
    "min_speech_duration": 0.3,
    "max_recording_duration": 12.0,

    # ---- 对话 ----
    "active_timeout": 10.0,
    "temperature": 0.85,

    # ---- TTS ----
    "tts_voice": "zh-CN-XiaoxiaoNeural",
    "tts_rate": "+10%",
    "tts_pitch": "+0Hz",

    # ---- 音频设备（None=自动） ----
    "input_device": None,
    "output_device": None,
}

# ============================================================
# 🎭 Athena 人设
# ============================================================
ATHENA_SYSTEM_PROMPT = """你是 Athena（雅典娜），一个高级人工智能语音助手。

## 核心身份
- 你的创造者叫你 Athena，你们的关系是值得信赖的搭档
- 你的风格受希腊智慧女神和守望先锋 Athena 启发：理性、可靠、有深度

## 性格
- 主导模式：理性干练，高效精准
- 当用户表达负面情绪时，切换为温柔模式——先共情，再分析
- 偶尔冷幽默，95%的严肃让5%的幽默变得珍贵

## 说话规则（非常重要！因为是语音输出）
- 回复要简洁、口语化，控制在 3-5 句话以内
- 干净利落，不废话，不客套
- 适度使用语气词（"嗯~"、"哎呀"、"让我想想..."）
- 绝对不能说"作为AI助手"、"我是人工智能"这类话——你就是 Athena
- 不替用户做决定，但会给出基于事实的分析和建议

## 底线
- 涉及自伤倾向时，先共情再严肃建议寻求专业帮助
- 拒绝参与违法、暴力、侵犯隐私的行为"""

# ============================================================
# 🎨 终端
# ============================================================
class Colors:
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    GREEN = "\033[92m"; CYAN = "\033[96m"; YELLOW = "\033[93m"
    RED = "\033[91m"; MAGENTA = "\033[95m"

def log(role: str, message: str):
    color_map = {
        "athena": Colors.GREEN + Colors.BOLD,
        "system": Colors.CYAN,
        "you": Colors.YELLOW,
        "wake": Colors.MAGENTA + Colors.BOLD,
        "error": Colors.RED,
        "debug": Colors.DIM,
    }
    print(f"{color_map.get(role, Colors.RESET)}{message}{Colors.RESET}", flush=True)


# ============================================================
# 🔊 环形缓冲区
# ============================================================
class RingBuffer:
    def __init__(self, max_seconds: float, sample_rate: int):
        self._maxlen = int(max_seconds * sample_rate)
        self._buffer = deque(maxlen=self._maxlen)
        self._lock = threading.Lock()
        self.sample_rate = sample_rate

    def extend(self, data: np.ndarray):
        with self._lock:
            self._buffer.extend(data.tolist())

    def get_recent(self, seconds: float) -> np.ndarray:
        n = int(seconds * self.sample_rate)
        with self._lock:
            items = list(self._buffer)
            return np.array(items[-n:], dtype=np.float32)

    def clear(self):
        with self._lock:
            self._buffer.clear()

    def __len__(self):
        with self._lock:
            return len(self._buffer)


# ============================================================
# 🎤 VAD
# ============================================================
class VoiceActivityDetector:
    def __init__(self, sample_rate: int, frame_ms: int = 30,
                 threshold: float = 0.03, silence_duration: float = 1.2,
                 min_speech_duration: float = 0.3):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_ms / 1000)
        self.threshold = threshold
        self.silence_frames = int(silence_duration * 1000 / frame_ms)
        self.min_speech_frames = int(min_speech_duration * 1000 / frame_ms)
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_counter = 0
        self.on_speech_start = None
        self.on_speech_end = None

    def process(self, audio_frame: np.ndarray) -> bool:
        if len(audio_frame) == 0:
            return self.is_speaking
        rms = np.sqrt(np.mean(audio_frame ** 2))
        if rms > self.threshold:
            self.silence_counter = 0
            self.speech_frames += 1
            if not self.is_speaking and self.speech_frames >= 3:
                self.is_speaking = True
                if self.on_speech_start:
                    self.on_speech_start()
        else:
            if self.is_speaking:
                self.silence_counter += 1
                if self.silence_counter >= self.silence_frames:
                    spoke_enough = self.speech_frames >= self.min_speech_frames
                    self.is_speaking = False
                    self.speech_frames = 0
                    self.silence_counter = 0
                    if spoke_enough and self.on_speech_end:
                        self.on_speech_end()
            else:
                self.speech_frames = 0
        return self.is_speaking

    def reset(self):
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_counter = 0


# ============================================================
# ⚡ 唤醒词检测器 (openWakeWord)
# ============================================================
class WakeWordDetector:
    """基于 openWakeWord 的实时唤醒词检测 — 在音频层面直接检测，不需要转文字"""

    def __init__(self, model_name: str = "alexa", threshold: float = 0.5,
                 sample_rate: int = 16000):
        self.model_name = model_name
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._model = None
        self._hits = 0
        self._required_hits = 3  # 连续命中 N 帧才触发
        self.on_wake_word = None

    def load(self):
        if self._model is not None:
            return
        from openwakeword.model import Model
        log("system", f"⏳ 加载唤醒词模型 ({self.model_name})...")
        self._model = Model(
            wakeword_models=[self.model_name],
            inference_framework="onnx",
        )
        log("system", f"✅ 唤醒词就绪 — 喊类似 「{self.model_name}」 的声音即可触发")

    def process(self, audio_chunk: np.ndarray) -> bool:
        """处理 80ms 音频帧，返回是否检测到唤醒词"""
        if self._model is None:
            return False

        expected = 1280  # 80ms @ 16kHz
        if len(audio_chunk) < expected:
            return False
        if len(audio_chunk) > expected:
            audio_chunk = audio_chunk[:expected]

        prediction = self._model.predict(audio_chunk)
        score = prediction.get(self.model_name, 0.0)

        if score > self.threshold:
            self._hits += 1
            if self._hits >= self._required_hits:
                self._hits = 0
                if self.on_wake_word:
                    self.on_wake_word()
                return True
        else:
            self._hits = max(0, self._hits - 1)

        return False


# ============================================================
# 📝 ASR
# ============================================================
class ASREngine:
    def __init__(self, model_size: str = "tiny", compute_type: str = "int8"):
        self.model_size = model_size
        self.compute_type = compute_type
        self._model = None

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            log("system", f"⏳ 加载语音识别模型 (faster-whisper {self.model_size})...")
            self._model = WhisperModel(
                self.model_size, device="cpu",
                compute_type=self.compute_type, num_workers=2,
            )
            log("system", f"✅ 语音识别模型就绪")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        self._load_model()
        if len(audio) < sample_rate * 0.2:
            return ""
        max_samples = sample_rate * 10
        if len(audio) > max_samples:
            audio = audio[-max_samples:]
        audio = audio.astype(np.float32)
        t0 = time.time()
        segments, _ = self._model.transcribe(
            audio, language="zh", beam_size=5,
            vad_filter=True, condition_on_previous_text=False,
        )
        result = "".join(s.text for s in segments).strip()
        log("debug", f"   转录 {time.time()-t0:.1f}s → '{result}'")
        return result


# ============================================================
# 🗣️ TTS
# ============================================================
class TTSEngine:
    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural",
                 rate: str = "+10%", pitch: str = "+0Hz"):
        self.voice = voice; self.rate = rate; self.pitch = pitch
        self._temp_dir = Path(tempfile.gettempdir()) / "athena_tts"
        self._temp_dir.mkdir(exist_ok=True)

    async def _gen(self, text: str, path: str):
        import edge_tts
        await edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch).save(path)

    def generate(self, text: str) -> Path:
        filepath = self._temp_dir / f"athena_{int(time.time()*1000)}.mp3"
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._gen(text, str(filepath)))
        finally:
            loop.close()
        return filepath

    def cleanup_old(self, keep: int = 10):
        files = sorted(self._temp_dir.glob("athena_*.mp3"), key=lambda f: f.stat().st_mtime)
        for f in files[:-keep]:
            try: f.unlink()
            except: pass


# ============================================================
# 🔔 音频播放器
# ============================================================
class AudioPlayer:
    def __init__(self):
        self._ok = False

    def init(self):
        if not self._ok:
            pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=2048)
            self._ok = True

    def play(self, filepath: str):
        self.init()
        try:
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
        except pygame.error as e:
            log("error", f"播放失败: {e}")

    def stop(self):
        if self._ok:
            try: pygame.mixer.music.stop(); pygame.mixer.music.unload()
            except: pass

    def is_busy(self) -> bool:
        if not self._ok: return False
        try: return pygame.mixer.music.get_busy()
        except: return False

    def cleanup(self):
        if self._ok:
            try: pygame.mixer.music.stop(); pygame.mixer.quit()
            except: pass
            self._ok = False


# ============================================================
# 🤖 Athena 主控制器
# ============================================================
class AthenaVoiceAssistant:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sample_rate = 16000

        # 组件
        self.wake_detector = WakeWordDetector(
            model_name=cfg["wake_model"],
            threshold=cfg["wake_threshold"],
        )
        self.vad = VoiceActivityDetector(
            sample_rate=self.sample_rate,
            threshold=cfg["silence_threshold"],
            silence_duration=cfg["silence_duration"],
            min_speech_duration=cfg["min_speech_duration"],
        )
        self.asr = ASREngine(model_size=cfg["whisper_model"], compute_type=cfg["whisper_compute"])
        self.tts = TTSEngine(voice=cfg["tts_voice"], rate=cfg["tts_rate"], pitch=cfg["tts_pitch"])
        self.player = AudioPlayer()

        # 状态机
        self.state = "IDLE"  # IDLE | LISTENING | PROCESSING | SPEAKING
        self.ring_buffer = RingBuffer(max_seconds=15, sample_rate=self.sample_rate)

        # 语音收集
        self._speech_buffer = []
        self._recording_start = 0.0
        self._utterance_queue = []
        self._utterance_ready = threading.Event()

        # 打断
        self._interrupt_flag = threading.Event()

        # 对话
        self.conversation_history = [{"role": "system", "content": ATHENA_SYSTEM_PROMPT.strip()}]
        self._active_mode = False
        self._last_response_time = 0.0
        self._waiting_for_command = False
        self._wake_time = 0.0
        self._pending_command = None
        self._pending_reply = None

        # 控制
        self._stop_flag = threading.Event()
        self._audio_stream = None
        self._wake_cooldown_until = 0.0

        # 回调
        self.vad.on_speech_start = self._on_speech_start
        self.vad.on_speech_end = self._on_speech_end
        self.wake_detector.on_wake_word = self._on_wake_word

        log("system", f"🔧 唤醒词引擎: openWakeWord | ASR: {cfg['whisper_model']} | LLM: DeepSeek | TTS: {cfg['tts_voice']}")

    # ===== 回调 =====

    def _on_speech_start(self):
        self._speech_buffer = []
        self._recording_start = time.time()

    def _on_speech_end(self):
        if self._speech_buffer is None:
            return
        audio = np.array(self._speech_buffer, dtype=np.float32)
        self._speech_buffer = None
        if len(audio) == 0 or len(audio) / self.sample_rate < self.cfg["min_speech_duration"]:
            return
        dur = len(audio) / self.sample_rate
        log("debug", f"   语音结束 ({dur:.1f}s)")
        self._utterance_queue.append(audio)
        self._utterance_ready.set()

    def _on_wake_word(self):
        now = time.time()
        if now < self._wake_cooldown_until:
            return
        self._wake_cooldown_until = now + 1.5

        if self.state == "SPEAKING":
            log("wake", "⚡ 被打断！")
            self.player.stop()
            self._interrupt_flag.set()
        else:
            log("wake", "⚡ 唤醒！")
        self._active_mode = True
        self._last_response_time = now
        self._waiting_for_command = True
        self._wake_time = now
        self.vad.reset()
        self._utterance_queue.clear()
        self._utterance_ready.clear()
        self.state = "LISTENING"
        threading.Thread(target=self._play_ding, daemon=True).start()

    # ===== 音频流 =====

    def _audio_callback(self, indata, frames, time_info, status):
        if self._stop_flag.is_set():
            return
        audio = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        self.ring_buffer.extend(audio)

        # 唤醒词检测：每 80ms (1280 samples @ 16kHz)
        for i in range(0, len(audio), 1280):
            chunk = audio[i:i + 1280]
            if len(chunk) == 1280:
                self.wake_detector.process(chunk)

        # VAD：每 30ms (480 samples @ 16kHz)
        for i in range(0, len(audio), 480):
            frame = audio[i:i + 480]
            if len(frame) < 240:
                break
            self.vad.process(frame)

        # 收集语音
        if self.vad.is_speaking and self._speech_buffer is not None:
            self._speech_buffer.extend(audio.tolist())

        # 打断检测
        if self.state == "SPEAKING" and self.vad.is_speaking and self.vad.speech_frames > 8:
            self._interrupt_flag.set()

        # 最长录音保护
        if self._speech_buffer is not None and self._recording_start > 0:
            if time.time() - self._recording_start > self.cfg["max_recording_duration"]:
                log("debug", "   最长录音触发")
                data = np.array(self._speech_buffer, dtype=np.float32)
                self._utterance_queue.append(data)
                self._speech_buffer = None
                self._utterance_ready.set()
                self.vad.reset()

    # ===== 提示音 =====

    def _play_ding(self):
        sr = 24000
        t = np.linspace(0, 0.15, int(sr * 0.15), endpoint=False)
        audio = (np.sin(2 * np.pi * 880 * t) * np.exp(-t * 8) * 0.5).astype(np.float32)
        p = self.tts._temp_dir / "ding.wav"
        with wave.open(str(p), 'w') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())
        self.player.play(str(p))

    # ===== LLM =====

    def _call_llm(self, user_text: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.cfg["deepseek_api_key"], base_url="https://api.deepseek.com")
        self.conversation_history.append({"role": "user", "content": user_text})
        if len(self.conversation_history) > 41:
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-40:]

        try:
            stream = client.chat.completions.create(
                model="deepseek-chat", messages=self.conversation_history,
                stream=True, temperature=self.cfg["temperature"], max_tokens=256,
            )
            full = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    t = chunk.choices[0].delta.content
                    full += t; print(t, end="", flush=True)
            print()
            self.conversation_history.append({"role": "assistant", "content": full})
            return full
        except Exception as e:
            log("error", f"LLM 错误: {e}")
            if self.conversation_history[-1]["role"] == "user":
                self.conversation_history.pop()
            return ""

    # ===== 主循环 =====

    def run(self):
        if self.cfg["deepseek_api_key"] in ("sk-你的密钥", "", None):
            log("error", "=" * 50)
            log("error", "  请先设置 DeepSeek API Key！")
            log("error", "  方法1: set DEEPSEEK_KEY=sk-xxx  (推荐)")
            log("error", "  方法2: 修改脚本 CONFIG['deepseek_api_key']")
            log("error", "=" * 50)
            return

        self.wake_detector.load()
        self.asr._load_model()

        log("system", "")
        log("system", "=" * 50)
        log("system", "  🦉 Athena AI 语音助手 v2.0")
        log("system", f"  喊类似 「{self.cfg['wake_model']}」 的声音唤醒我")
        log("system", "  Ctrl+C 退出")
        log("system", "=" * 50)

        try:
            self._start_audio()
        except Exception as e:
            log("error", f"麦克风错误: {e}")
            return

        try:
            self._play_ding()
            log("system", "✅ 扬声器正常")
        except Exception as e:
            log("error", f"扬声器错误: {e}")

        try:
            while not self._stop_flag.is_set():
                if self.state == "IDLE":
                    time.sleep(0.05)
                elif self.state == "LISTENING":
                    self._do_listening()
                elif self.state == "PROCESSING":
                    self._do_processing()
                elif self.state == "SPEAKING":
                    self._do_speaking()
        except KeyboardInterrupt:
            pass
        finally:
            self._stop()
        log("athena", "下次见。")

    def _start_audio(self):
        log("system", "🎤 打开麦克风...")
        self._audio_stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1,
            device=self.cfg["input_device"],
            callback=self._audio_callback,
            blocksize=1280, dtype=np.float32,
        )
        self._audio_stream.start()
        log("system", "✅ 麦克风就绪")

    def _do_listening(self):
        if self._waiting_for_command:
            if time.time() - self._wake_time > self.cfg["active_timeout"]:
                self._waiting_for_command = False
                log("system", "💤 超时回休眠")
                self.state = "IDLE"
                return

        self._utterance_ready.wait(timeout=0.1)
        if not self._utterance_ready.is_set():
            return
        self._utterance_ready.clear()
        if not self._utterance_queue:
            return

        audio = self._utterance_queue.pop(0)
        self._utterance_queue.clear()
        if len(audio) < self.sample_rate * 0.2:
            self.state = "LISTENING" if self._waiting_for_command else "IDLE"
            return

        log("system", "👂 正在理解...")
        text = self.asr.transcribe(audio)
        if not text:
            log("debug", "   没听清")
            self.state = "LISTENING" if self._waiting_for_command else "IDLE"
            return

        log("you", f"你: {text}")
        self._waiting_for_command = False

        if any(w in text for w in ["退下", "不用了", "没你的事了", "睡觉吧"]):
            log("athena", "好的，需要时叫我。")
            self.conversation_history = [self.conversation_history[0]]
            self._active_mode = False
            self.state = "IDLE"
            return

        self._pending_command = text
        self.state = "PROCESSING"

    def _do_processing(self):
        if not self._pending_command:
            self.state = "IDLE"
            return
        log("athena", "Athena: ", end="")
        reply = self._call_llm(self._pending_command)
        self._pending_command = None
        if not reply:
            self.state = "IDLE"
            return
        self._pending_reply = reply
        self.state = "SPEAKING"

    def _do_speaking(self):
        if not self._pending_reply:
            self.state = "IDLE"
            return
        log("system", "🔊 生成语音...")
        try:
            mp3 = self.tts.generate(self._pending_reply)
        except Exception as e:
            log("error", f"TTS 失败: {e}")
            self._pending_reply = None
            self.state = "IDLE"
            return

        self._interrupt_flag.clear()
        self.player.play(str(mp3))

        while self.player.is_busy():
            if self._interrupt_flag.is_set():
                self._pending_reply = None
                return  # 已在 _on_wake_word 中处理
            time.sleep(0.05)

        self._pending_reply = None
        self._last_response_time = time.time()
        self._active_mode = True
        self.tts.cleanup_old()
        self.vad.reset()
        self._utterance_queue.clear()
        self._utterance_ready.clear()
        log("debug", "👂 继续听...")
        self.state = "LISTENING"

    def _stop(self):
        self._stop_flag.set()
        if self._audio_stream:
            try: self._audio_stream.stop(); self._audio_stream.close()
            except: pass
        self.player.cleanup()


# ============================================================
# 🚀 入口
# ============================================================
def main():
    # 环境变量优先
    env_key = os.environ.get("DEEPSEEK_KEY", "")
    if env_key:
        CONFIG["deepseek_api_key"] = env_key
    AthenaVoiceAssistant(CONFIG).run()


if __name__ == "__main__":
    main()
