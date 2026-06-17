"""
Athena AI 语音助手 — v3.0 拼音模糊匹配版
============================================
管线：麦克风 → VAD → faster-whisper(ASR) → 拼音模糊匹配唤醒词
           → DeepSeek API(LLM) → edge-tts(TTS) → 扬声器

唤醒词方案：faster-whisper tiny 转录 + pypinyin 拼音对比
- 不用 openWakeWord（无法识别自定义唤醒词 "Athena"）
- 不用精确文字匹配（Whisper 可能把 Athena 转录成各种中文音译）
- 用拼音序列匹配：只要前几个音节对得上就算唤醒

用法：双击 start_athena.bat
"""

import sys
import os

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 🔑 从 .env 文件加载密钥（不会被上传到 GitHub）
def _load_env():
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    if key.strip() not in os.environ:
                        os.environ[key.strip()] = val.strip()
_load_env()

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
# 🎛️ 配置
# ============================================================
CONFIG = {
    # 密钥从此文件同级目录的 .env 文件读取（不会被上传到 GitHub）
    "deepseek_api_key": os.environ.get("DEEPSEEK_KEY", "sk-你的密钥"),

    # ---- 唤醒词（拼音模糊匹配） ----
    # 预期拼音序列（Athena 的中文音译 ≈ a si na / a xi na / a sei na）
    "wake_pinyin_patterns": [
        ["a", "si", "na"],      # 阿斯娜
        ["a", "xi", "na"],      # 阿西娜
        ["a", "sei", "na"],     # 阿瑟娜
        ["a", "di", "na"],      # 阿蒂娜
        ["a", "ti", "na"],      # 阿提娜
        ["a", "sen", "na"],     # 阿森纳
        ["a", "se", "na"],      # 阿色娜
        ["a", "shi", "na"],     # 阿是娜
        ["hei", "a", "si", "na"],    # 嘿阿斯娜
        ["hei", "a", "xi", "na"],    # 嘿阿西娜
        ["ya", "dian", "na"],   # 雅典娜
    ],
    # 匹配前 N 个拼音即可（1 太宽泛，3 最严格）
    "wake_min_pinyin_match": 2,

    # ---- 语音识别 ----
    "whisper_model": "tiny",
    "whisper_compute": "int8",

    # ---- 语音检测 ----
    "silence_threshold": 0.05,    # 提高！减少噪音误触发
    "silence_duration": 1.0,
    "min_speech_duration": 0.5,   # 提高！过滤短噪音
    "max_recording_duration": 10.0,

    # ---- 对话 ----
    "active_timeout": 8.0,   # 唤醒后等 8 秒
    "temperature": 0.85,

    # ---- TTS ----
    "tts_voice": "zh-CN-XiaoxiaoNeural",
    "tts_rate": "+10%",
    "tts_pitch": "+0Hz",

    # ---- 音频设备 ----
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

def log(role: str, msg: str):
    cmap = {"athena": Colors.GREEN + Colors.BOLD, "system": Colors.CYAN,
            "you": Colors.YELLOW, "wake": Colors.MAGENTA + Colors.BOLD,
            "error": Colors.RED, "debug": Colors.DIM}
    print(f"{cmap.get(role, Colors.RESET)}{msg}{Colors.RESET}", flush=True)


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
                    ok = self.speech_frames >= self.min_speech_frames
                    self.is_speaking = False
                    self.speech_frames = 0
                    self.silence_counter = 0
                    if ok and self.on_speech_end:
                        self.on_speech_end()
            else:
                self.speech_frames = 0
        return self.is_speaking

    def reset(self):
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_counter = 0


# ============================================================
# 📝 ASR
# ============================================================
class ASREngine:
    def __init__(self, model_size: str = "tiny", compute_type: str = "int8"):
        self.model_size = model_size; self.compute_type = compute_type
        self._model = None

    def load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            log("system", f"⏳ 加载语音识别模型 (faster-whisper {self.model_size})...")
            self._model = WhisperModel(
                self.model_size, device="cpu",
                compute_type=self.compute_type, num_workers=2,
            )
            log("system", "✅ 语音识别模型就绪")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if self._model is None:
            self.load()
        if len(audio) < sample_rate * 0.3:
            return ""

        # 限制长度：唤醒词只取 3 秒，命令取 8 秒
        max_s = sample_rate * 3 if len(audio) < sample_rate * 5 else sample_rate * 8
        if len(audio) > max_s:
            audio = audio[-max_s:]

        audio = audio.astype(np.float32)
        t0 = time.time()
        segments, _ = self._model.transcribe(
            audio, language="zh", beam_size=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # 提高准确度
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=500,
            ),
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        result = "".join(s.text for s in segments).strip()
        # 过滤纯噪音模式
        if result and len(set(result)) <= 2 and len(result) > 3:
            log("debug", f"   ⚠️ 疑似噪音，忽略")
            return ""
        log("debug", f"   转录 {time.time()-t0:.1f}s → '{result}'")
        return result


# ============================================================
# ⚡ 唤醒词检测器（拼音模糊匹配版）
# ============================================================
class WakeWordMatcher:
    """用 pypinyin 做拼音模糊匹配，不再需要精确文字匹配"""

    def __init__(self, patterns: list[list[str]], min_match: int = 2):
        self.patterns = patterns  # 预期的拼音序列列表
        self.min_match = min_match  # 最少匹配前几个拼音

    def _to_pinyin(self, text: str) -> list[str]:
        """将文本转为拼音列表（无声调）"""
        from pypinyin import lazy_pinyin, Style
        return lazy_pinyin(text, style=Style.NORMAL, errors="ignore")

    def match(self, text: str) -> tuple[bool, str]:
        """
        检查文本是否匹配唤醒词
        返回：(是否匹配, 去掉唤醒词后的文本)
        """
        if not text:
            return False, ""

        text_lower = text.lower().strip()

        # 1. 精确英文匹配（如果 Whisper 直接输出了英文 "athena"）
        for prefix in ["athena", "Athena", "ATHENA", "hey athena", "hey Athena"]:
            if text_lower.startswith(prefix):
                rest = text_lower[len(prefix):].strip()
                for sep in [",", " ", "，", "。"]:
                    if rest.startswith(sep):
                        rest = rest[1:].strip()
                log("debug", f"   英文精确匹配: '{prefix}'")
                return True, rest
            # 也检查包含
            if prefix in text_lower:
                pos = text_lower.find(prefix)
                if pos <= 3:  # 前3个字符内
                    rest = text_lower[pos + len(prefix):].strip()
                    for sep in [",", " ", "，", "。"]:
                        if rest.startswith(sep):
                            rest = rest[1:].strip()
                    log("debug", f"   英文包含匹配: '{prefix}'")
                    return True, rest

        # 2. 拼音模糊匹配
        text_pinyin = self._to_pinyin(text_lower)
        log("debug", f"   拼音: {text_pinyin}")

        for pattern in self.patterns:
            # 比较前 N 个拼音
            match_count = 0
            for i, (tp, pp) in enumerate(zip(text_pinyin, pattern)):
                if i >= len(pattern):
                    break
                if tp == pp:
                    match_count += 1
                elif self._similar(tp, pp):
                    match_count += 0.5  # 近似匹配算半分

            if match_count >= self.min_match:
                log("debug", f"   拼音匹配: pattern={pattern} score={match_count}")
                # 尝试提取后续文本
                # 找到最后匹配的拼音位置，之后的内容就是命令
                wake_end = len(pattern)
                remaining_chars = text_lower
                # 简单策略：如果文本比唤醒词长，返回剩余部分
                if len(text_lower) > wake_end:
                    rest = text_lower[wake_end:]
                    for sep in ["，", ",", " ", "。", "、"]:
                        if rest.startswith(sep):
                            rest = rest[1:].strip()
                    if len(rest) > 1:
                        return True, rest
                return True, ""

        # 3. 中文精确匹配（兜底）
        for ww in ["雅典娜", "阿斯娜", "阿西娜", "阿森纳", "嘿雅典娜", "嘿阿斯娜"]:
            if text_lower.startswith(ww):
                rest = text_lower[len(ww):].strip()
                for sep in ["，", ",", " ", "、", "。"]:
                    if rest.startswith(sep):
                        rest = rest[1:].strip()
                log("debug", f"   中文精确匹配: '{ww}'")
                return True, rest

        return False, ""

    @staticmethod
    def _similar(a: str, b: str) -> bool:
        """检查两个拼音是否近似"""
        # 常见混淆对
        similar_pairs = [
            ("si", "shi"), ("shi", "si"),
            ("xi", "si"), ("si", "xi"),
            ("xi", "shi"), ("shi", "xi"),
            ("na", "la"), ("la", "na"),
            ("na", "nuo"), ("nuo", "na"),
            ("sen", "shen"), ("shen", "sen"),
            ("se", "she"), ("she", "se"),
            ("di", "ti"), ("ti", "di"),
            ("zhi", "zi"), ("zi", "zhi"),
            ("chi", "ci"), ("ci", "chi"),
            ("juan", "jue"), ("lun", "long"),
            ("hei", "hai"), ("hai", "hei"),
        ]
        return (a, b) in similar_pairs or (b, a) in similar_pairs


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
        p = self._temp_dir / f"athena_{int(time.time()*1000)}.mp3"
        loop = asyncio.new_event_loop()
        try: loop.run_until_complete(self._gen(text, str(p)))
        finally: loop.close()
        return p

    def cleanup(self, keep: int = 10):
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
        try: pygame.mixer.music.load(filepath); pygame.mixer.music.play()
        except pygame.error as e: log("error", f"播放失败: {e}")

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
            except: pass; self._ok = False


# ============================================================
# 🤖 Athena 主控制器
# ============================================================
class AthenaVoiceAssistant:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sample_rate = 16000

        # 组件
        self.wake_matcher = WakeWordMatcher(
            patterns=cfg["wake_pinyin_patterns"],
            min_match=cfg["wake_min_pinyin_match"],
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
        self._speech_queue = []  # (audio, is_from_idle) 元组
        self._speech_event = threading.Event()

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

        log("system", f"🔧 唤醒词: 拼音模糊匹配 | ASR: {cfg['whisper_model']} | LLM: DeepSeek | TTS: {cfg['tts_voice']}")

    # ===== 回调 =====

    def _on_speech_start(self):
        self._speech_buffer = []
        self._recording_start = time.time()

    def _on_speech_end(self):
        if self._speech_buffer is None:
            return
        audio = np.array(self._speech_buffer, dtype=np.float32)
        self._speech_buffer = None
        dur = len(audio) / self.sample_rate
        if dur < self.cfg["min_speech_duration"]:
            return
        log("debug", f"   语音结束 ({dur:.1f}s)")

        # 标记来源：IDLE 状态的语音可能是唤醒词
        from_idle = (self.state == "IDLE")
        self._speech_queue.append((audio, from_idle))
        self._speech_event.set()

    # ===== 音频流 =====

    def _audio_callback(self, indata, frames, time_info, status):
        if self._stop_flag.is_set():
            return
        audio = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        self.ring_buffer.extend(audio)

        # VAD 处理
        for i in range(0, len(audio), 480):  # 30ms @ 16kHz
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

        # 最长录音
        if self._speech_buffer is not None and self._recording_start > 0:
            if time.time() - self._recording_start > self.cfg["max_recording_duration"]:
                log("debug", "   最长录音触发")
                data = np.array(self._speech_buffer, dtype=np.float32)
                self._speech_queue.append((data, self.state == "IDLE"))
                self._speech_buffer = None
                self._speech_event.set()
                self.vad.reset()

    # ===== 提示音 =====

    def _play_ding(self):
        sr = 24000
        t = np.linspace(0, 0.12, int(sr * 0.12), endpoint=False)
        audio = (np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 10) * 0.4).astype(np.float32)
        # 每次用不同的文件名，防止多线程冲突
        p = self.tts._temp_dir / f"ding_{threading.get_ident()}.wav"
        try:
            with wave.open(str(p), 'w') as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
                wf.writeframes((audio * 32767).astype(np.int16).tobytes())
            self.player.play(str(p))
        except PermissionError:
            pass  # 偶尔文件被占用，静默跳过

    # ===== LLM =====

    def _call_llm(self, user_text: str) -> str:
        from openai import OpenAI
        key = self.cfg["deepseek_api_key"]
        log("debug", f"   [API Key: {key[:12]}...{key[-4:]}]")
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
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
            log("error", "  请先设置 DeepSeek API Key!")
            log("error", "  set DEEPSEEK_KEY=sk-xxx")
            log("error", "=" * 50)
            return

        self.asr.load()

        log("system", "")
        log("system", "=" * 50)
        log("system", "  🦉 Athena AI 语音助手 v3.1")
        log("system", "  喊「Athena / 雅典娜 / 阿斯娜」唤醒我")
        log("system", "  Ctrl+C 退出")
        log("system", "=" * 50)

        try:
            self._start_audio()
        except Exception as e:
            log("error", f"麦克风错误: {e}")
            return

        try:
            self._play_ding()
            log("system", "✅ 就绪")
        except Exception as e:
            log("error", f"扬声器错误: {e}")

        try:
            while not self._stop_flag.is_set():
                if self.state == "IDLE":
                    self._do_idle()
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

    def _do_idle(self):
        """IDLE: 监听唤醒词"""
        # 等待语音事件
        self._speech_event.wait(timeout=0.1)
        if not self._speech_event.is_set():
            return
        self._speech_event.clear()

        if not self._speech_queue:
            return

        audio, from_idle = self._speech_queue.pop(0)
        self._speech_queue.clear()

        if len(audio) < self.sample_rate * 0.3:
            return

        # 转录
        log("system", "👂 正在听...")
        text = self.asr.transcribe(audio)
        if not text:
            return

        # 检查唤醒词（拼音模糊匹配）
        is_wake, rest = self.wake_matcher.match(text)

        # 在终端显示（不管是不是唤醒词，让用户看到 Whisper 听到了什么）
        label = "唤醒检测" if is_wake else "未匹配"
        log("debug", f"   [{label}] 听到: '{text}'")

        if not is_wake:
            # 没匹配到，忽略
            return

        # 唤醒成功！
        log("wake", f"⚡ 唤醒！")
        log("system", "🔊 叮~ 请说话...")

        now = time.time()
        if now < self._wake_cooldown_until:
            return
        self._wake_cooldown_until = now + 1.5

        self._active_mode = True
        self._last_response_time = now

        if rest and len(rest) > 1:
            # 唤醒词 + 命令：直接处理
            log("you", f"你: {rest}")
            self._pending_command = rest
            self.state = "PROCESSING"
        else:
            # 只有唤醒词：进入聆听模式
            threading.Thread(target=self._play_ding, daemon=True).start()
            self._waiting_for_command = True
            self._wake_time = now
            self.vad.reset()
            self._speech_queue.clear()
            self._speech_event.clear()
            self.state = "LISTENING"

    def _do_listening(self):
        """LISTENING: 等待用户指令"""
        if self._waiting_for_command:
            if time.time() - self._wake_time > self.cfg["active_timeout"]:
                self._waiting_for_command = False
                log("system", "💤 超时回休眠")
                self.state = "IDLE"
                return

        self._speech_event.wait(timeout=0.1)
        if not self._speech_event.is_set():
            return
        self._speech_event.clear()

        if not self._speech_queue:
            return

        audio, _ = self._speech_queue.pop(0)
        self._speech_queue.clear()

        if len(audio) < self.sample_rate * 0.2:
            return

        log("system", "👂 正在理解...")
        text = self.asr.transcribe(audio)
        if not text or len(text.strip()) < 1:
            log("debug", "   没听清")
            return

        log("you", f"你: {text}")
        self._waiting_for_command = False

        # 退出指令
        if any(w in text for w in ["退下", "不用了", "没你的事了", "睡觉吧", "拜拜", "再见"]):
            log("athena", "好的，需要时叫我。")
            self.conversation_history = [self.conversation_history[0]]
            self._active_mode = False
            self.state = "IDLE"
            return

        self._pending_command = text
        self.state = "PROCESSING"

    def _do_processing(self):
        """PROCESSING: LLM 思考"""
        if not self._pending_command:
            self.state = "IDLE"
            return
        log("athena", "Athena: ")
        print("  ", end="", flush=True)  # LLM 流式输出缩进
        reply = self._call_llm(self._pending_command)
        self._pending_command = None
        if not reply:
            self.state = "IDLE"
            return
        self._pending_reply = reply
        self.state = "SPEAKING"

    def _do_speaking(self):
        """SPEAKING: TTS 播放"""
        if not self._pending_reply:
            self.state = "IDLE"
            return
        log("system", "🔊 正在说话...")
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
                log("wake", "⚡ 被打断！")
                self.player.stop()
                self._pending_reply = None
                self._active_mode = True
                self._last_response_time = time.time()
                self._waiting_for_command = True
                self._wake_time = time.time()
                self.vad.reset()
                self._speech_queue.clear()
                self._speech_event.clear()
                threading.Thread(target=self._play_ding, daemon=True).start()
                self.state = "LISTENING"
                return
            time.sleep(0.05)

        self._pending_reply = None
        self._last_response_time = time.time()
        self._active_mode = True
        self.tts.cleanup()
        self.vad.reset()
        self._speech_queue.clear()
        self._speech_event.clear()
        log("debug", "👂 继续听（无需唤醒词）...")
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
    AthenaVoiceAssistant(CONFIG).run()


if __name__ == "__main__":
    main()
