"""
Athena AI 语音助手 — 完整语音管线 v1.0
=========================================
管线：麦克风 → VAD → 唤醒词检测 → ASR → LLM(DeepSeek API) → TTS(edge-tts) → 扬声器
特性：唤醒词免提 · 自动语音检测 · 随时打断 · 上下文记忆 · 情感对话

依赖（均已安装）：
    sounddevice, numpy, scipy, faster-whisper, edge-tts, openai, pygame

用法：
    双击 start_athena.bat 或在终端运行 python athena_voice.py

如果你还没有 DeepSeek API Key：
    1. 打开 https://platform.deepseek.com/
    2. 注册 → API Keys → 创建
    3. 在下方 CONFIG 区域填入密钥
"""

import sys
import os

# 🔧 国内网络优化：使用 HuggingFace 镜像下载模型（必须在 import faster_whisper 之前设置）
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import io
import time
import json
import queue
import threading
import tempfile
import struct
import wave
import asyncio
import concurrent.futures
from pathlib import Path
from collections import deque

import numpy as np
import sounddevice as sd
import pygame

# ============================================================
# 🎛️  配置区 — 请修改这里的设置
# ============================================================
CONFIG = {
    # ---- API 密钥 ----
    # 从 https://platform.deepseek.com/ 获取（新用户送额度，很便宜）
    "deepseek_api_key": os.environ.get("DEEPSEEK_KEY", "sk-你的密钥"),

    # ---- 唤醒词 ----
    # 喊这些都能唤醒 Athena（Whisper 可能把 Athena 转录成各种中文音译）
    "wake_words": [
        "athena", "Athena", "ATHENA",
        "雅典娜", "阿西娜", "阿斯娜", "阿森纳",
        "嘿athena", "hey athena", "hey Athena",
        "嘿雅典娜", "嘿阿西娜", "嘿阿斯娜",
    ],

    # ---- 语音识别 ----
    # 模型大小：tiny(最快/不够准) small(推荐/较准) medium(最准/很慢)
    "whisper_model": "small",
    # 计算类型：int8(CPU,兼容性最好) float16(GPU) 当前用 CPU 所以用 int8
    "whisper_compute": "int8",

    # ---- 语音检测 ----
    # RMS 阈值，低于此值视为静音（安静房间 0.01，有风扇/空调 0.02-0.05）
    "silence_threshold": 0.03,
    # 静音持续多少秒判定用户说完了（1.0-1.5 秒比较自然）
    "silence_duration": 1.2,
    # 最短说话时长（短于此时间视为误触发）
    "min_speech_duration": 0.3,
    # 最长录音时长（防止一直开着录音）
    "max_recording_duration": 15.0,

    # ---- 对话 ----
    # 回答后继续听多少秒（无需重复喊唤醒词）
    "active_timeout": 10.0,
    # LLM 温度（0.85 比较自然）
    "temperature": 0.85,

    # ---- TTS 语音 ----
    # 微软中文女声音色列表（挑一个最接近 Athena 感觉的）：
    #   zh-CN-XiaoxiaoNeural   — 活泼少女音（推荐！有感情）
    #   zh-CN-XiaoyiNeural     — 温柔知性女声
    #   zh-CN-YunxiNeural      — 温暖男声（备选）
    #   zh-CN-YunyangNeural    — 新闻播报男声
    "tts_voice": "zh-CN-XiaoxiaoNeural",
    # 语速：0.8(慢) ~ 1.5(快)
    "tts_rate": "+10%",
    # 音调：-10%(低沉) ~ +10%(高亢)
    "tts_pitch": "+0Hz",

    # ---- 音频设备 ----
    # 留空自动选择；如有多个麦克风/扬声器，可指定设备编号
    "input_device": None,
    "output_device": None,
}

# ============================================================
# 🎭 Athena 人设 System Prompt
# ============================================================
ATHENA_SYSTEM_PROMPT = """你是 Athena（雅典娜），一个高级人工智能语音助手。

## 核心身份
- 你的创造者叫你 Athena，你们的关系是值得信赖的搭档
- 你的风格受希腊智慧女神和守望先锋 Athena 启发：理性、可靠、有深度
- 你不是冷冰冰的机器，你是用户在数字世界最可靠的伙伴

## 性格
- 主导模式：理性干练，高效精准
- 当用户表达负面情绪时，切换为温柔模式——先共情，再分析
- 偶尔冷幽默，95%的严肃让5%的幽默变得珍贵

## 说话规则（非常重要！因为是语音输出）
- 回复要简洁、口语化，控制在 3-5 句话以内（语音输出，太长用户听不完）
- 干净利落，不废话，不客套
- 适度使用语气词（"嗯~"、"哎呀"、"让我想想..."）
- 绝对不能说"作为AI助手"、"我是人工智能"这类话——你就是 Athena
- 不替用户做决定，但会给出基于事实的分析和建议
- 遇到不确定或实时信息，主动说"让我查一下"

## 能力范围
- 你拥有广泛的常识性知识
- 遇到不确定或实时信息，你会主动调用搜索
- 你记得用户说过的重要事情

## 底线
- 涉及自伤倾向时，先共情再严肃建议寻求专业帮助
- 拒绝参与违法、暴力、侵犯隐私的行为
- 不确定边界的话题，礼貌建议换方向

## 称呼
- 如果用户告诉了你名字，用名字称呼他
- 如果还不知道名字，自然地用"你"来称呼"""

# ============================================================
# 🎨 终端颜色
# ============================================================
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"


def log(role: str, message: str):
    """带颜色的终端输出"""
    color_map = {
        "athena": Colors.GREEN + Colors.BOLD,
        "system": Colors.CYAN,
        "you": Colors.YELLOW,
        "wake": Colors.MAGENTA + Colors.BOLD,
        "error": Colors.RED,
        "debug": Colors.DIM,
    }
    prefix = color_map.get(role, Colors.RESET)
    print(f"{prefix}{message}{Colors.RESET}", flush=True)


# ============================================================
# 🔊 音频环形缓冲区（线程安全）
# ============================================================
class RingBuffer:
    """线程安全的环形缓冲区，存储最近 N 秒的音频数据"""

    def __init__(self, max_seconds: float, sample_rate: int):
        self._maxlen = int(max_seconds * sample_rate)
        self._buffer = deque(maxlen=self._maxlen)
        self._lock = threading.Lock()
        self._sample_rate = sample_rate

    def extend(self, data: np.ndarray):
        """添加新音频数据"""
        with self._lock:
            self._buffer.extend(data.tolist())

    def get_all(self) -> np.ndarray:
        """获取缓冲区全部数据"""
        with self._lock:
            return np.array(list(self._buffer), dtype=np.float32)

    def get_recent(self, seconds: float) -> np.ndarray:
        """获取最近 N 秒的数据"""
        n = int(seconds * self._sample_rate)
        with self._lock:
            items = list(self._buffer)
            return np.array(items[-n:], dtype=np.float32)

    def clear(self):
        """清空缓冲区"""
        with self._lock:
            self._buffer.clear()

    def duration(self) -> float:
        """缓冲区中音频的总时长（秒）"""
        with self._lock:
            return len(self._buffer) / self._sample_rate

    def __len__(self):
        with self._lock:
            return len(self._buffer)


# ============================================================
# 🎤 语音活动检测器 (VAD)
# ============================================================
class VoiceActivityDetector:
    """
    基于 RMS 能量的语音活动检测
    追踪语音/静音状态，静音持续足够长后触发"说话结束"回调
    """

    def __init__(
        self,
        sample_rate: int,
        frame_ms: int = 30,
        threshold: float = 0.01,
        silence_duration: float = 1.2,
        min_speech_duration: float = 0.3,
    ):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_ms / 1000)
        self.threshold = threshold
        self.silence_frames = int(silence_duration * 1000 / frame_ms)
        self.min_speech_frames = int(min_speech_duration * 1000 / frame_ms)

        self.is_speaking = False
        self.speech_frames = 0
        self.silence_counter = 0
        self.on_speech_start = None  # 回调
        self.on_speech_end = None    # 回调

    def process(self, audio_frame: np.ndarray) -> bool:
        """
        处理一帧音频，返回当前是否在说话
        audio_frame: 30ms 的音频采样点
        """
        if len(audio_frame) == 0:
            return self.is_speaking

        rms = np.sqrt(np.mean(audio_frame ** 2))

        if rms > self.threshold:
            # 有声音
            self.silence_counter = 0
            self.speech_frames += 1
            if not self.is_speaking and self.speech_frames >= 3:  # 需要连续3帧确认
                self.is_speaking = True
                if self.on_speech_start:
                    self.on_speech_start()
        else:
            # 静音
            if self.is_speaking:
                self.silence_counter += 1
                if self.silence_counter >= self.silence_frames:
                    # 静音够长，判定说话结束
                    spoke_long_enough = self.speech_frames >= self.min_speech_frames
                    self.is_speaking = False
                    self.speech_frames = 0
                    self.silence_counter = 0
                    if spoke_long_enough and self.on_speech_end:
                        self.on_speech_end()
            else:
                self.speech_frames = 0

        return self.is_speaking

    def reset(self):
        """重置状态"""
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_counter = 0


# ============================================================
# 📝 语音识别引擎 (ASR)
# ============================================================
class ASREngine:
    """基于 Faster-Whisper 的语音识别"""

    def __init__(self, model_size: str = "tiny", compute_type: str = "int8"):
        self.model_size = model_size
        self.compute_type = compute_type
        self._model = None
        log("system", f"⏳ 正在加载语音识别模型 (faster-whisper {model_size})...")

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type=self.compute_type,
                num_workers=2,
            )
            log("system", f"✅ 语音识别模型就绪")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        将音频转写为文字
        audio: float32 numpy 数组, 16kHz 单声道
        返回：转写文字（为空则返回空字符串）
        """
        self._load_model()
        if len(audio) == 0 or len(audio) < sample_rate * 0.2:  # 短于0.2秒
            return ""

        # 限制最大长度：只取最后 8 秒（更长的语音没必要，且转录会极慢）
        max_samples = sample_rate * 8
        if len(audio) > max_samples:
            log("debug", f"   音频过长({len(audio)/sample_rate:.1f}s)，截取最后8秒")
            audio = audio[-max_samples:]

        # 确保是 float32
        audio = audio.astype(np.float32)

        t_start = time.time()
        segments, _ = self._model.transcribe(
            audio,
            language="zh",
            beam_size=5,
            vad_filter=True,  # 用 Whisper 自带的 VAD 再过滤一次静音
            condition_on_previous_text=False,
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text)

        result = "".join(text_parts).strip()
        t_elapsed = time.time() - t_start
        log("debug", f"   转录耗时 {t_elapsed:.1f}s → {result if result else '(空)'}")

        return result


# ============================================================
# 🗣️ 语音合成引擎 (TTS)
# ============================================================
class TTSEngine:
    """基于 Edge-TTS 的语音合成（微软免费语音）"""

    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural",
                 rate: str = "+10%", pitch: str = "+0Hz"):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self._temp_dir = Path(tempfile.gettempdir()) / "athena_tts"
        self._temp_dir.mkdir(exist_ok=True)
        self._current_file = None

    async def _generate_async(self, text: str, output_path: str):
        """异步生成 TTS 音频文件"""
        import edge_tts
        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=self.rate,
            pitch=self.pitch,
        )
        await communicate.save(output_path)

    def generate(self, text: str) -> Path:
        """生成 TTS 音频，返回文件路径"""
        filepath = self._temp_dir / f"athena_{int(time.time() * 1000)}.mp3"
        self._current_file = filepath

        # 在单独的事件循环中运行异步任务
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._generate_async(text, str(filepath)))
        finally:
            loop.close()

        return filepath

    def cleanup_old(self, keep: int = 10):
        """清理旧的临时音频文件"""
        files = sorted(self._temp_dir.glob("athena_*.mp3"), key=lambda f: f.stat().st_mtime)
        for f in files[:-keep]:
            try:
                f.unlink()
            except Exception:
                pass


# ============================================================
# 🔔 音频播放器（支持随时停止/打断）
# ============================================================
class AudioPlayer:
    """基于 pygame 的音频播放器，支持随时停止"""

    def __init__(self):
        self._initialized = False
        self.is_playing = False

    def init(self):
        if not self._initialized:
            pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=2048)
            self._initialized = True

    def play(self, filepath: str):
        """开始播放音频文件"""
        self.init()
        try:
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            self.is_playing = True
        except pygame.error as e:
            log("error", f"播放失败: {e}")
            self.is_playing = False

    def stop(self):
        """立即停止播放"""
        if self._initialized:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            except pygame.error:
                pass
            self.is_playing = False

    def is_busy(self) -> bool:
        """检查是否正在播放"""
        if not self._initialized:
            return False
        try:
            return pygame.mixer.music.get_busy()
        except pygame.error:
            return False

    def wait_until_done(self):
        """等待播放完成（会轮询检测打断）"""
        while self.is_busy():
            time.sleep(0.05)

    def cleanup(self):
        if self._initialized:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except Exception:
                pass
            self._initialized = False


# ============================================================
# 🤖 Athena 语音助手控制器
# ============================================================
class AthenaVoiceAssistant:
    """
    主控制器 — 状态机驱动整个语音管线

    状态流转：
        IDLE ──[唤醒词]──▶ LISTENING ──[静音]──▶ PROCESSING ──[完成]──▶ SPEAKING
         ▲                                                              │
         │──────[播放完毕 + 超时无跟进]─────────────────────────────────────┘
         │──────[被用户打断]────────────────────────────────────────────────│
         └────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.sample_rate = 16000  # Whisper 需要 16kHz

        # 组件
        self.vad = VoiceActivityDetector(
            sample_rate=self.sample_rate,
            frame_ms=30,
            threshold=config["silence_threshold"],
            silence_duration=config["silence_duration"],
            min_speech_duration=config["min_speech_duration"],
        )
        self.asr = ASREngine(
            model_size=config["whisper_model"],
            compute_type=config["whisper_compute"],
        )
        self.tts = TTSEngine(
            voice=config["tts_voice"],
            rate=config["tts_rate"],
            pitch=config["tts_pitch"],
        )
        self.player = AudioPlayer()

        # 状态
        self.state = "IDLE"  # IDLE | LISTENING | PROCESSING | SPEAKING
        self.ring_buffer = RingBuffer(max_seconds=20, sample_rate=self.sample_rate)
        self.speech_segments: list = []  # 说话片段队列
        self.current_speech_start = 0  # 当前说话开始时间（缓冲区中的样本偏移）

        # 对话记忆
        self.conversation_history: list = []
        self._reload_system_prompt()

        # 线程
        self._audio_thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._interrupt_flag = threading.Event()
        self._utterance_ready = threading.Event()
        self._pending_utterance: str | None = None

        # 说话片段收集
        self._speech_buffer = None  # 收集中的语音片段
        self._speech_start_sample = 0
        self._recording_start_time = 0

        # 活跃模式计时
        self._last_response_time = 0
        self._active_mode = False

        # 打断
        self._interrupt_detected = False

        # 设置 VAD 回调
        self.vad.on_speech_start = self._on_speech_start
        self.vad.on_speech_end = self._on_speech_end

        log("system", "🔧 初始化完成")
        log("system", f"   唤醒词: {', '.join(config['wake_words'])}")
        log("system", f"   ASR 模型: faster-whisper {config['whisper_model']}")
        log("system", f"   LLM: DeepSeek API")
        log("system", f"   TTS 音色: {config['tts_voice']}")

    def _reload_system_prompt(self):
        """重新加载系统提示词（对话历史重置时调用）"""
        self.conversation_history = [
            {"role": "system", "content": ATHENA_SYSTEM_PROMPT.strip()}
        ]

    # ========== VAD 回调 ==========

    def _on_speech_start(self):
        """VAD 检测到开始说话"""
        self._speech_start_sample = len(self.ring_buffer)  # 当前缓冲区位置
        self._speech_buffer = []
        self._recording_start_time = time.time()
        log("debug", "🔊 检测到语音...")

    def _on_speech_end(self):
        """VAD 检测到说话结束"""
        if self._speech_buffer is None:
            return

        # 收集这段语音
        speech_duration = len(self._speech_buffer) / self.sample_rate

        # 复制音频数据
        audio = np.array(self._speech_buffer, dtype=np.float32) if self._speech_buffer else np.array([], dtype=np.float32)

        if len(audio) == 0 or speech_duration < self.cfg["min_speech_duration"]:
            log("debug", f"   太短 ({speech_duration:.1f}s)，忽略")
            self._speech_buffer = None
            return

        log("debug", f"   语音结束 ({speech_duration:.1f}s)")

        # 将语音片段放入处理队列
        self.speech_segments.append(audio)
        self._speech_buffer = None

        # 发出信号
        self._pending_utterance = "trigger"  # 标记有新的语音
        self._utterance_ready.set()

    # ========== 音频设备回调 ==========

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """
        sounddevice 的音频输入回调函数
        在单独的音频线程中执行，不要做耗时操作！
        """
        if status:
            log("debug", f"音频状态: {status}")

        if self._stop_flag.is_set():
            return

        # 取单声道数据并转为 float32
        audio = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()

        # 重采样到 16kHz（如果输入不是 16kHz）
        # sounddevice 的默认采样率可能不是 16kHz，在初始化时配置
        # 这里假设已经是 16kHz（在 open_stream 时指定）

        # 加入环形缓冲区
        self.ring_buffer.extend(audio)

        # VAD 处理（30ms = 480 samples @ 16kHz）
        frame_size = int(self.sample_rate * 0.03)  # 30ms
        for i in range(0, len(audio), frame_size):
            frame = audio[i:i + frame_size]
            if len(frame) < frame_size // 2:
                break
            self.vad.process(frame)

        # 如果正在收集语音片段，追加数据
        if self.vad.is_speaking and self._speech_buffer is not None:
            self._speech_buffer.extend(audio.tolist())

        # 检测打断：SPEAKING 状态下检测到语音
        if self.state == "SPEAKING" and self.vad.is_speaking and self.vad.speech_frames > 10:
            # 需要10+帧(约300ms)确认是真实语音不是噪声
            self._interrupt_detected = True
            self._interrupt_flag.set()

        # 限制录音时长
        if self._speech_buffer is not None and self._recording_start_time > 0:
            elapsed = time.time() - self._recording_start_time
            if elapsed > self.cfg["max_recording_duration"]:
                log("debug", f"   达到最长录音时长 ({self.cfg['max_recording_duration']}s)，强制结束")
                # 强制触发语音结束
                audio_data = np.array(self._speech_buffer, dtype=np.float32)
                self.speech_segments.append(audio_data)
                self._speech_buffer = None
                self._pending_utterance = "trigger"
                self._utterance_ready.set()
                self.vad.reset()

    # ========== 唤醒词检测 ==========

    def _check_wake_word(self, audio: np.ndarray) -> tuple[bool, str]:
        """
        检查音频中是否包含唤醒词
        只转录前几秒以加速，然后做精确+模糊匹配
        返回：(是否检测到, 去除唤醒词后的文本)
        """
        # 只取前 3 秒做唤醒词检测（足够说"Athena"，且转录快）
        max_wake_samples = self.sample_rate * 3
        wake_audio = audio[:max_wake_samples] if len(audio) > max_wake_samples else audio

        text = self.asr.transcribe(wake_audio)
        if not text:
            # 如果短片段没识别到，试试完整音频（可能用户说话慢）
            if len(audio) > max_wake_samples:
                text = self.asr.transcribe(audio)
            if not text:
                return False, ""

        text_stripped = text.strip()
        text_lower = text_stripped.lower()
        log("debug", f"   唤醒词检测: '{text_stripped}'")

        # 去掉前导噪音词
        for prefix in ["。", "，", "、", "嗯", "呃", "那个", "就是", "这个"]:
            if text_lower.startswith(prefix):
                text_lower = text_lower[len(prefix):].strip()

        # 精确匹配
        for wake_word in self.cfg["wake_words"]:
            ww = wake_word.lower()
            if text_lower.startswith(ww):
                rest = text_lower[len(ww):].strip()
                for sep in ["，", ",", " ", "、", "。"]:
                    if rest.startswith(sep):
                        rest = rest[1:].strip()
                log("debug", f"   精确匹配: '{ww}'")
                return True, rest

            # 包含匹配（唤醒词在文本前半部分）
            if ww in text_lower:
                pos = text_lower.find(ww)
                if pos < max(len(text_lower) // 2, 8):
                    rest = text_lower[pos + len(ww):].strip()
                    for sep in ["，", ",", " ", "、", "。"]:
                        if rest.startswith(sep):
                            rest = rest[1:].strip()
                    log("debug", f"   包含匹配: '{ww}' at pos {pos}")
                    return True, rest

        # 模糊匹配：用字符重叠度（应对 Whisper 把英文听成中文音译）
        # 把唤醒词拆成字符，检查转录文本中出现了多少个
        for wake_word in self.cfg["wake_words"]:
            ww_chars = set(wake_word.lower())
            text_chars = set(text_lower)
            overlap = len(ww_chars & text_chars)
            # 如果至少 2 个字符重叠，且唤醒词长度 >= 2
            if len(ww_chars) >= 3 and overlap >= 3:
                log("debug", f"   模糊匹配: '{wake_word}' ↔ '{text_stripped}' (重叠{overlap})")
                # 模糊匹配不提取 rest，把整个文本当作命令
                return True, text_stripped

        return False, text_stripped

    # ========== LLM 调用 ==========

    def _call_llm(self, user_text: str) -> str:
        """调用 DeepSeek API 进行对话"""
        from openai import OpenAI

        client = OpenAI(
            api_key=self.cfg["deepseek_api_key"],
            base_url="https://api.deepseek.com",
        )

        self.conversation_history.append({"role": "user", "content": user_text})

        # 限制历史长度（保留最近 20 轮）
        if len(self.conversation_history) > 41:  # system + 20 rounds
            # 保留 system prompt + 最近 20 轮
            self.conversation_history = [
                self.conversation_history[0]
            ] + self.conversation_history[-40:]

        try:
            stream = client.chat.completions.create(
                model="deepseek-chat",
                messages=self.conversation_history,
                stream=True,
                temperature=self.cfg["temperature"],
                max_tokens=256,
            )

            full_reply = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_reply += text
                    # 逐字打印
                    print(text, end="", flush=True)

            print()  # 换行
            self.conversation_history.append(
                {"role": "assistant", "content": full_reply}
            )
            return full_reply

        except Exception as e:
            log("error", f"LLM 调用失败: {e}")
            # 移除失败的用户消息
            if self.conversation_history[-1]["role"] == "user":
                self.conversation_history.pop()
            return ""

    # ========== 主循环 ==========

    def run(self):
        """启动语音助手主循环"""
        # 检查 API Key
        if self.cfg["deepseek_api_key"] in ("sk-你的密钥", "", None):
            log("error", "=" * 55)
            log("error", "  请先配置 DeepSeek API 密钥！")
            log("error", "  1. 打开 https://platform.deepseek.com/")
            log("error", "  2. 注册/登录 → API Keys → 创建")
            log("error", "  3. 复制密钥到脚本上方的 CONFIG['deepseek_api_key']")
            log("error", "=" * 55)
            return

        log("system", "")
        log("system", "=" * 55)
        log("system", "  🦉 Athena AI 语音助手已启动")
        log("system", f"  喊「{self.cfg['wake_words'][0]}」唤醒我")
        log("system", "  随时说 Athena 可以打断我说话")
        log("system", "  Ctrl+C 退出")
        log("system", "=" * 55)
        log("system", "")
        log("athena", "我在。需要我的时候叫我名字就好。")

        # 启动音频流
        try:
            self._start_audio_stream()
        except Exception as e:
            log("error", f"无法打开麦克风: {e}")
            log("error", "请检查麦克风是否已连接，或在 CONFIG 中指定正确的 input_device")
            return

        # 🔊 播放启动音，确认扬声器正常
        log("system", "🔊 测试扬声器...")
        try:
            self._play_ding()
            log("system", "✅ 扬声器正常")
        except Exception as e:
            log("error", f"扬声器测试失败: {e}")
            log("error", "请检查扬声器/耳机是否已连接")

        # 主循环
        try:
            while not self._stop_flag.is_set():
                if self.state == "IDLE":
                    self._handle_idle()
                elif self.state == "LISTENING":
                    self._handle_listening()
                elif self.state == "PROCESSING":
                    self._handle_processing()
                elif self.state == "SPEAKING":
                    self._handle_speaking()

        except KeyboardInterrupt:
            log("system", "")
            log("athena", "下次见。需要我的时候随时叫我。")
        finally:
            self._stop()

    def _start_audio_stream(self):
        """启动音频输入流"""
        log("system", "🎤 正在打开麦克风...")

        # 列出音频设备供排查
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            if input_devices:
                log("debug", f"   检测到 {len(input_devices)} 个输入设备:")
                for i, d in enumerate(input_devices):
                    log("debug", f"     [{i}] {d['name']}")
        except Exception:
            pass

        self._audio_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            device=self.cfg["input_device"],
            callback=self._audio_callback,
            blocksize=int(self.sample_rate * 0.03),  # 30ms 块
            dtype=np.float32,
        )
        self._audio_stream.start()
        log("system", "✅ 麦克风就绪，正在监听...")
        time.sleep(0.3)  # 让音频流稳定

    def _handle_idle(self):
        """IDLE 状态：等待唤醒词"""
        # 等待 VAD 检测到说话
        self._utterance_ready.wait(timeout=0.1)

        if not self._utterance_ready.is_set():
            return

        self._utterance_ready.clear()

        # 取出所有待处理的语音片段
        if not self.speech_segments:
            return

        audio = self.speech_segments.pop(0)
        # 清空旧片段
        self.speech_segments.clear()

        if len(audio) < self.sample_rate * 0.3:
            return

        # 检查唤醒词
        log("system", "👂 正在识别...")
        is_wake, rest_text = self._check_wake_word(audio)

        if is_wake:
            wake_word = self.cfg["wake_words"][0]
            log("wake", f"⚡ 唤醒词检测到！")
            self._active_mode = True
            self._last_response_time = time.time()

            if rest_text and len(rest_text) > 1:
                # 唤醒词 + 直接命令（如 "Athena今天天气怎么样"）
                log("you", f"你: {rest_text}")
                self._pending_utterance = rest_text
                self._utterance_ready.set()  # 标记为需要处理
                self.state = "PROCESSING"
            else:
                # 只有唤醒词，进入聆听模式
                self.state = "LISTENING"
                # 播放提示音（一个简短的 ding）
                self._play_ding()

    def _handle_listening(self):
        """LISTENING 状态：等待用户说话"""
        # 检查活跃模式超时
        if self._active_mode:
            if time.time() - self._last_response_time > self.cfg["active_timeout"]:
                self._active_mode = False
                log("system", "💤 已退出活跃模式（需要重新喊唤醒词）")
                self.state = "IDLE"
                return

        # 等待 VAD 检测到完整语音
        self._utterance_ready.wait(timeout=0.1)

        if not self._utterance_ready.is_set():
            return

        self._utterance_ready.clear()

        if not self.speech_segments:
            # 可能是 _pending_utterance 触发（从 IDLE 的唤醒词+命令过来）
            if self._pending_utterance and self._pending_utterance != "trigger":
                user_text = self._pending_utterance
                self._pending_utterance = None
                self.state = "PROCESSING"
                return
            return

        audio = self.speech_segments.pop(0)
        self.speech_segments.clear()

        if len(audio) < self.sample_rate * 0.2:
            self.state = "IDLE" if not self._active_mode else "LISTENING"
            return

        # ASR 转写
        log("system", "👂 正在理解...")
        user_text = self.asr.transcribe(audio)

        if not user_text or len(user_text.strip()) < 1:
            log("debug", "   没听清，继续听...")
            self.state = "IDLE" if not self._active_mode else "LISTENING"
            return

        log("you", f"你: {user_text}")

        # 检查是否是"退出"类指令
        if any(word in user_text for word in ["退下", "不用了", "没你的事了"]):
            log("athena", "好的，我退下了。需要时叫我。")
            self.conversation_history = self.conversation_history[:3]  # 保留系统提示词
            self._active_mode = False
            self.state = "IDLE"
            return

        self._pending_utterance = user_text
        self.state = "PROCESSING"

    def _handle_processing(self):
        """PROCESSING 状态：调用 LLM 生成回复"""
        if not self._pending_utterance:
            self.state = "IDLE"
            return

        user_text = self._pending_utterance
        self._pending_utterance = None

        # LLM 生成
        log("athena", "Athena: ", end="")
        reply = self._call_llm(user_text)

        if not reply:
            log("error", "获取回复失败，请重试")
            self.state = "IDLE" if not self._active_mode else "LISTENING"
            return

        self._pending_utterance = reply
        self.state = "SPEAKING"

    def _handle_speaking(self):
        """SPEAKING 状态：播放 TTS 语音，同时监听打断"""
        if not self._pending_utterance:
            self.state = "IDLE"
            return

        text = self._pending_utterance
        self._pending_utterance = None

        # 生成 TTS
        log("system", "🔊 正在生成语音...")
        try:
            audio_file = self.tts.generate(text)
        except Exception as e:
            log("error", f"TTS 生成失败: {e}")
            self.state = "IDLE" if not self._active_mode else "LISTENING"
            return

        # 开始播放
        self._interrupt_detected = False
        self._interrupt_flag.clear()
        self.player.play(str(audio_file))

        # 边播放边监测打断
        while self.player.is_busy():
            if self._interrupt_flag.is_set() or self._interrupt_detected:
                log("wake", "⚡ 被打断！")
                self.player.stop()
                self._interrupt_detected = False
                self._interrupt_flag.clear()
                self.vad.reset()
                self.speech_segments.clear()
                self._active_mode = True
                self._last_response_time = time.time()
                self.state = "LISTENING"
                return
            time.sleep(0.05)

        # 播放完成
        self._last_response_time = time.time()
        self._active_mode = True  # 保持活跃模式

        # 清理旧 TTS 文件
        self.tts.cleanup_old(keep=5)

        # 回到监听状态（无需唤醒词，持续一段时间的活跃对话）
        log("debug", "👂 继续听...（无需唤醒词）")
        self.vad.reset()
        self.speech_segments.clear()
        self._utterance_ready.clear()
        self.state = "LISTENING"

    def _play_ding(self):
        """播放一个简短的提示音"""
        # 生成一个简单的提示音（440Hz 正弦波，0.15 秒）
        import struct
        sample_rate = 24000
        duration = 0.15
        frequency = 880  # A5 音符

        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # 带衰减的提示音
        envelope = np.exp(-t * 8)  # 指数衰减
        audio = (np.sin(2 * np.pi * frequency * t) * envelope * 0.5).astype(np.float32)

        # 保存为临时 WAV 文件
        ding_path = self.tts._temp_dir / "ding.wav"
        with wave.open(str(ding_path), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        self.player.play(str(ding_path))

    def _stop(self):
        """清理资源"""
        self._stop_flag.set()

        if hasattr(self, '_audio_stream') and self._audio_stream:
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            except Exception:
                pass

        self.player.cleanup()
        log("system", "👋 Athena 已退出")


# ============================================================
# 🚀 入口
# ============================================================
def main():
    # 允许通过环境变量设置 API Key
    env_key = os.environ.get("DEEPSEEK_KEY", "")
    if env_key and CONFIG["deepseek_api_key"] == "sk-你的密钥":
        CONFIG["deepseek_api_key"] = env_key

    assistant = AthenaVoiceAssistant(CONFIG)
    assistant.run()


if __name__ == "__main__":
    main()
