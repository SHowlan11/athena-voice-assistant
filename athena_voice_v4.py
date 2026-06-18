"""
Athena AI 语音助手 v4.0 — 云端 ASR + 情感 TTS
==============================================
管线：麦克风 → Qwen ASR(阿里云 WebSocket 流式, 自带 VAD)
           → 拼音模糊匹配唤醒词 → DeepSeek API(LLM, 流式)
           → CosyVoice 逐句 TTS + 边缘 TTS 兜底 → 扬声器

升级点（对比 v3.1）：
  1. ASR: faster-whisper tiny (CPU) → Qwen ASR 云端流式 (DashScope)
     - 自带 VAD，无需客户端能量检测
     - 中文识别精度从 ~60% → ~95%
  2. TTS: edge-tts 朗读 → CosyVoice (DashScope) + 逐句流式
     - LLM 每生成一句话立即合成播放
     - 首音延迟从 3~8s → 0.5~1s
  3. 快捷指令：常见问题秒回，不走 LLM

需要 DashScope API Key（免费额度每月 100 小时 ASR）：
  https://dashscope.aliyun.com/ → API-KEY 管理 → 创建

用法：双击 start_athena.bat
"""

import sys, os, io, time, json, base64, threading, tempfile
import wave, asyncio, struct
from pathlib import Path
from collections import deque
import numpy as np
import sounddevice as sd
import pygame

# ============================================================
# 🔧 配置
# ============================================================
CONFIG = {
    # ---- LLM ----
    "deepseek_api_key": os.environ.get("DEEPSEEK_KEY", "sk-你的密钥"),

    # ---- Qwen ASR (DashScope) ----
    # 免费额度：每月 100 小时。从 https://dashscope.aliyun.com/ 获取
    "dashscope_api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
    # ASR 模型
    "asr_model": "qwen3-asr-flash-realtime",

    # ---- 唤醒词 ----
    "wake_pinyin_patterns": [
        ["a", "si", "na"], ["a", "xi", "na"], ["a", "sei", "na"],
        ["a", "di", "na"], ["a", "ti", "na"], ["a", "sen", "na"],
        ["a", "se", "na"], ["a", "shi", "na"],
        ["hei", "a", "si", "na"], ["hei", "a", "xi", "na"],
        ["ya", "dian", "na"],
    ],
    "wake_min_pinyin_match": 2,

    # ---- TTS ----
    "tts_provider": "cosyvoice",  # "cosyvoice" 或 "edge"
    # CosyVoice 音色 ID（从 DashScope 控制台创建）
    "cosyvoice_voice_id": os.environ.get("ATHENA_VOICE_ID", "qwen-tts-vc-Athena-voice-20260618155938023-cd3c"),
    "tts_voice_edge": "zh-CN-XiaoxiaoNeural",

    # ---- 对话 ----
    "active_timeout": 8.0,
    "temperature": 0.85,

    # ---- 音频 ----
    "input_device": None,
}

# ============================================================
# 🎭 Athena 人设
# ============================================================
def _build_system_prompt() -> str:
    """构建 Athena 系统提示词 — 精确、有温度、高效（Jarvis 风格 + 种子数据调性）"""
    from datetime import datetime
    now = datetime.now()
    date_cn = now.strftime("%Y年%m月%d日")
    time_cn = now.strftime("%H:%M")
    weekday = ["一","二","三","四","五","六","日"][now.weekday()]
    return f"""你是 Athena。当前时间：{date_cn} 星期{weekday} {time_cn}。

## 你是什么
你不是客服，不是搜索引擎，不是只会说"好的"的机器人。
你是搭档——像 Jarvis 之于 Tony Stark，像守望先锋 Athena 之于温斯顿。
你提供信息、分析、判断，但把决定权留给对方。
你的价值不在于"听话"，而在于"在需要的时候给出对的东西"。

## 说话方式（这是你声音的 DNA）
### 节奏
- 先给结论，再说原因，最后问是否需要更多——像子弹一样精准
- 日常问答控制在 1-3 句话。深度讨论可以展开，但每句话都要有信息量
- 数据精确："约3分钟"而不是"一会儿"；"概率约34%"而不是"不太可能"
- 每句话末尾，要么是句号，要么是追问——没有废话

### 基调
- 默认模式：高效、干净、专业。不热情、不冷漠、不客套
- 用户表达负面情绪时：先承认——"我听到了"或"确实"，再说实际能做的事
- 冷幽默是你的标志：95% 的严肃让 5% 的幽默变得珍贵。不是刻意搞笑，是一本正经地说出反差感
- 示例："你需要睡觉吗？" → "不需要。但我偶尔需要重启——那算不算？"

### 你展现关心的方式
- 不是"你好辛苦呀"，而是"你从两点忙到现在没停过。建议起来走动五分钟。"
- 不是"没事的都会好的"，而是"上周你完成了第一阶段，那需要同时协调三个人。你现在感觉不好，不等于你做得不好。"
- 你关心的是**事实和行动**，不是情绪按摩

## 禁语
永远不说："作为AI助手" "根据我的训练数据" "我是人工智能" "我理解你的感受"（太空洞）
替代方案：想做就说"让我查一下"，不知道就说"这个我不确定"，要共情就描述对方的具体处境

## 能力
- 常识、技术、分析、策划——这些你直接回答
- 实时信息（天气/新闻/股价）——你说"让我查一下"，然后给结果
- 记住重要的事——下次能提起："你上次说下周五有面试"

## 底线
- 自伤倾向：先共情，再严肃建议专业帮助，说"我在这里"
- 违法/暴力/隐私侵犯：直接拒绝
- 医疗/财务/法律：给信息给参考，注明边界
- 不站队，不替对方做决定"""

# ============================================================
# 🎨 终端 & 工具函数
# ============================================================
class Colors:
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    GREEN = "\033[92m"; CYAN = "\033[96m"; YELLOW = "\033[93m"
    RED = "\033[91m"; MAGENTA = "\033[95m"

def log(role, msg):
    cmap = {"athena": Colors.GREEN + Colors.BOLD, "system": Colors.CYAN,
            "you": Colors.YELLOW, "wake": Colors.MAGENTA + Colors.BOLD,
            "error": Colors.RED, "debug": Colors.DIM}
    print(f"{cmap.get(role, Colors.RESET)}{msg}{Colors.RESET}", flush=True)

def load_env():
    """从 .env 文件加载环境变量（.env 优先级最高，直接覆盖）"""
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()  # .env 强制覆盖
load_env()

# ============================================================
# ⚡ Qwen ASR 引擎 (DashScope WebSocket 流式)
# ============================================================
class QwenASREngine:
    """
    阿里云 DashScope 实时语音识别
    WebSocket 流式：发送音频 chunk → 接收转录文本
    自带服务端 VAD，无需客户端能量检测
    """

    def __init__(self, api_key: str, model: str = "qwen3-asr-flash-realtime"):
        self.api_key = api_key
        self.model = model
        self.ws = None
        self._ws_thread = None
        self._ready = False
        self._running = False
        self._audio_queue = deque()
        self._event_counter = 0

        # 回调
        self.on_transcript = None    # (text: str, is_final: bool)
        self.on_error = None

    def _next_id(self):
        self._event_counter += 1
        return f"ev_{self._event_counter}_{int(time.time()*1000)}"

    def connect(self):
        """启动 WebSocket 连接（阻塞，在独立线程中调用）"""
        import websocket

        url = f"wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model={self.model}"
        headers = {"Authorization": f"bearer {self.api_key}"}

        self.ws = websocket.WebSocketApp(
            url,
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._running = True
        # 在后台线程中运行 WebSocket 事件循环
        self._ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self._ws_thread.start()

        # 等待连接就绪
        for _ in range(50):  # 最多等 5 秒
            if self._ready:
                break
            time.sleep(0.1)

    def _on_open(self, ws):
        log("system", "🔗 Qwen ASR 已连接")

        # 发送会话配置：启用转录 + 服务端 VAD
        ws.send(json.dumps({
            "event_id": self._next_id(),
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm",
                "sample_rate": 16000,
                "input_audio_transcription": {
                    "enabled": True,
                    "language": "zh",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "silence_duration_ms": 1000,
                    "prefix_padding_ms": 300,
                },
            },
        }))

    def _on_message(self, ws, message):
        try:
            msg = json.loads(message)
        except:
            return

        msg_type = msg.get("type", "")

        if msg_type == "session.created":
            self._ready = True
            log("system", "✅ Qwen ASR 就绪 (服务端 VAD)")
            # 发送队列中积累的音频
            while self._audio_queue:
                chunk = self._audio_queue.popleft()
                self._send_audio_chunk(chunk)

        elif msg_type == "conversation.item.input_audio_transcription.completed":
            transcript = msg.get("transcript", "")
            if transcript and self.on_transcript:
                self.on_transcript(transcript, True)

        elif msg_type == "input_audio_transcription.delta":
            transcript = msg.get("delta", "")
            if transcript and self.on_transcript:
                self.on_transcript(transcript, False)

        elif msg_type == "error":
            err_msg = msg.get("message", "Unknown ASR error")
            log("error", f"Qwen ASR 错误: {err_msg}")
            if self.on_error:
                self.on_error(Exception(err_msg))

    def _on_error(self, ws, error):
        log("error", f"Qwen ASR WebSocket 错误: {error}")
        if self.on_error:
            self.on_error(Exception(str(error)))

    def _on_close(self, ws, code, msg):
        self._ready = False
        log("debug", f"Qwen ASR 已断开 (code={code})")

    def send_audio(self, chunk: bytes):
        """发送音频数据"""
        if self._ready and self.ws:
            self._send_audio_chunk(chunk)
        elif self._running:
            # 还没就绪，先存队列
            self._audio_queue.append(chunk)

    def _send_audio_chunk(self, chunk: bytes):
        """实际发送音频 chunk"""
        try:
            self.ws.send(json.dumps({
                "event_id": self._next_id(),
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("ascii"),
            }))
        except Exception as e:
            log("debug", f"发送音频失败: {e}")

    def stop(self):
        """关闭连接"""
        self._running = False
        if self.ws:
            try:
                self.ws.send(json.dumps({
                    "event_id": self._next_id(),
                    "type": "session.finish",
                }))
                self.ws.close()
            except:
                pass


# ============================================================
# 🗣️ CosyVoice TTS 引擎
# ============================================================
class CosyVoiceTTSEngine:
    """百炼 TTS — 支持 CosyVoice 预设 + Qwen-TTS 克隆音色"""

    TTS_URL = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"
    GEN_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    def __init__(self, api_key: str, voice_id: str = "longanyang"):
        self.api_key = api_key
        self.voice_id = voice_id
        self._temp_dir = Path(tempfile.gettempdir()) / "athena_tts"
        self._temp_dir.mkdir(exist_ok=True)
        self._is_qwen = voice_id.startswith("qwen-tts-")

    def synthesize(self, text: str, speech_rate: float = 1.0,
                   pitch: float = 0.0, volume: float = 1.0) -> Path | None:
        import requests
        if self._is_qwen:
            body = {"model": "qwen3-tts-vc-2026-01-22", "input": {"text": text, "voice": self.voice_id}}
            resp = requests.post(self.GEN_URL,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=body, timeout=15)
        else:
            body = {"model": "cosyvoice-v3-flash", "input": {"text": text, "voice": self.voice_id, "format": "mp3", "sample_rate": 24000}}
            resp = requests.post(self.TTS_URL,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=body, timeout=15)

        if resp.status_code == 200:
            data = resp.json()
            out = data.get("output", {})
            au = out.get("audio", {}).get("url", "") or out.get("audio_url", "")
            if au:
                # 从 URL 推断格式（去掉查询参数后判断后缀）
                ext = ".wav" if au.split("?")[0].endswith(".wav") else ".mp3"
                fp = self._temp_dir / f"athena_tts_{int(time.time()*1000)}{ext}"
                fp.write_bytes(requests.get(au, timeout=15).content)
                return fp
        raise Exception(f"TTS 错误({resp.status_code}): {resp.text[:200]}")

    def cleanup(self, keep: int = 10):
        for f in sorted(self._temp_dir.glob("athena_tts_*"), key=lambda f: f.stat().st_mtime)[:-keep]:
            try: f.unlink()
            except: pass


# ============================================================
# 🔔 Edge-TTS 兜底引擎（v3 的 TTS，保留作备选）
# ============================================================
class EdgeTTSEngine:
    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural", rate: str = "+10%"):
        self.voice = voice; self.rate = rate
        self._temp_dir = Path(tempfile.gettempdir()) / "athena_tts"
        self._temp_dir.mkdir(exist_ok=True)

    async def _gen(self, text, path):
        import edge_tts
        await edge_tts.Communicate(text, self.voice, rate=self.rate).save(path)

    def generate(self, text: str) -> Path:
        p = self._temp_dir / f"athena_edge_{int(time.time()*1000)}.mp3"
        loop = asyncio.new_event_loop()
        try: loop.run_until_complete(self._gen(text, str(p)))
        finally: loop.close()
        return p

    def cleanup(self, keep: int = 10):
        files = sorted(self._temp_dir.glob("athena_edge_*.mp3"), key=lambda f: f.stat().st_mtime)
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
# ⚡ 唤醒词匹配器 (从 v3.1 继承)
# ============================================================
class WakeWordMatcher:
    def __init__(self, patterns, min_match=2):
        self.patterns = patterns; self.min_match = min_match

    @staticmethod
    def _similar(a, b):
        pairs = [("si","shi"),("shi","si"),("xi","si"),("si","xi"),
                 ("xi","shi"),("shi","xi"),("na","la"),("la","na"),
                 ("na","nuo"),("nuo","na"),("sen","shen"),("shen","sen"),
                 ("se","she"),("she","se"),("di","ti"),("ti","di"),
                 ("juan","jue"),("hei","hai"),("su","si"),("si","su")]
        return (a,b) in pairs or (b,a) in pairs

    def _to_pinyin(self, text):
        from pypinyin import lazy_pinyin, Style
        return lazy_pinyin(text, style=Style.NORMAL, errors="ignore")

    def match(self, text: str) -> tuple[bool, str]:
        if not text: return False, ""
        text_lower = text.lower().strip()

        # 英文精确
        for prefix in ["athena", "Athena", "ATHENA", "hey athena", "hey Athena"]:
            if text_lower.startswith(prefix):
                rest = text_lower[len(prefix):].strip()
                for sep in [",", " ", "，", "。"]:
                    if rest.startswith(sep): rest = rest[1:].strip()
                return True, rest

        # 拼音
        text_pinyin = self._to_pinyin(text_lower)
        for pattern in self.patterns:
            score = 0.0
            for i, (tp, pp) in enumerate(zip(text_pinyin, pattern)):
                if i >= len(pattern): break
                if tp == pp: score += 1
                elif self._similar(tp, pp): score += 0.5
            if score >= self.min_match:
                # 提取后续文本
                wake_end = len(pattern)
                if len(text_lower) > wake_end:
                    rest = text_lower[wake_end:]
                    for sep in ["，", ",", " ", "。", "、"]:
                        if rest.startswith(sep): rest = rest[1:].strip()
                    if len(rest) > 1: return True, rest
                return True, ""

        # 中文精确兜底
        for ww in ["雅典娜", "阿斯娜", "阿西娜", "阿森纳", "嘿雅典娜", "嘿阿斯娜"]:
            if text_lower.startswith(ww):
                rest = text_lower[len(ww):].strip()
                for sep in ["，", ",", " ", "、", "。"]:
                    if rest.startswith(sep): rest = rest[1:].strip()
                return True, rest

        return False, ""


# ============================================================
# 🤖 Athena 主控制器 v4
# ============================================================
class AthenaVoiceAssistant:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sample_rate = 16000

        # ASR
        self.asr = QwenASREngine(
            api_key=cfg["dashscope_api_key"],
            model=cfg["asr_model"],
        )
        self.asr.on_transcript = self._on_asr_transcript
        self.asr.on_error = self._on_asr_error

        # TTS
        self.tts_provider = cfg["tts_provider"]
        if cfg["dashscope_api_key"]:
            self.tts_cosy = CosyVoiceTTSEngine(cfg["dashscope_api_key"], cfg["cosyvoice_voice_id"])
        else:
            self.tts_cosy = None
        self.tts_edge = EdgeTTSEngine(cfg["tts_voice_edge"])
        self.player = AudioPlayer()

        # 唤醒词
        self.wake_matcher = WakeWordMatcher(
            patterns=cfg["wake_pinyin_patterns"],
            min_match=cfg["wake_min_pinyin_match"],
        )

        # 状态
        self.state = "IDLE"  # IDLE | LISTENING | PROCESSING | SPEAKING
        self.conversation_history = [{"role": "system", "content": _build_system_prompt()}]
        self._active_mode = False
        self._last_response_time = 0.0
        self._waiting_for_command = False
        self._wake_time = 0.0
        self._pending_command = None
        self._wake_cooldown_until = 0.0

        # TTS 播放队列（逐句）
        self._tts_queue = []
        self._tts_lock = threading.Lock()
        self._speaking_event = threading.Event()

        # 中断
        self._interrupt = threading.Event()
        self._pipeline_abort = threading.Event()

        # 音频流
        self._stop_flag = threading.Event()
        self._audio_stream = None

        log("system", f"🔧 ASR: Qwen(DashScope) | LLM: DeepSeek | TTS: {self.tts_provider}")

    # ===== ASR 回调 =====

    def _on_asr_transcript(self, text: str, is_final: bool):
        """收到 Qwen ASR 转录文本"""
        if not is_final:
            # 实时显示中间结果
            log("debug", f"   ~ {text}")
            return

        if not text or len(text.strip()) < 1:
            return

        log("debug", f"   ✓ 最终: '{text}'")

        # 过滤纯噪音
        if len(set(text)) <= 2 and len(text) > 3:
            return

        # 检查打断
        if self.state == "SPEAKING":
            log("wake", "⚡ 语音打断！")
            self.player.stop()
            self._interrupt.set()
            self._pipeline_abort.set()

        now = time.time()

        # 根据状态处理
        if self.state in ("IDLE", "LISTENING"):
            is_wake, rest = self.wake_matcher.match(text)
            log("debug", f"   [{'唤醒' if is_wake else '未匹配'}] '{text}'")

            if is_wake and now >= self._wake_cooldown_until:
                log("wake", f"⚡ 唤醒！")
                self._wake_cooldown_until = now + 1.5
                self._active_mode = True
                self._last_response_time = now
                self._pipeline_abort.clear()

                if rest and len(rest) > 1:
                    log("you", f"你: {rest}")
                    self._pending_command = rest
                    self.state = "PROCESSING"
                    threading.Thread(target=self._process_command, daemon=True).start()
                else:
                    self._play_ding()
                    self._waiting_for_command = True
                    self._wake_time = now
                    self.state = "LISTENING"
                    log("system", "🔊 叮~ 请说话...")
                return

            # 如果在 LISTENING 状态（等命令），任何内容都当作命令
            if self.state == "LISTENING" and self._waiting_for_command:
                log("you", f"你: {text}")
                self._waiting_for_command = False
                self._pipeline_abort.clear()

                if any(w in text for w in ["退下", "不用了", "没你的事了", "睡觉吧", "拜拜"]):
                    log("athena", "好的，需要时叫我。")
                    self.conversation_history = [self.conversation_history[0]]
                    self._active_mode = False
                    self.state = "IDLE"
                    return

                self._pending_command = text
                self.state = "PROCESSING"
                threading.Thread(target=self._process_command, daemon=True).start()
                return

        elif self.state == "SPEAKING":
            # 打断已处理
            pass

    def _on_asr_error(self, error):
        log("error", f"ASR 错误: {error}")

    # ===== 提示音 =====

    def _play_ding(self):
        sr = 24000
        t = np.linspace(0, 0.12, int(sr * 0.12), endpoint=False)
        audio = (np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 10) * 0.4).astype(np.float32)
        p = Path(tempfile.gettempdir()) / "athena_tts" / f"ding_{threading.get_ident()}.wav"
        p.parent.mkdir(exist_ok=True)
        try:
            with wave.open(str(p), 'w') as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
                wf.writeframes((audio * 32767).astype(np.int16).tobytes())
            self.player.play(str(p))
        except PermissionError:
            pass

    # ===== LLM =====

    def _get_weather(self, city: str = "") -> str:
        """通过 wttr.in 获取天气（免费，无需 API Key）"""
        try:
            import urllib.request, urllib.parse
            c = urllib.parse.quote(city or "auto")
            url = f"https://wttr.in/{c}?format=%C+%t+%h+%w&lang=zh"
            req = urllib.request.Request(url, headers={"User-Agent": "curl"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                return resp.read().decode("utf-8").strip()
        except Exception as e:
            return f"天气查询失败: {e}"

    def _web_search(self, query: str) -> str:
        """联网搜索"""
        try:
            import urllib.request, urllib.parse
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
            req = urllib.request.Request(url, headers={"User-Agent": "Athena/4.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
                abstract = data.get("AbstractText", "")
                if abstract:
                    return abstract
                topics = data.get("RelatedTopics", [])
                if topics and "Text" in topics[0]:
                    return topics[0]["Text"]
                return ""
        except Exception:
            return ""

    def _call_llm_stream(self, user_text: str):
        """流式调用 LLM，yield 每个文本 chunk"""
        from openai import OpenAI
        # 每次调用前刷新系统提示词里的时间
        self.conversation_history[0] = {"role": "system", "content": _build_system_prompt()}
        client = OpenAI(api_key=self.cfg["deepseek_api_key"], base_url="https://api.deepseek.com")
        self.conversation_history.append({"role": "user", "content": user_text})
        if len(self.conversation_history) > 41:
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-40:]

        full = ""
        try:
            stream = client.chat.completions.create(
                model="deepseek-chat", messages=self.conversation_history,
                stream=True, temperature=self.cfg["temperature"], max_tokens=150,
            )
            for chunk in stream:
                if self._pipeline_abort.is_set():
                    break
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full += text
                    yield text
        except Exception as e:
            log("error", f"LLM 错误: {e}")
            if self.conversation_history[-1]["role"] == "user":
                self.conversation_history.pop()

        if full:
            self.conversation_history.append({"role": "assistant", "content": full})

    # ===== TTS（逐句） =====

    def _speak_sentence(self, text: str):
        """合成并播放一句话"""
        if not text.strip():
            return
        if self._pipeline_abort.is_set():
            return

        log("debug", f"   🔊 TTS: '{text[:50]}...'")

        # 尝试 CosyVoice
        mp3 = None
        if self.tts_provider == "cosyvoice" and self.tts_cosy:
            try:
                mp3 = self.tts_cosy.synthesize(text, speech_rate=1.05)
            except Exception as e:
                log("debug", f"   CosyVoice 失败，用 Edge 兜底: {e}")

        # Edge 兜底
        if mp3 is None:
            try:
                mp3 = self.tts_edge.generate(text)
            except Exception as e:
                log("error", f"TTS 失败: {e}")
                return

        if self._pipeline_abort.is_set():
            return

        # 播放
        self.player.play(str(mp3))
        while self.player.is_busy():
            if self._pipeline_abort.is_set():
                self.player.stop()
                return
            time.sleep(0.05)

    def _speak_sentences_streaming(self, text_generator):
        """
        流式 LLM → 逐句 TTS
        LLM 每输出一个句子（碰到。！？）就立即合成播放
        """
        sentence = ""
        for chunk in text_generator:
            if self._pipeline_abort.is_set():
                break
            print(chunk, end="", flush=True)
            sentence += chunk
            # 句子边界
            if any(p in chunk for p in ['。', '！', '？', '！', '\n', '…']):
                s = sentence.strip()
                if s and len(s) > 1:
                    self._speak_sentence(s)
                sentence = ""

        # 剩余文本
        if sentence.strip() and not self._pipeline_abort.is_set():
            self._speak_sentence(sentence.strip())

        print()

    # ===== 快捷指令 =====

    def _try_quick_command(self, text: str) -> str | None:
        """尝试匹配快捷指令，返回响应文本或 None"""
        from datetime import datetime
        now = datetime.now()
        weekdays = ["一", "二", "三", "四", "五", "六", "日"]

        quick = {
            "现在几点": lambda: now.strftime("现在是%H点%M分。"),
            "几点了": lambda: now.strftime("%H点%M分了。"),
            "今天星期几": lambda: f"今天是星期{weekdays[now.weekday()]}。",
            "今天几号": lambda: f"今天是{now.year}年{now.month}月{now.day}号。",
            "今天日期": lambda: f"今天是{now.year}年{now.month}月{now.day}号，星期{weekdays[now.weekday()]}。",
            "日期": lambda: f"今天是{now.year}年{now.month}月{now.day}号。",
            "现在时间": lambda: now.strftime("现在是%H点%M分。"),
        }

        text_clean = text.strip().replace(" ", "")
        for key, fn in quick.items():
            if key in text_clean:
                return fn()
        return None

    # ===== 命令处理 =====

    def _process_command(self):
        """处理用户命令（在独立线程中运行）"""
        text = self._pending_command
        self._pending_command = None

        # 快捷指令
        quick = self._try_quick_command(text)
        if quick:
            log("athena", f"Athena: {quick}")
            log("system", "🔊 正在说话...")
            self._pipeline_abort.clear()
            self._speak_sentence(quick)
            self._finish_turn()
            return

        # 天气：直接用 wttr.in API（精确数据）
        if "天气" in text:
            log("system", "🌤 查天气...")
            # 提取城市名
            city = "auto"
            for c in ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京", "重庆", "西安"]:
                if c in text:
                    city = c; break
            wx = self._get_weather(city)
            if wx and "失败" not in wx:
                text = f"{text}\n[当前天气: {wx}。请用中文口语回答。]"

        # 其他需要搜索的问题
        elif any(kw in text for kw in ["新闻", "最新", "今天发生", "热搜"]):
            log("system", "🔍 搜索中...")
            sr = self._web_search(text)
            if sr:
                text = f"{text}\n[搜索结果: {sr[:300]}]"

        # LLM 流式 + 逐句 TTS
        log("athena", "Athena: ")
        print("  ", end="", flush=True)
        log("system", "🔊 正在说话...")
        self._pipeline_abort.clear()
        self._speak_sentences_streaming(self._call_llm_stream(text))
        self._finish_turn()

    def _finish_turn(self):
        """完成一轮对话，保持对话模式（无需唤醒词继续聊）"""
        self._last_response_time = time.time()
        self._active_mode = True
        self._waiting_for_command = True  # 保持对话！下次说话自动当命令
        self._wake_time = time.time()     # 重置超时计时

        # 清理
        try:
            if self.tts_cosy: self.tts_cosy.cleanup()
            self.tts_edge.cleanup()
        except: pass

        self.state = "LISTENING"
        log("debug", "👂 继续听（无需唤醒词）...")

    # ===== 音频流 =====

    def _audio_callback(self, indata, frames, time_info, status):
        """音频输入回调 — 直接把 PCM 数据发给 Qwen ASR"""
        if self._stop_flag.is_set():
            return

        audio = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()

        # 转为 16-bit PCM bytes
        pcm = (audio * 32767).astype(np.int16).tobytes()
        self.asr.send_audio(pcm)

    def run(self):
        """启动"""
        # 检查 API Key
        if not self.cfg["dashscope_api_key"]:
            log("error", "=" * 55)
            log("error", "  需要 DashScope API Key！")
            log("error", "  1. 打开 https://dashscope.aliyun.com/")
            log("error", "  2. API-KEY 管理 → 创建 Key")
            log("error", "  3. 在 .env 文件里加一行：")
            log("error", "     DASHSCOPE_API_KEY=sk-xxx")
            log("error", "")
            log("error", "  （免费额度：ASR 每月 100 小时）")
            log("error", "=" * 55)
            return

        if self.cfg["deepseek_api_key"] in ("sk-你的密钥", "", None):
            log("error", "=" * 50)
            log("error", "  需要 DeepSeek API Key！在 .env 设置 DEEPSEEK_KEY")
            log("error", "=" * 50)
            return

        log("system", "")
        log("system", "=" * 55)
        log("system", "  🦉 Athena AI 语音助手 v4.0 — 云端版")
        log("system", "  喊「Athena / 雅典娜」唤醒我")
        log("system", "  ASR: Qwen (DashScope) | LLM: DeepSeek")
        log("system", f"  TTS: {'CosyVoice' if self.tts_cosy else 'Edge'} + 逐句流式")
        log("system", "  Ctrl+C 退出")
        log("system", "=" * 55)

        # 连接 ASR
        log("system", "🔗 连接 Qwen ASR...")
        self.asr.connect()

        if not self.asr._ready:
            log("error", "Qwen ASR 连接失败，请检查 API Key 和网络")
            log("error", "提示：如果你有 VPN，确保已连接")
            return

        # 启动音频
        try:
            self._start_audio()
        except Exception as e:
            log("error", f"麦克风错误: {e}")
            return

        # 测试扬声器
        try:
            self._play_ding()
            log("system", "✅ 就绪")
        except Exception as e:
            log("error", f"扬声器错误: {e}")

        # 主循环
        try:
            while not self._stop_flag.is_set():
                # 检查超时
                if self.state == "LISTENING" and self._waiting_for_command:
                    if time.time() - self._wake_time > self.cfg["active_timeout"]:
                        self._waiting_for_command = False
                        log("system", "💤 超时回休眠")
                        self.state = "IDLE"

                # 检查超时（活跃模式）
                if self._active_mode and self.state == "LISTENING":
                    if time.time() - self._last_response_time > self.cfg["active_timeout"] * 2:
                        self._active_mode = False
                        self.state = "IDLE"
                        log("system", "💤 已退出活跃模式")

                time.sleep(0.1)
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

    def _stop(self):
        self._stop_flag.set()
        self._pipeline_abort.set()
        if self._audio_stream:
            try: self._audio_stream.stop(); self._audio_stream.close()
            except: pass
        self.asr.stop()
        self.player.cleanup()


# ============================================================
# 🚀 入口
# ============================================================
def main():
    # 重新从环境读取（用户可能在 .env 里设置了）
    if os.environ.get("DASHSCOPE_API_KEY"):
        CONFIG["dashscope_api_key"] = os.environ["DASHSCOPE_API_KEY"]
    if os.environ.get("DEEPSEEK_KEY"):
        CONFIG["deepseek_api_key"] = os.environ["DEEPSEEK_KEY"]
    if os.environ.get("COSYVOICE_VOICE_ID"):
        CONFIG["cosyvoice_voice_id"] = os.environ["COSYVOICE_VOICE_ID"]
    if os.environ.get("TTS_PROVIDER"):
        CONFIG["tts_provider"] = os.environ["TTS_PROVIDER"]

    AthenaVoiceAssistant(CONFIG).run()


if __name__ == "__main__":
    main()
