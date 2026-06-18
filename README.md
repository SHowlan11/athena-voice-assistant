# 🦉 Athena AI 语音助手

个性化 AI 语音助手 — 唤醒词免提、云端流式识别、克隆音色、逐句对话。

> **当前版本：v4.0** | 阿里云 Qwen ASR + DeepSeek LLM + 百炼声音克隆 + 联网搜索

## 功能

- ⚡ **唤醒词免提** — 喊 "Athena / 雅典娜" 唤醒，拼音模糊匹配，ASR 转录容错
- 🎤 **云端流式 ASR** — 阿里云 Qwen 实时语音识别（WebSocket），自带 VAD，中文精度 ~95%
- 🗣️ **声音克隆** — 用你的音频素材克隆专属 Athena 声音（百炼 Qwen-TTS）
- 📻 **逐句流式 TTS** — LLM 每生成一句话立即合成播放，首音延迟 < 1s
- 🧠 **Jarvis 风格人设** — 精确、高效、有温度：先给结论再说原因，用事实和行动表达关心
- ✋ **随时打断** — 说话时任意语音即可打断
- 💬 **持续对话** — 回答后自动继续听，无需重复唤醒词
- 🔍 **联网搜索** — 天气（wttr.in 免费 API）、新闻自动搜索
- ⏰ **快捷指令** — 日期/时间秒回，不调 LLM
- 🔄 **持续对话模式** — 回答后自动聆听，8 秒无交互回休眠

## 快速开始

### 1. 环境要求

- Python 3.10+
- Windows / macOS / Linux
- 麦克风 + 扬声器

### 2. 安装依赖

```bash
pip install sounddevice numpy faster-whisper edge-tts openai pygame pypinyin websocket-client requests
```

### 3. 设置 API Key

复制 `.env.example` 为 `.env`，填入两个 API Key：

```bash
cp .env.example .env
```

```env
# DeepSeek API Key（LLM 对话）— platform.deepseek.com
DEEPSEEK_KEY=sk-你的密钥

# 阿里云百炼 API Key（ASR + TTS + 声音克隆）— bailian.console.aliyun.com
# ASR 每月免费 100 小时
DASHSCOPE_API_KEY=sk-你的密钥
```

### 4. 声音克隆（可选）

如需自定义 Athena 的声音：

1. 准备 10-20 秒的干净语音素材
2. 百炼控制台 → 声音复刻 → 上传 → 获取 `voice_id`
3. 在 `.env` 中添加 `ATHENA_VOICE_ID=你的voice_id`

不配置则默认使用 `longanyang` 温柔知性女声。

### 5. 启动

```bash
python athena_voice_v4.py
```

或双击 `start_athena.bat`

## 使用方法

- 喊 **"Athena"**（或 "雅典娜"、"阿斯娜"）唤醒，听到"叮"
- 直接说出问题，说完自动识别
- 回答后直接继续说，无需重复唤醒词
- 任何时候喊名字可打断她正在说的话
- 说 **"退下"** / **"睡觉吧"** 让 Athena 回到休眠
- 试试：**"今天几号"**（秒回）、**"北京天气"**（联网查）、**"我心情不好"**（共情模式）

## 技术架构

```
麦克风(16kHz PCM)
    ↓ WebSocket 流式发送
Qwen ASR（阿里云，服务端 VAD + 实时转录）
    ↓ 转录文本
pypinyin 拼音模糊匹配唤醒词
    ↓ 唤醒后
DeepSeek API 流式生成（Jarvis 风格 System Prompt）
    ↓ 每遇句号立即送 TTS
Qwen-TTS 克隆音色 / CosyVoice 预设音色
    ↓
pygame 播放 → 自动回到聆听模式
```

## 已知局限

- **股价/实时财经**：免费 API 在中国可用性差，待接入专用数据源
- **TTS 情感韵律**：Qwen-TTS 不支持运行时情感参数，待换 CosyVoice 3.0 本地版
- **长期记忆**：单次会话内记忆，跨会话记忆待加 ChromaDB
- **UI**：CMD 终端运行，无图形界面

## 许可证

[CC BY-NC-ND 4.0](LICENSE) — 可分享，禁止商用，禁止二次修改分发。
