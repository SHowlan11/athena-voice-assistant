# 🦉 Athena AI 语音助手

一个基于 Python 的个性化 AI 语音助手，支持唤醒词免提对话、随时打断、上下文记忆。

> **当前版本：v3.1** | 全管线已打通：唤醒 → 语音识别 → LLM → 语音合成

## 功能

- ⚡ **唤醒词免提** — 喊 "Athena / 雅典娜" 唤醒，拼音模糊匹配，对 Whisper 转录容错
- 🎤 **自动语音检测** — VAD 自动判断你说完没有，无需按键
- 🧠 **LLM 对话** — DeepSeek API 驱动，带 Athena 专属人设
- 🔊 **语音合成** — 微软 edge-tts 免费语音
- ✋ **随时打断** — Athena 说话时喊名字即可打断
- 💬 **上下文记忆** — 多轮对话自动关联

## 快速开始

### 1. 环境要求

- Python 3.10+
- Windows / macOS / Linux
- 麦克风 + 扬声器

### 2. 安装依赖

```bash
pip install sounddevice numpy faster-whisper edge-tts openai pygame pypinyin
```

### 3. 设置 API Key

复制 `.env.example` 为 `.env`，填入你的 [DeepSeek API Key](https://platform.deepseek.com/)：

```bash
cp .env.example .env
# 编辑 .env，把 sk-你的密钥 替换为真实密钥
```

### 4. 启动

```bash
python athena_voice_v3.py
```

首次运行会自动下载 faster-whisper 模型（约 240MB），国内用户已配置 HuggingFace 镜像加速。

## 使用方法

- 喊 **"Athena"**（或 "雅典娜"、"阿斯娜"）→ 听到"叮" → 说出你的问题
- 回应后 8 秒内可以继续对话，无需重复喊唤醒词
- 任何时候喊 **"Athena"** 可以打断她正在说的话
- 说 "退下" / "睡觉吧" 让 Athena 回到休眠状态

## 技术架构

```
麦克风 → VAD(能量检测) → faster-whisper tiny(ASR) → pypinyin(拼音匹配)
    → DeepSeek API(LLM) → edge-tts(TTS) → pygame 播放
```

## 已知局限

- **ASR 精度**：faster-whisper tiny 在 CPU 上运行，中文准确率有提升空间
- **TTS 情感**：edge-tts 不支持情感控制和音色定制
- **对话深度**：依赖 System Prompt 调优

## 许可证

[CC BY-NC-ND 4.0](LICENSE) — 可分享，禁止商用，禁止二次修改分发。
