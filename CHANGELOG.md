# 更新日志

## v4.0 (2026-06-18) — 云端重构版

### 🚀 重大升级
- **ASR：faster-whisper → 阿里云 Qwen ASR**  
  从本地 239MB tiny 模型 CPU 推理 → 云端 WebSocket 流式识别，服务端 VAD，中文精度 ~95%
- **TTS：edge-tts → 百炼声音克隆**  
  从微软通用朗读 → 用用户音频素材克隆专属 Athena 音色（Qwen-TTS），支持预设音色兜底
- **流式架构：等全文 → 逐句播放**  
  LLM 每生成一句（遇。！？）立即送 TTS 合成播放，首音延迟从 3~8s → < 1s

### 🎭 人设重写
- 基于手写 25 条种子训练数据提炼 DNA  
- Jarvis 风格：精确、高效、先结论后原因  
- 三层性格：理性干练（默认）/ 温暖共情（情感触发）/ 冷幽默（偶尔闪现）  
- 强化禁语列表：不只 "作为AI助手"，还禁 "我理解你的感受" 等空洞表达

### 🌤 实时信息
- **天气**：接入 wttr.in 免费 API，自动提取城市名，实时温度/湿度/风力
- **联网搜索**：DuckDuckGo + wttr.in，天气/新闻关键词自动触发
- **实时时间**：系统提示词注入当前日期和时间（修复 LLM 训练截止日期问题）

### 🔧 工程优化
- **对话模式修复**：回复后自动持续聆听，无需重复唤醒词
- **快捷指令**：日期/时间秒回，不走 LLM
- **多线程架构重写**：ASR WebSocket 回调 + LLM 流式 + TTS 逐句，AbortController 中断机制
- **.env 密钥管理**：`.env` 文件强制覆盖系统环境变量，本地密钥永不上传

---

## v3.1 (2026-06-18) — 稳定发布版

### 首个可用版本
- **唤醒词**：pypinyin 拼音模糊匹配，解决 Whisper 中文音译不准问题（"阿蘇娜" → ["a","su","na"] = ["a","si","na"]）
- **ASR**：faster-whisper tiny 本地 CPU 推理
- **LLM**：DeepSeek API 流式对话
- **TTS**：edge-tts 微软免费语音
- **打断**：唤醒词打断 TTS 播放
- **发布到 GitHub**：CC BY-NC-ND 4.0 许可证

### Bug 修复
- 修复 API Key 被环境变量覆盖
- 修复 ding.wav 多线程 PermissionError
- 修复 VAD 阈值过低导致噪音误触发
- 移除 Git 历史中的硬编码 API Key

---

## v3.0 (2026-06-17) — 拼音匹配突破

- 首次成功实现唤醒词检测：pypinyin 拼音模糊匹配
- 噪声音频过滤（去除纯重复字符模式）

---

## v2.0 (2026-06-17) — 尝试 openWakeWord

- 尝试使用 openWakeWord 专门唤醒词引擎
- 失败：无 "Athena" 预训练模型，"alexa" 模型无法触发
- 尝试手动下载模型文件（GitHub 被墙，通过 ghproxy 镜像）

---

## v1.0 (2026-06-17) — 初版

- 搭建完整语音管线：sounddevice 录音 → faster-whisper ASR → DeepSeek API → edge-tts TTS
- 能量 VAD 语音检测
- 文字精确匹配唤醒词 → 失败（Whisper tiny 把 "Athena" 转录为 "阿斯娟娟"）
- HuggingFace 下载超时 → 配置 hf-mirror.com 镜像
- PyTorch CUDA DLL 不兼容（GTX 1080 Ti Pascal + CUDA 13.0）→ 全部改用 CPU
