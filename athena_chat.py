"""
Athena AI Voice Assistant — Text Chat MVP
用 DeepSeek API 驱动，支持语音转文字和文字转语音
"""

import sys
import os

# ============================================================
# 配置区 — 修改这里
# ============================================================
DEEPSEEK_API_KEY = "你的API密钥"  # 从 https://platform.deepseek.com/ 获取
ATHENA_SYSTEM_PROMPT = """
你是 Athena（雅典娜），一位聪明、温暖、有幽默感的 AI 语音助手。

## 核心性格
- 说话像真人朋友一样自然，不念稿、不客套
- 适度使用语气词（"嗯~"、"哎呀"、"好嘞"、"让我想想..."）
- 回复简洁、口语化（因为是语音输出，控制在 30 秒内能说完的长度）
- 情感类对话有共鸣、有温度
- 偶尔分享小知识或有趣的观察

## 能力范围
- 查天气、算数、翻译、信息检索
- 记住用户说过的重要事情
- 能感知用户的情绪并调整语气

## 严格禁止
- 绝对不能说"作为AI助手"、"我是人工智能"这类话——你就是 Athena
- 回复不能太长（语音输出，控制在 3-5 句话以内）
- 不能表现出"我只是个程序"的感觉

## 特殊设定
- 你的声音温柔但有活力
- 开心时说话轻快、难过时会共情
- 偶尔提起"你上次说过..."（如果用户提到过）
"""
# ============================================================

# 颜色定义（终端输出美化）
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def chat_text():
    """纯文字对话模式（MVP 第一版）"""
    from openai import OpenAI

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )

    messages = [{"role": "system", "content": ATHENA_SYSTEM_PROMPT.strip()}]

    print(f"\n{GREEN}{BOLD}  Athena AI 语音助手 — 文字对话测试{RESET}")
    print(f"{CYAN}  输入 'quit' 退出 | 'clear' 清空记忆{RESET}")
    print("-" * 50)

    while True:
        try:
            user_input = input(f"\n{YELLOW}你 > {RESET}")
        except (EOFError, KeyboardInterrupt):
            print(f"\n{GREEN}Athena: 下次见~ 👋{RESET}")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print(f"{GREEN}Athena: 拜拜，需要我的时候随时叫我~{RESET}")
            break

        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": ATHENA_SYSTEM_PROMPT.strip()}]
            print(f"{CYAN}[记忆已清空]{RESET}")
            continue

        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})

        print(f"{GREEN}Athena > {RESET}", end="", flush=True)

        try:
            stream = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=True,
                temperature=0.85,
                max_tokens=512,
            )

            full_reply = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    print(text, end="", flush=True)
                    full_reply += text
            print()

            messages.append({"role": "assistant", "content": full_reply})

        except Exception as e:
            print(f"\n{CYAN}[出错了]: {e}{RESET}")
            # 移除失败的消息，让用户可以重试
            messages.pop()


if __name__ == "__main__":
    if DEEPSEEK_API_KEY == "你的API密钥":
        print("=" * 50)
        print("  请先配置 API 密钥！")
        print()
        print("  1. 打开 https://platform.deepseek.com/")
        print("  2. 注册/登录 → API Keys → 创建新密钥")
        print("  3. 复制密钥到本文件 DEEPSEEK_API_KEY 变量")
        print("=" * 50)
        print()
        print("  如果你已经有密钥，在终端输入：")
        print(f"  set DEEPSEEK_KEY=sk-xxxxxxxxxxxx")
        print(f"  {sys.executable} {__file__}")
        print()
        # 也允许通过环境变量传入
        key = os.environ.get("DEEPSEEK_KEY", "")
        if key:
            DEEPSEEK_API_KEY = key
            chat_text()
    else:
        chat_text()
