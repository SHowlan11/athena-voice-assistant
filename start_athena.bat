@echo off
chcp 65001 >nul
title Athena AI 语音助手 v2

echo.
echo   🦉 Athena AI 语音助手 v2.0
echo   ============================
echo.

:: 设置 API Key（如果还没设）
if defined DEEPSEEK_KEY (
    echo   ✅ DEEPSEEK_KEY 已设置
) else (
    echo   💡 提示：可通过环境变量设置 Key
    echo      set DEEPSEEK_KEY=sk-xxxxxxxxxxxx
    echo.
)

:: 激活虚拟环境并启动
call D:\athena-project\venv\Scripts\activate.bat
python D:\athena-project\athena_voice_v3.py

pause
