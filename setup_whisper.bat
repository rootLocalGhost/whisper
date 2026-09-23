@echo off
setlocal enabledelayedexpansion

echo ======================================================
echo   Vivestream Revived - Whisper AI Engine Setup
echo ======================================================
echo.

:: 1. Check for conda
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Conda not found in PATH!
    echo Please install Miniconda or Anaconda, or install Python 3.10-3.12.
    pause
    exit /b 1
)

echo [1/4] Checking conda environment 'whisper'...
conda env list | findstr /i "whisper" >nul
if %errorlevel% neq 0 (
    echo [*] Creating 'whisper' conda environment with Python 3.12...
    call conda create -y -n whisper python=3.12
) else (
    echo [*] 'whisper' conda environment already exists.
)

echo.
echo [2/4] Installing/Verifying dependencies and FFmpeg...
call conda install -y -n whisper ffmpeg -c defaults
call conda run -n whisper pip install -r requirements.txt gradio

echo.
echo [3/4] Testing Hardware Acceleration (Intel Arc XPU / CPU)...
call conda run -n whisper python service.py --check

echo.
echo ======================================================
echo   Setup Complete! Whisper is ready for Vivestream!
echo ======================================================
echo.
pause
