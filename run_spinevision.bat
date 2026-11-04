@echo off
setlocal
set "PYTHON=D:\Application\CodeEditor\anaconda3\envs\pytorch\python.exe"
cd /d "%~dp0"
"%PYTHON%" SpineVision_AI_real_infer.py
endlocal
