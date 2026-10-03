@echo off
cd /d "%~dp0"
call setup_env.bat || exit /b 1
".venv\Scripts\python.exe" hypnosis.py %*
