@echo off
cd /d "%~dp0"
call setup_env.bat || exit /b 1
".venv\Scripts\python.exe" -m pip install pyinstaller
rmdir /s /q build dist 2>nul
del Hypnosis.spec 2>nul
set ICON=
if exist jkmrf-t2seg-001.ico set ICON=--icon jkmrf-t2seg-001.ico
if exist hypnosis_icon.ico set ICON=--icon hypnosis_icon.ico
".venv\Scripts\python.exe" -m PyInstaller --clean --onedir --noconsole --name Hypnosis %ICON% --hidden-import tkinter --hidden-import proctap._native --collect-all proctap --exclude-module scipy --collect-all miniaudio --collect-all moderngl --hidden-import glcontext --hidden-import cv2 --collect-all cv2 --hidden-import av --collect-all av --hidden-import PIL --hidden-import comtypes --collect-all pycaw --collect-all comtypes --collect-submodules winrt hypnosis.py
copy /y spotify_icon.png dist\Hypnosis\ >nul 2>&1
echo.
echo Done: dist\Hypnosis\Hypnosis.exe   (keep the whole dist\Hypnosis folder together)
pause
