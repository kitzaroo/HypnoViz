@echo off
rem Creates the project's virtual environment (.venv, in this folder) if it is missing and installs requirements.txt into it.
rem Called by run.bat and build_exe.bat - you can also double-click it to (re)install.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment in .venv ...
  python -m venv .venv 2>nul || py -3 -m venv .venv
  if not exist ".venv\Scripts\python.exe" (
    echo Could not create the virtual environment. Install Python 3.10+ and make sure "python" or "py" works.
    pause
    exit /b 1
  )
)
fc /b requirements.txt ".venv\requirements.installed" >nul 2>&1
if errorlevel 1 (
  echo Installing packages from requirements.txt ...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Package install failed.
    pause
    exit /b 1
  )
  copy /y requirements.txt ".venv\requirements.installed" >nul
)
exit /b 0
