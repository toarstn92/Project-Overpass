@echo off
rem Project Overpass launcher - double-click or run from the project folder.
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.10+ is required but was not found on PATH.
  pause
  exit /b 1
)
python init.py %*
if errorlevel 1 pause
