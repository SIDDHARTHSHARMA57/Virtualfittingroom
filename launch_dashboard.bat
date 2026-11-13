@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0
".venv\Scripts\python.exe" src\camera\dashboard.py %*
