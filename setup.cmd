@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m pip install --no-cache-dir -r requirements-service.txt
if errorlevel 1 exit /b 1
if not exist ".env" copy /y ".env.example" ".env" >nul
echo Ready. Run the start command, then open http://127.0.0.1:8765/docs
pause
