@echo off
echo ================================================
echo    Hybrid-IDS Framework - Production Deployment
echo ================================================
echo.

:: Create logs directory if not exists
if not exist logs mkdir logs

set LOGFILE=logs\production.log
echo [%DATE% %TIME%] Starting production deployment... >> %LOGFILE%

echo [1/4] Checking Docker status...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker Desktop.
    echo [%DATE% %TIME%] ERROR: Docker not running >> %LOGFILE%
    pause
    exit /b
)

echo [2/4] Stopping any old containers...
docker compose down >> %LOGFILE% 2>&1

echo [3/4] Building and starting services (Streamlit + Flask)...
echo [%DATE% %TIME%] Running: docker compose up --build -d >> %LOGFILE%
docker compose up --build -d

echo.
echo ================================================
echo ✅ Deployment Successful!
echo ================================================
echo.
echo 📊 Streamlit Dashboard → http://localhost:8501
echo 🔌 Flask API          → http://localhost:5000
echo 📝 Log file saved at: logs\production.log
echo.
echo To stop the services, double-click stop_production.bat
echo.
pause