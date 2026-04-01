@echo off
echo ================================================
echo    Hybrid-IDS Framework - Stopping Services
echo ================================================
echo.

set LOGFILE=logs\production.log
echo [%DATE% %TIME%] Stopping all services... >> %LOGFILE%

docker compose down

echo.
echo ✅ All services stopped successfully.
echo Log saved to logs\production.log
echo.
pause