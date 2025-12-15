@echo off
REM ============================================================================
REM WCN WASM Test Server
REM ============================================================================
REM This script starts a local HTTP server and opens the WASM test page.
REM ============================================================================

echo ========================================
echo WCN WASM Test Server
echo ========================================
echo.

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python not found!
    echo Please install Python or use another HTTP server.
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

echo Starting HTTP server on port 9000...
echo.
echo Test page will be available at:
echo   http://localhost:9000/examples/wasm/test.html
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start Python HTTP server
python -m http.server 9000
