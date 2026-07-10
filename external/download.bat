@echo off
setlocal

if not defined WGPU_VER set "WGPU_VER=v29.0.1.1"

echo WebGPU-Native Download Script (Windows)  v=%WGPU_VER%
echo ================================

where powershell >nul 2>&1
if errorlevel 1 (
    echo Error: PowerShell is required to resolve version-matched WebGPU headers.
    echo Check the PowerShell installation and try again.
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0download.ps1"
set "DOWNLOAD_RESULT=%ERRORLEVEL%"

if not "%DOWNLOAD_RESULT%"=="0" (
    echo WebGPU-Native installation failed with exit code %DOWNLOAD_RESULT%.
)

exit /b %DOWNLOAD_RESULT%
