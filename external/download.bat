@echo off
REM WebGPU-Native Res: https://github.com/gfx-rs/wgpu-native/releases/tag/v25.0.2.2
REM Let users choose system, architecture, compiler, and mode to download specified packages
REM Extract to external\wgpu directory

echo WebGPU-Native Download Script (Windows)
echo ================================

REM Check if PowerShell is installed
where powershell >nul 2>&1
if %errorlevel% equ 0 (
    echo PowerShell detected, using PowerShell script to download...
    powershell -ExecutionPolicy Bypass -File "%~dp0download.ps1"
) else (
    echo PowerShell not found, using batch script to download...
    
    REM Set download URL and filename
    set "DOWNLOAD_URL="
    set "FILENAME="
    
    REM Let user choose platform
    echo Please select platform:
    echo 1. Windows x86_64 (MSVC)
    echo 2. Windows x86_64 (GNU)
    echo 3. Windows i686 (MSVC)
    echo 4. Windows aarch64 (MSVC)
    echo 5. Linux x86_64
    echo 6. MacOS x86_64
    echo 7. Android aarch64
    
    choice /c 1234567 /m "Select platform"
    if %errorlevel% equ 1 (
        set "PLATFORM=windows-x86_64-msvc"
        set "FILENAME=wgpu-windows-x86_64-msvc-release.zip"
        set "DOWNLOAD_URL=https://github.com/gfx-rs/wgpu-native/releases/download/v25.0.2.2/wgpu-windows-x86_64-msvc-release.zip"
    ) else if %errorlevel% equ 2 (
        set "PLATFORM=windows-x86_64-gnu"
        set "FILENAME=wgpu-windows-x86_64-gnu-release.zip"
        set "DOWNLOAD_URL=https://github.com/gfx-rs/wgpu-native/releases/download/v25.0.2.2/wgpu-windows-x86_64-gnu-release.zip"
    ) else if %errorlevel% equ 3 (
        set "PLATFORM=windows-i686-msvc"
        set "FILENAME=wgpu-windows-i686-msvc-release.zip"
        set "DOWNLOAD_URL=https://github.com/gfx-rs/wgpu-native/releases/download/v25.0.2.2/wgpu-windows-i686-msvc-release.zip"
    ) else if %errorlevel% equ 4 (
        set "PLATFORM=windows-aarch64-msvc"
        set "FILENAME=wgpu-windows-aarch64-msvc-release.zip"
        set "DOWNLOAD_URL=https://github.com/gfx-rs/wgpu-native/releases/download/v25.0.2.2/wgpu-windows-aarch64-msvc-release.zip"
    ) else if %errorlevel% equ 5 (
        set "PLATFORM=linux-x86_64"
        set "FILENAME=wgpu-linux-x86_64-release.zip"
        set "DOWNLOAD_URL=https://github.com/gfx-rs/wgpu-native/releases/download/v25.0.2.2/wgpu-linux-x86_64-release.zip"
    ) else if %errorlevel% equ 6 (
        set "PLATFORM=macos-x86_64"
        set "FILENAME=wgpu-macos-x86_64-release.zip"
        set "DOWNLOAD_URL=https://github.com/gfx-rs/wgpu-native/releases/download/v25.0.2.2/wgpu-macos-x86_64-release.zip"
    ) else if %errorlevel% equ 7 (
        set "PLATFORM=android-aarch64"
        set "FILENAME=wgpu-android-aarch64-release.zip"
        set "DOWNLOAD_URL=https://github.com/gfx-rs/wgpu-native/releases/download/v25.0.2.2/wgpu-android-aarch64-release.zip"
    )
    
    REM Create download directory
    if not exist "%~dp0wgpu" mkdir "%~dp0wgpu"
    
    REM Download file
    echo Downloading %FILENAME%...
    powershell -Command "Invoke-WebRequest -Uri '%DOWNLOAD_URL%' -OutFile '%~dp0wgpu\%FILENAME%'"
    
    REM Extract file
    echo Extracting %FILENAME%...
    powershell -Command "Expand-Archive -Path '%~dp0wgpu\%FILENAME%' -DestinationPath '%~dp0wgpu' -Force"
    
    REM Delete archive
    del "%~dp0wgpu\%FILENAME%"
    
    echo Download and extraction complete!
)

pause