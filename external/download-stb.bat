@echo off
REM STB Libraries Download Script (Windows)
REM STB Repository: https://github.com/nothings/stb

echo STB Libraries Download Script (Windows)
echo ================================

REM Check if PowerShell is installed
where powershell >nul 2>&1
if %errorlevel% equ 0 (
    echo PowerShell detected, using PowerShell script to download...
    powershell -ExecutionPolicy Bypass -File "%~dp0download-stb.ps1"
) else (
    echo PowerShell not found, using batch script to download...
    
    REM Create download directory
    set "STB_DIR=%~dp0stb"
    set "INCLUDE_DIR=%STB_DIR%\include"
    
    if not exist "%INCLUDE_DIR%" mkdir "%INCLUDE_DIR%"
    
    REM Download headers
    echo Downloading STB header files...
    
    echo Downloading stb_truetype.h...
    powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/nothings/stb/master/stb_truetype.h' -OutFile '%INCLUDE_DIR%\stb_truetype.h'"
    if %errorlevel% equ 0 (
        echo Downloaded stb_truetype.h
    ) else (
        echo Failed to download stb_truetype.h
    )
    
    echo Downloading stb_image.h...
    powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/nothings/stb/master/stb_image.h' -OutFile '%INCLUDE_DIR%\stb_image.h'"
    if %errorlevel% equ 0 (
        echo Downloaded stb_image.h
    ) else (
        echo Failed to download stb_image.h
    )
    
    echo Downloading stb_image_resize2.h...
    powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/nothings/stb/master/stb_image_resize2.h' -OutFile '%INCLUDE_DIR%\stb_image_resize2.h'"
    if %errorlevel% equ 0 (
        echo Downloaded stb_image_resize2.h
    ) else (
        echo Failed to download stb_image_resize2.h
    )
    
    echo Downloading stb_image_write.h...
    powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h' -OutFile '%INCLUDE_DIR%\stb_image_write.h'"
    if %errorlevel% equ 0 (
        echo Downloaded stb_image_write.h
    ) else (
        echo Failed to download stb_image_write.h
    )
    
    echo ================================
    echo Download complete!
    echo Headers installed to: %INCLUDE_DIR%
)

pause
