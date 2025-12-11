@echo off
REM ============================================================================
REM WCN WebAssembly Build Script (Windows)
REM ============================================================================
REM This script builds WCN for WebAssembly using Emscripten on Windows.
REM 
REM Prerequisites:
REM   - Emscripten SDK installed and activated
REM   - Run: emsdk\emsdk_env.bat
REM
REM Usage:
REM   build-wasm.bat [Debug|Release]
REM ============================================================================

setlocal enabledelayedexpansion

REM Default build type
set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Release

echo ========================================
echo WCN WebAssembly Build
echo ========================================
echo.

REM Check if Emscripten is available
where emcc >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Emscripten not found!
    echo Please install and activate Emscripten SDK:
    echo   git clone https://github.com/emscripten-core/emsdk.git
    echo   cd emsdk
    echo   emsdk install latest
    echo   emsdk activate latest
    echo   emsdk_env.bat
    exit /b 1
)

echo [OK] Emscripten found
emcc --version | findstr /C:"emcc"
echo.

REM Create build directory
set BUILD_DIR=build-wasm
if exist "%BUILD_DIR%" (
    echo Cleaning existing build directory...
    rmdir /s /q "%BUILD_DIR%"
)

mkdir "%BUILD_DIR%"
cd "%BUILD_DIR%"

echo Configuring CMake for WebAssembly...
echo Build type: %BUILD_TYPE%
echo.

REM Configure with Emscripten
call emcmake cmake .. ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DWCN_BUILD_WASM=ON ^
    -DWCN_ENABLE_SIMD=ON ^
    -G "MinGW Makefiles"

if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed!
    exit /b 1
)

echo.
echo Building WCN for WebAssembly...
echo.

REM Build
call emmake make wcn_wasm -j%NUMBER_OF_PROCESSORS%

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b 1
)

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Output files:
echo   JavaScript : %BUILD_DIR%\wcn.js
echo   WebAssembly: %BUILD_DIR%\wcn.wasm
echo.

REM Check file sizes
if exist "wcn.js" (
    for %%A in (wcn.js) do echo   wcn.js:   %%~zA bytes
)
if exist "wcn.wasm" (
    for %%A in (wcn.wasm) do echo   wcn.wasm: %%~zA bytes
)
echo.

echo To use in a web page:
echo   ^<script src="wcn.js"^>^</script^>
echo   ^<script^>
echo     createWCNModule().then(WCN =^> {
echo       // Use WCN here
echo       console.log('WCN loaded!');
echo     });
echo   ^</script^>
echo.

echo See docs\WASM_BUILD.md for more information.

cd ..
