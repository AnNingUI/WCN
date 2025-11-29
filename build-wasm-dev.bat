@echo off
REM WCN WebAssembly Development Build Script
REM This script builds the WASM version and copies files to the examples directory

echo Building WCN WebAssembly version...

REM Create build directory if it doesn't exist
if not exist "build-wasm" mkdir "build-wasm"

REM Change to build directory
cd build-wasm

REM Configure with Emscripten
echo Configuring with Emscripten...
emcmake cmake .. -DCMAKE_BUILD_TYPE=Debug

REM Build WASM version
echo Building WASM...
cmake --build . --target wcn_wasm

REM Check if build was successful
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Build successful!
    echo.
    echo Copying files to examples/wasm directory...
    
    REM Copy the built files to the examples directory
    copy "wcn.js" "..\examples\wasm\" >nul
    if exist "wcn.wasm" copy "wcn.wasm" "..\examples\wasm\" >nul
    
    echo Files copied successfully!
    echo.
    echo To test, start a local server:
    echo   cd examples/wasm
    echo   python -m http.server 8000
    echo.
    echo Then open http://localhost:8000/simple_test.html in your browser
) else (
    echo.
    echo Build failed with error level %ERRORLEVEL%
    echo Please check the error messages above
)

pause