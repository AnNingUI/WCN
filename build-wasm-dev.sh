#!/bin/bash

# WCN WebAssembly Development Build Script
# This script builds the WASM version and copies files to the examples directory

echo "Building WCN WebAssembly version..."

# Create build directory if it doesn't exist
if [ ! -d "build-wasm" ]; then
    mkdir "build-wasm"
fi

# Change to build directory
cd build-wasm

# Configure with Emscripten
echo "Configuring with Emscripten..."
emcmake cmake .. -DCMAKE_BUILD_TYPE=Debug

# Build WASM version
echo "Building WASM..."
cmake --build . --target wcn_wasm

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Build successful!"
    echo ""
    echo "Copying files to examples/wasm directory..."
    
    # Copy the built files to the examples directory
    cp "wcn.js" "../examples/wasm/" 2>/dev/null || true
    if [ -f "wcn.wasm" ]; then
        cp "wcn.wasm" "../examples/wasm/" 2>/dev/null || true
    fi
    
    echo "Files copied successfully!"
    echo ""
    echo "To test, start a local server:"
    echo "  cd examples/wasm"
    echo "  python -m http.server 8000"
    echo ""
    echo "Then open http://localhost:8000/simple_test.html in your browser"
else
    echo ""
    echo "Build failed"
    echo "Please check the error messages above"
fi