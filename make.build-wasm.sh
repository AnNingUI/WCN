#!/bin/bash

# ============================================================================
# WCN WebAssembly Build Script
# ============================================================================
# This script builds WCN for WebAssembly using Emscripten.
# 
# Prerequisites:
#   - Emscripten SDK installed and activated
#   - Run: source /path/to/emsdk/emsdk_env.sh
#
# Usage:
#   ./build-wasm.sh [debug|release]
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default build type
BUILD_TYPE=${1:-Release}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}WCN WebAssembly Build${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if Emscripten is available
if ! command -v emcc &> /dev/null; then
    echo -e "${RED}Error: Emscripten not found!${NC}"
    echo "Please install and activate Emscripten SDK:"
    echo "  git clone https://github.com/emscripten-core/emsdk.git"
    echo "  cd emsdk"
    echo "  ./emsdk install latest"
    echo "  ./emsdk activate latest"
    echo "  source ./emsdk_env.sh"
    exit 1
fi

echo -e "${GREEN}✓ Emscripten found: $(emcc --version | head -n 1)${NC}"
echo ""

# Create build directory
BUILD_DIR="build-wasm"
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Cleaning existing build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo -e "${GREEN}Configuring CMake for WebAssembly...${NC}"
echo "Build type: $BUILD_TYPE"
echo ""

# Configure with Emscripten
emcmake cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DWCN_BUILD_WASM=ON \
    -DWCN_ENABLE_SIMD=ON

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Building WCN for WebAssembly...${NC}"
echo ""

# Build
emmake make wcn_wasm -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Output files:"
echo "  JavaScript : $BUILD_DIR/wcn.js"
echo "  WebAssembly: $BUILD_DIR/wcn.wasm"
echo ""

# Check file sizes
if [ -f "wcn.js" ] && [ -f "wcn.wasm" ]; then
    JS_SIZE=$(du -h wcn.js | cut -f1)
    WASM_SIZE=$(du -h wcn.wasm | cut -f1)
    echo "File sizes:"
    echo "  wcn.js  : $JS_SIZE"
    echo "  wcn.wasm: $WASM_SIZE"
    echo ""
fi

echo -e "${GREEN}To use in a web page:${NC}"
echo "  <script src=\"wcn.js\"></script>"
echo "  <script>"
echo "    createWCNModule().then(WCN => {"
echo "      // Use WCN here"
echo "      console.log('WCN loaded!');"
echo "    });"
echo "  </script>"
echo ""

echo -e "${GREEN}See docs/WASM_BUILD.md for more information.${NC}"
