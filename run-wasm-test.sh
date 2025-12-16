#!/bin/bash

# ============================================================================
# WCN WASM Test Server
# ============================================================================
# This script starts a local HTTP server and opens the WASM test page.
# ============================================================================

echo "========================================"
echo "WCN WASM Test Server"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Python not found!"
    echo "Please install Python or use another HTTP server."
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "[OK] Python found"
$PYTHON_CMD --version
echo ""

echo "Starting HTTP server on port 8000..."
echo ""
echo "Test page will be available at:"
echo "  http://localhost:8000/examples/wasm/simple_test.html"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Python HTTP server
$PYTHON_CMD -m http.server 8000
