#!/bin/bash

# WebGPU-Native Download Script (Bash)
# ================================

echo -e "\033[0;32mWebGPU-Native Download Script (Bash)\033[0m"
echo "================================"

# Check system type
get_platform() {
    case "$(uname -s)" in
        Darwin*)
            echo "macos"
            ;;
        Linux*)
            echo "linux"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Get system architecture
get_arch() {
    case "$(uname -m)" in
        x86_64)
            echo "x86_64"
            ;;
        aarch64|arm64)
            echo "aarch64"
            ;;
        *)
            echo "x86_64"  # Default
            ;;
    esac
}

# Select default file based on system and architecture
SYSTEM=$(get_platform)
ARCH=$(get_arch)

echo -e "\033[0;33mDetected system: $SYSTEM ($ARCH)\033[0m"

# Define download options
declare -A platforms
platforms[1]="Windows x86_64 (MSVC)|wgpu-windows-x86_64-msvc-release.zip|https://github.com/gfx-rs/wgpu-native/releases/download/v25.0.2.2/wgpu-windows-x86_64-msvc-release.zip"
platforms[2]="Windows x86_64 (GNU)|wgpu-windows-x86_64-gnu-release.zip|https://github.com/gfx-rs/wgpu-native/releases/download/v25.0.2.2/wgpu-windows-x86_64-gnu-release.zip"
platforms[3]="Linux x86_64|wgpu-linux-x86_64-release.zip|https://github.com/gfx-rs/wgpu-native/releases/download/v25.0.2.2/wgpu-linux-x86_64-release.zip"
platforms[4]="MacOS x86_64|wgpu-macos-x86_64-release.zip|https://github.com/gfx-rs/wgpu-native/releases/download/v25.0.2.2/wgpu-macos-x86_64-release.zip"
platforms[5]="MacOS aarch64|wgpu-macos-aarch64-release.zip|https://github.com/gfx-rs/wgpu-native/releases/download/v25.0.2.2/wgpu-macos-aarch64-release.zip"
platforms[6]="Android aarch64|wgpu-android-aarch64-release.zip|https://github.com/gfx-rs/wgpu-native/releases/download/v25.0.2.2/wgpu-android-aarch64-release.zip"

# Show platform options
echo -e "\033[0;33mPlease select platform:\033[0m"
for i in "${!platforms[@]}"; do
    name=$(echo "${platforms[$i]}" | cut -d'|' -f1)
    echo "$i. $name"
done

# Get user choice
read -p "Please enter option number (1-6): " choice

# Validate user choice
if [[ ! ${platforms[$choice]} ]]; then
    echo -e "\033[0;31mInvalid choice!\033[0m"
    exit 1
fi

# Get selected platform information
selected_platform=${platforms[$choice]}
name=$(echo "$selected_platform" | cut -d'|' -f1)
filename=$(echo "$selected_platform" | cut -d'|' -f2)
download_url=$(echo "$selected_platform" | cut -d'|' -f3)

echo -e "\033[0;36mYou selected: $name\033[0m"

# Create download directory
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
wgpu_dir="$script_dir/wgpu"
mkdir -p "$wgpu_dir"

# Download file
file_path="$wgpu_dir/$filename"
echo -e "\033[0;33mDownloading $filename...\033[0m"

# Try using curl or wget
if command -v curl &> /dev/null; then
    curl -L "$download_url" -o "$file_path"
elif command -v wget &> /dev/null; then
    wget "$download_url" -O "$file_path"
else
    echo -e "\033[0;31mError: curl or wget not found!\033[0m"
    exit 1
fi

if [ $? -eq 0 ]; then
    echo -e "\033[0;32mDownload complete!\033[0m"
else
    echo -e "\033[0;31mDownload failed!\033[0m"
    exit 1
fi

# Extract file
echo -e "\033[0;33mExtracting $filename...\033[0m"
if command -v unzip &> /dev/null; then
    unzip -o "$file_path" -d "$wgpu_dir"
    if [ $? -eq 0 ]; then
        echo -e "\033[0;32mExtraction complete!\033[0m"
    else
        echo -e "\033[0;31mExtraction failed!\033[0m"
        exit 1
    fi
else
    echo -e "\033[0;31mError: unzip tool not found!\033[0m"
    exit 1
fi

# Delete archive
rm "$file_path"
echo -e "\033[0;32mCleanup complete!\033[0m"

echo -e "\033[0;32mWebGPU-Native download and installation complete!\033[0m"