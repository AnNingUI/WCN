#!/bin/bash

# STB Libraries Download Script (Bash)
# ================================

echo -e "\033[0;32mSTB Libraries Download Script (Bash)\033[0m"
echo "================================"

# Create download directory
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
stb_dir="$script_dir/stb"
include_dir="$stb_dir/include"
mkdir -p "$include_dir"

# Define STB headers to download
declare -A stb_headers
stb_headers[stb_truetype.h]="https://raw.githubusercontent.com/nothings/stb/master/stb_truetype.h"
stb_headers[stb_image.h]="https://raw.githubusercontent.com/nothings/stb/master/stb_image.h"
stb_headers[stb_image_resize2.h]="https://raw.githubusercontent.com/nothings/stb/master/stb_image_resize2.h"
stb_headers[stb_image_write.h]="https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h"

echo -e "\033[0;33mDownloading STB header files...\033[0m"

# Download each header file
for header in "${!stb_headers[@]}"; do
    download_url="${stb_headers[$header]}"
    header_path="$include_dir/$header"
    
    echo -e "\033[0;33mDownloading $header...\033[0m"
    
    # Try using curl or wget
    if command -v curl &> /dev/null; then
        curl -L "$download_url" -o "$header_path"
    elif command -v wget &> /dev/null; then
        wget "$download_url" -O "$header_path"
    else
        echo -e "\033[0;31mError: curl or wget not found!\033[0m"
        exit 1
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "\033[0;32mDownloaded $header\033[0m"
    else
        echo -e "\033[0;31mFailed to download $header\033[0m"
    fi
done

echo -e "\033[0;32m================================"
echo -e "Download complete!\033[0m"
echo -e "Headers installed to: $include_dir"
