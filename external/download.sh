#!/bin/bash
# WebGPU-Native Download Script (Bash)
# Set WGPU_VER to override the default version: WGPU_VER=v30.0.0 ./download.sh

set -uo pipefail

WGPU_VER="${WGPU_VER:-v29.0.1.1}"
WGPU_REPO="https://github.com/gfx-rs/wgpu-native"
WGPU_API_REPO="https://api.github.com/repos/gfx-rs/wgpu-native"
WEBGPU_HEADERS_REPO="https://raw.githubusercontent.com/webgpu-native/webgpu-headers"
USER_AGENT="WCN-wgpu-downloader"

die() {
    echo -e "\033[0;31mError: $*\033[0m" >&2
    exit 1
}

fetch_url() {
    local url="$1"
    if command -v curl &> /dev/null; then
        curl --fail --silent --show-error --location \
            -H "Accept: application/vnd.github+json" \
            -H "User-Agent: $USER_AGENT" \
            "$url"
    elif command -v wget &> /dev/null; then
        wget --quiet -O - \
            --header="Accept: application/vnd.github+json" \
            --user-agent="$USER_AGENT" \
            "$url"
    else
        return 127
    fi
}

download_file() {
    local url="$1"
    local destination="$2"
    local description="$3"

    if command -v curl &> /dev/null; then
        curl --fail --show-error --location \
            -H "User-Agent: $USER_AGENT" \
            "$url" -o "$destination" || return 1
    elif command -v wget &> /dev/null; then
        wget --user-agent="$USER_AGENT" "$url" -O "$destination" || return 1
    else
        return 127
    fi

    [[ -s "$destination" ]] || {
        echo "$description download produced an empty file: $url" >&2
        return 1
    }
}

echo -e "\033[0;32mWebGPU-Native Download Script  v=$WGPU_VER\033[0m"
echo "================================"

get_platform() {
    case "$(uname -s)" in
        Darwin*) echo "macos" ;;
        Linux*)  echo "linux" ;;
        *)       echo "unknown" ;;
    esac
}

get_arch() {
    case "$(uname -m)" in
        x86_64)        echo "x86_64" ;;
        aarch64|arm64) echo "aarch64" ;;
        *)             echo "x86_64" ;;
    esac
}

SYSTEM=$(get_platform)
ARCH=$(get_arch)
echo -e "\033[0;33mDetected: $SYSTEM ($ARCH)\033[0m"

declare -A platforms
platforms[1]="Windows x86_64 (MSVC)|wgpu-windows-x86_64-msvc-release.zip"
platforms[2]="Windows x86_64 (GNU)|wgpu-windows-x86_64-gnu-release.zip"
platforms[3]="Linux x86_64|wgpu-linux-x86_64-release.zip"
platforms[4]="MacOS x86_64|wgpu-macos-x86_64-release.zip"
platforms[5]="MacOS aarch64|wgpu-macos-aarch64-release.zip"
platforms[6]="Android aarch64|wgpu-android-aarch64-release.zip"

echo -e "\033[0;33mPlease select platform:\033[0m"
for i in "${!platforms[@]}"; do
    name=$(echo "${platforms[$i]}" | cut -d'|' -f1)
    echo "$i. $name"
done

read -r -p "Option (1-6): " choice
if [[ -z "${platforms[$choice]:-}" ]]; then
    die "Invalid choice."
fi

selected="${platforms[$choice]}"
name=$(echo "$selected" | cut -d'|' -f1)
filename=$(echo "$selected" | cut -d'|' -f2)
download_url="$WGPU_REPO/releases/download/$WGPU_VER/$filename"
headers_api_url="$WGPU_API_REPO/contents/ffi/webgpu-headers?ref=$WGPU_VER"
wgpu_header_url="https://raw.githubusercontent.com/gfx-rs/wgpu-native/$WGPU_VER/ffi/wgpu.h"

echo -e "\033[0;36mSelected: $name\033[0m"
echo -e "\033[0;33mResolving webgpu-headers revision for $WGPU_VER...\033[0m"

if ! api_response=$(fetch_url "$headers_api_url"); then
    die "Failed to resolve webgpu-headers revision. Check the network connection, GitHub availability, and WGPU_VER ($WGPU_VER)."
fi

webgpu_headers_sha=$(
    printf '%s\n' "$api_response" |
        grep -Eo '"sha"[[:space:]]*:[[:space:]]*"[0-9a-fA-F]{40}"' |
        head -n 1 |
        grep -Eo '[0-9a-fA-F]{40}' || true
)

if [[ ! "$webgpu_headers_sha" =~ ^[0-9a-fA-F]{40}$ ]]; then
    die "GitHub returned an invalid webgpu-headers SHA for $WGPU_VER: '$webgpu_headers_sha'"
fi

webgpu_header_url="$WEBGPU_HEADERS_REPO/$webgpu_headers_sha/webgpu.h"
echo -e "\033[0;32mResolved webgpu-headers SHA: $webgpu_headers_sha\033[0m"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
wgpu_dir="$script_dir/wgpu"
mkdir -p "$wgpu_dir"

archive_path="$wgpu_dir/.$filename.part"
staged_headers_dir=$(mktemp -d "$wgpu_dir/.webgpu-headers.XXXXXX") || \
    die "Failed to create the temporary header directory."
headers_dir="$wgpu_dir/include/webgpu"
headers_backup_dir="$wgpu_dir/.webgpu-headers-backup.$$"

cleanup() {
    rm -f "$archive_path"
    if [[ -n "${staged_headers_dir:-}" && -d "$staged_headers_dir" ]]; then
        rm -rf "$staged_headers_dir"
    fi
}
trap cleanup EXIT

download_file "$wgpu_header_url" "$staged_headers_dir/wgpu.h" "wgpu.h" || \
    die "wgpu.h download failed. Check the network connection and GitHub availability. URL: $wgpu_header_url"
download_file "$webgpu_header_url" "$staged_headers_dir/webgpu.h" "webgpu.h" || \
    die "webgpu.h download failed. Check the network connection and GitHub availability. URL: $webgpu_header_url"

echo -e "\033[0;33mDownloading $filename...\033[0m"
download_file "$download_url" "$archive_path" "$filename" || \
    die "$filename download failed. Check the network connection and GitHub availability. URL: $download_url"

echo -e "\033[0;33mExtracting $filename...\033[0m"
if command -v unzip &> /dev/null; then
    unzip -o "$archive_path" -d "$wgpu_dir" > /dev/null || \
        die "Extraction failed for $filename."
else
    die "unzip is required to extract $filename."
fi

if [[ -e "$headers_backup_dir" ]]; then
    rm -rf "$headers_backup_dir"
fi
if [[ -d "$headers_dir" ]]; then
    mv "$headers_dir" "$headers_backup_dir" || \
        die "Failed to stage the existing WebGPU headers."
fi

if ! mv "$staged_headers_dir" "$headers_dir"; then
    if [[ -d "$headers_backup_dir" && ! -e "$headers_dir" ]]; then
        mv "$headers_backup_dir" "$headers_dir" || true
    fi
    die "Failed to install the version-matched WebGPU headers."
fi
staged_headers_dir=""

rm -rf "$headers_backup_dir"
rm -f "$archive_path"
trap - EXIT

echo -e "\033[0;32mInstalled WebGPU-Native $WGPU_VER\033[0m"
echo -e "\033[0;32mwebgpu-headers SHA: $webgpu_headers_sha\033[0m"
