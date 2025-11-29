# WebGPU-Native Download Script (PowerShell)
# ================================

# https://github.com/gfx-rs/wgpu-native/blob/trunk/ffi/wgpu.h
# https://github.com/webgpu-native/webgpu-headers/blob/main/webgpu.h

Write-Host "WebGPU-Native Download Script (PowerShell)" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green

# Define platform options
$platforms = @{
    "1" = @{ Name = "Windows x86_64 (MSVC)"; File = "wgpu-windows-x86_64-msvc-release.zip"; Url = "https://github.com/gfx-rs/wgpu-native/releases/download/v27.0.2.0/wgpu-windows-x86_64-msvc-release.zip" }
    "2" = @{ Name = "Windows x86_64 (GNU)"; File = "wgpu-windows-x86_64-gnu-release.zip"; Url = "https://github.com/gfx-rs/wgpu-native/releases/download/v27.0.2.0/wgpu-windows-x86_64-gnu-release.zip" }
    "3" = @{ Name = "Windows i686 (MSVC)"; File = "wgpu-windows-i686-msvc-release.zip"; Url = "https://github.com/gfx-rs/wgpu-native/releases/download/v27.0.2.0/wgpu-windows-i686-msvc-release.zip" }
    "4" = @{ Name = "Windows aarch64 (MSVC)"; File = "wgpu-windows-aarch64-msvc-release.zip"; Url = "https://github.com/gfx-rs/wgpu-native/releases/download/v27.0.2.0/wgpu-windows-aarch64-msvc-release.zip" }
    "5" = @{ Name = "Linux x86_64"; File = "wgpu-linux-x86_64-release.zip"; Url = "https://github.com/gfx-rs/wgpu-native/releases/download/v27.0.2.0/wgpu-linux-x86_64-release.zip" }
    "6" = @{ Name = "MacOS x86_64"; File = "wgpu-macos-x86_64-release.zip"; Url = "https://github.com/gfx-rs/wgpu-native/releases/download/v27.0.2.0/wgpu-macos-x86_64-release.zip" }
    "7" = @{ Name = "Android aarch64"; File = "wgpu-android-aarch64-release.zip"; Url = "https://github.com/gfx-rs/wgpu-native/releases/download/v27.0.2.0/wgpu-android-aarch64-release.zip" }
}

$headerFiles = @{
    "wgpu.h"   = "https://github.com/gfx-rs/wgpu-native/raw/trunk/ffi/wgpu.h"
    "webgpu.h" = "https://github.com/webgpu-native/webgpu-headers/raw/bac520839ff5ed2e2b648ed540bd9ec45edbccbc/webgpu.h"
}



# Show platform options
Write-Host "Please select platform:" -ForegroundColor Yellow
foreach ($key in $platforms.Keys | Sort-Object) {
    Write-Host "$key. $($platforms[$key].Name)"
}

# Get user choice
$choice = Read-Host "Please enter option number (1-7)"

# Validate user choice
if (-not $platforms.ContainsKey($choice)) {
    Write-Host "Invalid choice!" -ForegroundColor Red
    exit 1
}

# Get selected platform information
$selectedPlatform = $platforms[$choice]
$filename = $selectedPlatform.File
$downloadUrl = $selectedPlatform.Url

Write-Host "You selected: $($selectedPlatform.Name)" -ForegroundColor Cyan

# Create download directory
$wgpuDir = Join-Path $PSScriptRoot "wgpu"
if (-not (Test-Path $wgpuDir)) {
    New-Item -ItemType Directory -Path $wgpuDir | Out-Null
}


$headersDir = Join-Path $wgpuDir "include/webgpu"
if (-not (Test-Path $headersDir)) {
    New-Item -ItemType Directory -Path $headersDir | Out-Null
}

foreach ($header in $headerFiles.GetEnumerator()) {
    $headerPath = Join-Path $headersDir $header.Key
    try {
        Invoke-WebRequest -Uri $header.Value -OutFile $headerPath
        Write-Host "Downloaded $($header.Key)" -ForegroundColor Green
    }
    catch {
        Write-Host "Failed to download $($header.Key): $_" -ForegroundColor Red
    }
}

Write-Host "Header files download complete!" -ForegroundColor Green

Write-Host "WebGPU-Native download and installation complete!" -ForegroundColor Green


# Download file
$filePath = Join-Path $wgpuDir $filename
Write-Host "Downloading $filename..." -ForegroundColor Yellow

try {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $filePath
    Write-Host "Download complete!" -ForegroundColor Green
}
catch {
    Write-Host "Download failed: $_" -ForegroundColor Red
    exit 1
}

# Extract file
Write-Host "Extracting $filename..." -ForegroundColor Yellow
try {
    Expand-Archive -Path $filePath -DestinationPath $wgpuDir -Force
    Write-Host "Extraction complete!" -ForegroundColor Green
}
catch {
    Write-Host "Extraction failed: $_" -ForegroundColor Red
    exit 1
}

# Delete archive
Remove-Item $filePath -Force
Write-Host "Cleanup complete!" -ForegroundColor Green

Write-Host "WebGPU-Native download and installation complete!" -ForegroundColor Green