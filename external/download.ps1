# WebGPU-Native Download Script (PowerShell)
# Set WGPU_VER to override the default version.

$ErrorActionPreference = "Stop"

$WGPU_VER = if ($env:WGPU_VER) { $env:WGPU_VER } else { "v29.0.1.1" }
$WGPU_REPO = "https://github.com/gfx-rs/wgpu-native"
$WGPU_API_REPO = "https://api.github.com/repos/gfx-rs/wgpu-native"
$WEBGPU_HEADERS_REPO = "https://raw.githubusercontent.com/webgpu-native/webgpu-headers"
$REQUEST_HEADERS = @{
    Accept = "application/vnd.github+json"
    "User-Agent" = "WCN-wgpu-downloader"
}

function Invoke-RequiredDownload {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$Description
    )

    try {
        Invoke-WebRequest -Uri $Url -OutFile $Destination -Headers $REQUEST_HEADERS
    }
    catch {
        throw "$Description download failed. Check the network connection and GitHub availability. URL: $Url. $($_.Exception.Message)"
    }

    $downloadedFile = Get-Item -LiteralPath $Destination -ErrorAction SilentlyContinue
    if (-not $downloadedFile -or $downloadedFile.Length -eq 0) {
        throw "$Description download produced an empty file. URL: $Url"
    }
}

Write-Host "WebGPU-Native Download Script  v=$WGPU_VER" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green

$platforms = @{
    "1" = @{ Name = "Windows x86_64 (MSVC)"; File = "wgpu-windows-x86_64-msvc-release.zip" }
    "2" = @{ Name = "Windows x86_64 (GNU)"; File = "wgpu-windows-x86_64-gnu-release.zip" }
    "3" = @{ Name = "Windows i686 (MSVC)"; File = "wgpu-windows-i686-msvc-release.zip" }
    "4" = @{ Name = "Windows aarch64 (MSVC)"; File = "wgpu-windows-aarch64-msvc-release.zip" }
    "5" = @{ Name = "Linux x86_64"; File = "wgpu-linux-x86_64-release.zip" }
    "6" = @{ Name = "MacOS x86_64"; File = "wgpu-macos-x86_64-release.zip" }
    "7" = @{ Name = "Android aarch64"; File = "wgpu-android-aarch64-release.zip" }
}

Write-Host "Please select platform:" -ForegroundColor Yellow
foreach ($key in $platforms.Keys | Sort-Object) {
    Write-Host "$key. $($platforms[$key].Name)"
}

$choice = Read-Host "Please enter option number (1-7)"
if (-not $platforms.ContainsKey($choice)) {
    Write-Host "Invalid choice!" -ForegroundColor Red
    exit 1
}

$selectedPlatform = $platforms[$choice]
$filename = $selectedPlatform.File
$downloadUrl = "$WGPU_REPO/releases/download/$WGPU_VER/$filename"
$encodedVersion = [Uri]::EscapeDataString($WGPU_VER)
$headersApiUrl = "$WGPU_API_REPO/contents/ffi/webgpu-headers?ref=$encodedVersion"
$wgpuHeaderUrl = "https://raw.githubusercontent.com/gfx-rs/wgpu-native/$WGPU_VER/ffi/wgpu.h"

Write-Host "You selected: $($selectedPlatform.Name)" -ForegroundColor Cyan
Write-Host "Resolving webgpu-headers revision for $WGPU_VER..." -ForegroundColor Yellow

try {
    $submoduleEntry = Invoke-RestMethod -Uri $headersApiUrl -Headers $REQUEST_HEADERS
}
catch {
    Write-Host "Failed to resolve webgpu-headers revision. Check the network connection, GitHub availability, and WGPU_VER ($WGPU_VER)." -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

$webgpuHeadersSha = [string]$submoduleEntry.sha
if ($webgpuHeadersSha -notmatch '^[0-9a-fA-F]{40}$') {
    Write-Host "GitHub returned an invalid webgpu-headers SHA for ${WGPU_VER}: '$webgpuHeadersSha'" -ForegroundColor Red
    exit 1
}

$webgpuHeaderUrl = "$WEBGPU_HEADERS_REPO/$webgpuHeadersSha/webgpu.h"
Write-Host "Resolved webgpu-headers SHA: $webgpuHeadersSha" -ForegroundColor Green

$wgpuDir = Join-Path $PSScriptRoot "wgpu"
New-Item -ItemType Directory -Path $wgpuDir -Force | Out-Null

$archivePath = Join-Path $wgpuDir ".$filename.part"
$stagedHeadersDir = Join-Path $wgpuDir ".webgpu-headers-$PID"
$headersDir = Join-Path $wgpuDir "include/webgpu"
$headersBackupDir = Join-Path $wgpuDir ".webgpu-headers-backup-$PID"

try {
    if (Test-Path -LiteralPath $stagedHeadersDir) {
        Remove-Item -LiteralPath $stagedHeadersDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $stagedHeadersDir -Force | Out-Null

    Invoke-RequiredDownload -Url $wgpuHeaderUrl `
        -Destination (Join-Path $stagedHeadersDir "wgpu.h") `
        -Description "wgpu.h"
    Invoke-RequiredDownload -Url $webgpuHeaderUrl `
        -Destination (Join-Path $stagedHeadersDir "webgpu.h") `
        -Description "webgpu.h"

    Write-Host "Downloading $filename..." -ForegroundColor Yellow
    Invoke-RequiredDownload -Url $downloadUrl -Destination $archivePath -Description $filename

    Write-Host "Extracting $filename..." -ForegroundColor Yellow
    Expand-Archive -LiteralPath $archivePath -DestinationPath $wgpuDir -Force

    if (Test-Path -LiteralPath $headersBackupDir) {
        Remove-Item -LiteralPath $headersBackupDir -Recurse -Force
    }
    if (Test-Path -LiteralPath $headersDir) {
        Move-Item -LiteralPath $headersDir -Destination $headersBackupDir
    }

    try {
        Move-Item -LiteralPath $stagedHeadersDir -Destination $headersDir
    }
    catch {
        if ((Test-Path -LiteralPath $headersBackupDir) -and -not (Test-Path -LiteralPath $headersDir)) {
            Move-Item -LiteralPath $headersBackupDir -Destination $headersDir
        }
        throw
    }

    if (Test-Path -LiteralPath $headersBackupDir) {
        Remove-Item -LiteralPath $headersBackupDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $archivePath -Force -ErrorAction SilentlyContinue

    Write-Host "Installed WebGPU-Native $WGPU_VER" -ForegroundColor Green
    Write-Host "webgpu-headers SHA: $webgpuHeadersSha" -ForegroundColor Green
}
catch {
    Remove-Item -LiteralPath $archivePath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stagedHeadersDir -Recurse -Force -ErrorAction SilentlyContinue
    if ((Test-Path -LiteralPath $headersBackupDir) -and -not (Test-Path -LiteralPath $headersDir)) {
        Move-Item -LiteralPath $headersBackupDir -Destination $headersDir -ErrorAction SilentlyContinue
    }
    Write-Host "Installation failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
