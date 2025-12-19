# STB Libraries Download Script (PowerShell)
# STB Repository: https://github.com/nothings/stb

Write-Host "STB Libraries Download Script (PowerShell)" -ForegroundColor Green
Write-Host "================================"

# Create download directory
$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$StbDir = Join-Path $ScriptPath "stb"
$IncludeDir = Join-Path $StbDir "include"

if (-not (Test-Path $IncludeDir)) {
    New-Item -ItemType Directory -Force -Path $IncludeDir | Out-Null
}

# Define STB headers to download
$StbHeaders = @{
    "stb_truetype.h" = "https://raw.githubusercontent.com/nothings/stb/master/stb_truetype.h"
    "stb_image.h" = "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h"
    "stb_image_resize2.h" = "https://raw.githubusercontent.com/nothings/stb/master/stb_image_resize2.h"
    "stb_image_write.h" = "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h"
}

Write-Host "Downloading STB header files..." -ForegroundColor Yellow

# Download each header file
foreach ($header in $StbHeaders.Keys) {
    $downloadUrl = $StbHeaders[$header]
    $headerPath = Join-Path $IncludeDir $header
    
    Write-Host "Downloading $header..." -ForegroundColor Yellow
    
    try {
        Invoke-WebRequest -Uri $downloadUrl -OutFile $headerPath -ErrorAction Stop
        Write-Host "Downloaded $header" -ForegroundColor Green
    }
    catch {
        Write-Host "Failed to download $header : $_" -ForegroundColor Red
    }
}

Write-Host "================================" -ForegroundColor Green
Write-Host "Download complete!" -ForegroundColor Green
Write-Host "Headers installed to: $IncludeDir" -ForegroundColor Green
