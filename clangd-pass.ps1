# clangd-pass.ps1
# Generate compile_commands.json for clangd/IntelliSense support
# This script creates a compilation database without actually compiling

# ============================================================================
# Configuration
# ============================================================================

# Project root directory (current directory by default)
$ProjectRoot = $PSScriptRoot

# Compiler path
$Compiler = "D:\Software\Dev\msys64\ucrt64\bin\cc.exe"

# Include directories (relative to project root or absolute paths)
$IncludeDirs = @(
    "$ProjectRoot/external/wgpu/include",
    "$ProjectRoot/include",
    "$ProjectRoot/external/stb/include",
    "D:/Software/Dev/msys64/ucrt64/include/freetype2",
    "D:/Software/Dev/msys64/ucrt64/include"
)

# Source files to include in compile_commands.json
# Format: @{ File = "relative/path/to/file.h"; Output = "output.o" }
$SourceFiles = @(
    @{ File = "impl/wcn_freetype2_impl.h"; Output = "wcn_freetype2_impl.o" },
    @{ File = "impl/wcn_stb_truetype_impl.h"; Output = "wcn_stb_truetype_impl.o" },
    @{ File = "impl/wcn_stb_image_impl.h"; Output = "wcn_stb_image_impl.o" },
    @{ File = "impl/wcn_glfw_impl.h"; Output = "wcn_glfw_impl.o" }
)

# Output file path
$OutputFile = "$ProjectRoot/impl/compile_commands.json"

# Additional compiler flags (optional)
$AdditionalFlags = @()

# ============================================================================
# Script Logic
# ============================================================================

Write-Host "Generating compile_commands.json..." -ForegroundColor Cyan

# Build include flags
$IncludeFlags = $IncludeDirs | ForEach-Object {
    "-I$_"
}

# Build compilation database entries
$CompileCommands = @()

foreach ($source in $SourceFiles) {
    $filePathForward = "$ProjectRoot/$($source.File)"

    # Build command string
    $commandParts = @($Compiler, "-c") + $IncludeFlags + $AdditionalFlags + @("-o", $source.Output, $source.File)
    $command = $commandParts -join " "

    # Create entry
    $entry = [PSCustomObject]@{
        directory = $ProjectRoot -replace '\\', '/'
        command   = $command
        file      = $filePathForward -replace '\\', '/'
    }

    $CompileCommands += $entry
}

# Convert to JSON and write to file
$json = $CompileCommands | ConvertTo-Json -Depth 10
$json | Out-File -FilePath $OutputFile -Encoding UTF8

Write-Host "Successfully generated: $OutputFile" -ForegroundColor Green
Write-Host "Total entries: $($CompileCommands.Count)" -ForegroundColor Green

# Display summary
Write-Host "`nConfiguration Summary:" -ForegroundColor Yellow
Write-Host "  Compiler: $Compiler"
Write-Host "  Include Directories: $($IncludeDirs.Count)"
Write-Host "  Source Files: $($SourceFiles.Count)"
Write-Host "  Output: $OutputFile"
