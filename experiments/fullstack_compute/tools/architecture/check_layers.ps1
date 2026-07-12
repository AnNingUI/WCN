param(
    [Parameter(Mandatory = $true)]
    [string]$Root
)
$ErrorActionPreference = 'Stop'
$rootPath = (Resolve-Path -LiteralPath $Root).Path
$violations = [System.Collections.Generic.List[string]]::new()
function Get-SourceFiles {
    param([string[]]$Directories)
    foreach ($directory in $Directories) {
        $path = Join-Path $rootPath $directory
        if (-not (Test-Path -LiteralPath $path)) { continue }
        Get-ChildItem -LiteralPath $path -Recurse -File |
            Where-Object { $_.Extension -in '.c', '.cc', '.cpp', '.h', '.hpp', '.mm' }
    }
}
function Test-Rule {
    param([string]$Name, [string[]]$Directories, [string]$Pattern,
          [string[]]$AllowedRelativePaths)
    $allowed = @{}
    foreach ($item in $AllowedRelativePaths) { $allowed[$item.Replace('/', '\')] = $true }
    foreach ($file in Get-SourceFiles $Directories) {
        $relative = $file.FullName.Substring($rootPath.Length).TrimStart([char[]]@('\', '/'))
        $matches = Select-String -LiteralPath $file.FullName -Pattern $Pattern -AllMatches
        if (-not $matches -or $allowed.ContainsKey($relative)) { continue }
        foreach ($match in $matches) {
            $violations.Add("[$Name] $relative`:$($match.LineNumber): $($match.Line.Trim())")
        }
    }
}
# Known Phase 0 legacy violations are allowlisted only in their current files.
# Later phases remove entries; new files are never added to this list.
Test-Rule 'platform-symbol-in-core' @('include', 'src') '(?i)\b(GLFW|glfw|SDL_|ANativeWindow|CAMetalLayer|UIKit)\b' @()
Test-Rule 'surface-present-in-core' @('include', 'src') '\b(wgpuSurfacePresent|wgpuSurfaceGetCurrentTexture)\b' @()
Test-Rule 'concrete-backend-in-ui' @('examples/ui') '(?i)\b(GLFW|glfw|SDL_)' @('examples\ui\app.hpp', 'examples\ui\router.hpp')
Test-Rule 'core-rendering-in-platform-backend' @('impl') '\b(fs_core_encode|fs_effects_create_presentation_pipeline)\b' @('impl\fullstack_glfw_backend.c')
Test-Rule 'presentation-owned-by-effects' @('include', 'src') '\bfs_effects_create_presentation_pipeline\b' @('include\fullstack_effects.h', 'src\fullstack_effects.h', 'src\fullstack_effects.c')
if ($violations.Count -gt 0) {
    Write-Error (($violations | Sort-Object) -join [Environment]::NewLine)
    exit 1
}
Write-Output 'Architecture layer audit passed with the Phase 0 legacy allowlist.'
