# Window Arena portable Win64 package design

This specification adds a reproducible portable package for Window Arena. The
package runs on Windows 10 or Windows 11 x64 without an installed MSYS2 or
MinGW environment. It follows the recursive `ldd` dependency collection used
by `D:/Dev-Project/vala/test1/script/win32_res.sh` and adds dependency auditing,
assets, licenses, ZIP generation, and an isolated runtime test.

> **Note:** This is an experimental game package under active development.

## Goals

The package must provide a self-contained game directory and a ZIP archive.
The implementation must meet these requirements:

- Build `multi_window_arena_demo.exe` before packaging.
- Discover runtime DLLs from the executable and every copied DLL recursively.
- Run dependency discovery in an MSYS2 UCRT64 environment.
- Copy non-system DLLs from the build directory and `/ucrt64/bin`.
- Include the WebGPU runtime, both optional App window backends, and the MinGW
  thread runtime.
- Include the Window Arena font and its license.
- Record copied dependencies, source paths, sizes, and SHA-256 hashes.
- Generate `dist/window-arena-win64/` and
  `dist/window-arena-win64.zip`.
- Launch the packaged game with a `PATH` that does not contain MSYS2 or MinGW.

The package targets Windows 10 and Windows 11 x64. It does not produce an MSI,
a single-file self-extracting executable, or packages for other operating
systems.

## Output layout

The portable directory uses this structure:

```text
dist/window-arena-win64/
  WindowArena.exe
  wgpu_native.dll
  glfw3.dll
  SDL3.dll
  libwinpthread-1.dll
  VCRUNTIME140.dll
  dependency-manifest.txt
  assets/
    window_arena/
      Oxanium-VariableFont_wght.ttf
      OFL.txt
  licenses/
    third-party-notices.txt
```

The exact DLL set can grow when a dependency adds imports. The collector must
derive the set instead of relying on the example list above.

## Dependency collection

The repository adds a Bash packaging script under `tools/packaging/`. The
script receives the built executable, output directory, asset directory, and
optional dependency search directories.

The script sets an explicit UCRT64 search environment before invoking `ldd`:

```bash
PATH=/ucrt64/bin:/usr/bin
```

For each input PE file, the collector performs these steps:

1. Run `ldd` and parse resolved absolute DLL paths.
2. Reject unresolved non-system imports.
3. Normalize each path with `realpath`.
4. Skip dependencies already present in the visited set.
5. Classify Windows system and API-set libraries without copying them.
6. Copy every distributable runtime DLL into the package root.
7. Recurse into each copied DLL.

System classification uses resolved path prefixes and case-insensitive DLL
names. It excludes core libraries under Windows `System32`, `SysWOW64`, and
`WinSxS`, plus API-set loader contracts. It does not use a broad deletion pass
after copying because that can silently remove a required runtime.

`VCRUNTIME140.dll` is an explicit app-local runtime exception. The pinned
`wgpu_native.dll` imports it, so the collector includes the resolved x64 copy
and records its source in the manifest. This keeps the portable package from
depending on an independently installed Visual C++ runtime.

## CMake integration

CMake adds a `package_window_arena_win64` custom target on Windows. The target
depends on `multi_window_arena_demo` and the existing `fs_wgpu_runtime` target.

The configure step derives the MSYS2 root from `CMAKE_C_COMPILER`, then checks
for `usr/bin/bash.exe`. If Bash or `ldd` is unavailable, CMake reports a clear
packaging error without affecting normal builds.

The packaging target performs these operations:

1. Remove only the known package staging directory.
2. Run the Bash collector with the UCRT64 environment.
3. Copy and rename the game executable to `WindowArena.exe`.
4. Copy assets and license notices.
5. Write the dependency manifest.
6. Create a deterministic ZIP with `cmake -E tar`.
7. Run the isolated package smoke test.

Normal build targets do not package automatically. You invoke the workflow
explicitly with:

```powershell
cmake --build build --target package_window_arena_win64
```

## Manifest and licenses

`dependency-manifest.txt` contains the package target, build timestamp, source
executable, and one sorted entry for every copied file. Each dependency entry
includes its package name, resolved source path, byte size, and SHA-256 hash.

The package keeps the Oxanium OFL beside the font for runtime lookup and copies
a concise third-party notice into `licenses/`. The notice identifies GLFW,
SDL3, wgpu-native, tinyfiledialogs, and the Oxanium font with their upstream
projects and licenses. Packaging must not download licenses at build time.

## Runtime verification

The package test must not inherit an MSYS2 or MinGW search path. It starts a
new process with a restricted path containing only the package directory and
Windows system directories:

```text
<package>;C:\Windows\System32;C:\Windows
```

The process runs with `FS_APP_BACKEND=glfw` and
`FS_WINDOW_ARENA_AUTOTEST=1`, then repeats with `FS_APP_BACKEND=sdl3`. Both
processes must exit with code zero within the existing smoke-test timeout.

The verifier also runs `ldd` for audit purposes. Any unresolved import that is
not a Windows API-set contract fails packaging. The verifier confirms that the
font, OFL, manifest, and required runtime DLLs exist before it launches the
game.

## Failure handling

The packaging script uses strict shell options and exits on missing inputs,
unresolved dependencies, copy failures, missing assets, manifest failures, or
runtime smoke-test failures. It validates the resolved output path before any
recursive removal.

Normal compilation remains usable when packaging prerequisites are missing.
Only the explicit packaging target fails, and its message identifies the
missing tool or dependency.

## Verification

Implementation verification must cover these checks:

- Build the complete GLFW plus SDL3 configuration.
- Generate the portable directory and ZIP from a clean staging path.
- Confirm the ZIP contains the same relative files as the staging directory.
- Confirm all non-system `ldd` dependencies resolve inside the package.
- Run both GLFW and SDL3 automated game flows with a clean path.
- Confirm no packaged file resolves from `/ucrt64/bin` during the smoke test.
- Run `git diff --check` and the existing architecture-layer test.

## Next steps

Implement the Bash dependency collector first, then connect it to CMake and add
the isolated smoke verifier. Generate the final package only after both game
backends pass from the staging directory.
