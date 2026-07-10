# Resolve version-matched WebGPU headers

The download scripts must install `wgpu-native`, `wgpu.h`, and `webgpu.h`
from compatible revisions. The scripts resolve the `webgpu-headers` submodule
commit pinned by the selected `wgpu-native` tag instead of using `trunk` or a
manually maintained hash.

## Goals

The implementation provides the same version-resolution behavior on Windows,
Linux, and macOS.

- Let `WGPU_VER` select an exact `wgpu-native` release tag.
- Download `wgpu.h` from that exact tag.
- Query the GitHub Contents API for the `ffi/webgpu-headers` submodule entry at
  that tag.
- Validate that the returned submodule SHA contains exactly 40 hexadecimal
  characters.
- Download `webgpu.h` from the resolved submodule SHA.
- Stop immediately when a network request, API response, archive extraction,
  hash validation, or header download fails.
- Preserve the existing valid headers until both replacement headers download
  successfully.

## Non-goals

This change does not select the newest `webgpu-headers/main` revision. It also
does not silently fall back to `trunk`, an older hard-coded hash, or headers
from another `wgpu-native` version.

## Version resolution

For a selected version such as `v29.0.1.1`, the scripts request this endpoint:

```text
https://api.github.com/repos/gfx-rs/wgpu-native/contents/ffi/webgpu-headers?ref=v29.0.1.1
```

The response describes the Git submodule entry. The scripts read its `sha`
field, validate it, and construct these header URLs:

```text
https://raw.githubusercontent.com/gfx-rs/wgpu-native/v29.0.1.1/ffi/wgpu.h
https://raw.githubusercontent.com/webgpu-native/webgpu-headers/<sha>/webgpu.h
```

The release archive continues to use the selected `WGPU_VER` and platform
filename.

## Platform behavior

`download.ps1` contains the Windows implementation. It uses
`Invoke-RestMethod` to resolve the submodule SHA and `Invoke-WebRequest` for
downloads. `download.bat` remains a small wrapper that passes `WGPU_VER` to
`download.ps1` and reports an error when PowerShell is unavailable.

`download.sh` implements the same flow with `curl` or `wget`. It parses the
single GitHub API `sha` field without requiring `jq`, then validates the result
before constructing the header URL.

## Atomic header replacement

Each script downloads `wgpu.h` and `webgpu.h` to temporary files in the header
directory. After both downloads succeed and produce non-empty files, the
script replaces the existing headers. A partial or failed download cannot
leave one old header paired with one new header.

## Failure handling

Every remote operation is mandatory. When a request fails, the script exits
with a nonzero status and identifies the failed URL or operation. API and
header failures include a message that points to network or GitHub
availability as the likely cause.

The scripts also fail when the API response has no submodule SHA or when the
SHA fails the 40-character hexadecimal validation.

## Verification

Verification covers behavior without replacing the installed SDK during
routine syntax checks.

1. Run PowerShell parser validation for `download.ps1`.
2. Run Bash syntax validation with `bash -n` for `download.sh` when Bash is
   available.
3. Inspect the batch wrapper and verify that it propagates the PowerShell exit
   code.
4. Query the API for `v29.0.1.1` and confirm that it resolves to
   `673658bc2bd70ec39fc55ebe6bb0173cf6d0a603`.
5. Verify that the generated raw header URLs return non-empty content.

## Next steps

Implement the shared resolution rules in all three scripts, run syntax and
network-resolution checks, and review the resulting Git diff.
