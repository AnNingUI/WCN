# App window appearance and Window Arena gameplay design

This specification adds backend-neutral runtime controls for window decoration,
whole-window opacity, and mouse passthrough to the optional App extension. It
also uses decoration and opacity in Window Arena to turn rift windows into
draggable, borderless game spaces with stability feedback. Root `include/` and
`src/` remain unchanged.

> **Note:** This is an experimental App API and game demo under active
> development.

## Goals

The implementation must provide these behaviors:

- App callers can read and change a window's decorated state at runtime.
- App callers can read and change whole-window opacity at runtime.
- App callers can read and change whole-window mouse passthrough at runtime.
- Existing backends that expose an older window-control capability remain safe.
- GLFW and SDL3 on Win32 implement all three controls.
- Unsupported platform operations return `FS_RESULT_UNSUPPORTED`.
- The Window Arena main window retains its system decoration.
- Window Arena rift windows become borderless and draggable with the right
  pointer button.
- Rift opacity communicates spawn, stability, impact, and collapse state.
- Window Arena does not enable mouse passthrough.

This iteration does not add per-pixel or region-based input hit testing. It
does not make the main game window borderless, and it does not add platform
window APIs to Core.

## Repository boundary

Production changes remain inside the App extension:

```text
app/include/fullstack_app_window_control.h
app/src/fullstack_app_window_control.c
app/impl/backends/glfw/
app/impl/backends/sdl3/
app/impl/backends/mock/
```

Window Arena integration remains under `examples/`, and verification remains
under `test/`. Root `include/` and `src/` must not include App, GLFW, SDL3, or
Win32 window-control code.

## Public App API

`FS_AppWindowControlCapability` remains the single capability for runtime
window management. The implementation appends six optional function pointers
to the end of the structure and raises its minor capability version from 1.0
to 1.1.

The public API adds these functions:

```c
FS_API FS_Result FS_CALL fs_app_window_get_decorated(
    FS_App* app, FS_AppWindow* window, bool* out_decorated,
    FS_Error* error);
FS_API FS_Result FS_CALL fs_app_window_set_decorated(
    FS_App* app, FS_AppWindow* window, bool decorated, FS_Error* error);

FS_API FS_Result FS_CALL fs_app_window_get_opacity(
    FS_App* app, FS_AppWindow* window, float* out_opacity,
    FS_Error* error);
FS_API FS_Result FS_CALL fs_app_window_set_opacity(
    FS_App* app, FS_AppWindow* window, float opacity, FS_Error* error);

FS_API FS_Result FS_CALL fs_app_window_get_mouse_passthrough(
    FS_App* app, FS_AppWindow* window, bool* out_enabled,
    FS_Error* error);
FS_API FS_Result FS_CALL fs_app_window_set_mouse_passthrough(
    FS_App* app, FS_AppWindow* window, bool enabled, FS_Error* error);
```

The decorated state controls system borders and the title bar. Opacity applies
one uniform value to the complete native window. Mouse passthrough controls
whether pointer hit testing passes through the complete window to windows
behind it.

`FS_AppWindowDesc.transparent` keeps its existing meaning. It requests a
per-pixel transparent framebuffer when the backend creates a window. It does
not replace runtime whole-window opacity.

## ABI compatibility

The capability query continues to request major version 1. Before App reads a
new function pointer, it uses `header.struct_size` and `fs_abi_has_field` with
the field's `offsetof` and size. A backend that returns the previous structure
size therefore receives no out-of-bounds reads. Calls to absent functions
return `FS_RESULT_UNSUPPORTED`.

Adding the fields at the structure tail keeps the existing function pointer
offsets stable. Backends that implement the new fields advertise capability
version 1.1 and set the header size to the complete structure size.

## Validation and state invalidation

All six operations require a non-null App, a non-null window owned by that App,
and execution on the App thread. Getters also require a non-null output
pointer and initialize the output before querying a backend.

Opacity must be finite and within the closed interval `[0.0f, 1.0f]`. The API
returns `FS_RESULT_INVALID_ARGUMENT` for NaN, infinity, and out-of-range values.
It does not clamp invalid input.

A successful decorated-state change requests
`FS_APP_REDRAW_WINDOW | FS_APP_REDRAW_LAYOUT | FS_APP_REDRAW_PAINT`. A window
manager may move the window or change its client area after decoration changes,
so the backend refreshes metrics and subsequent observed geometry is
authoritative. A caller that needs final bounds must call
`fs_app_window_get_geometry` again.

A successful opacity change requests paint invalidation. A successful mouse
passthrough change does not require visual invalidation.

## Backend implementation

The GLFW 3.4 backend maps the operations to these native APIs:

- Decoration uses `GLFW_DECORATED` with `glfwGetWindowAttrib` and
  `glfwSetWindowAttrib`.
- Opacity uses `glfwGetWindowOpacity` and `glfwSetWindowOpacity`.
- Mouse passthrough uses `GLFW_MOUSE_PASSTHROUGH` with the GLFW window
  attribute functions.

The SDL3 backend maps portable operations to these APIs:

- Decoration reads `SDL_GetWindowFlags` and writes with
  `SDL_SetWindowBordered`.
- Opacity uses `SDL_GetWindowOpacity` and `SDL_SetWindowOpacity`.

SDL3 does not expose a portable mouse-passthrough function. The current Win32
backend gets the native `HWND` from SDL window properties and calls a private
Win32 adapter. The adapter owns the required extended-window-style handling
and exposes a backend-private get/set pair. SDL3 builds without a platform
adapter return `FS_RESULT_UNSUPPORTED`. Future Cocoa, X11, and Wayland backends
can implement the same private adapter without changing the public App API.

The Mock backend stores `decorated`, `opacity`, and `mouse_passthrough` in each
mock window. Creation initializes them from `FS_AppWindowDesc.decorated`,
`1.0f`, and `false`, respectively. A successful setter updates the stored
value, and its getter returns the latest value.

Backend failures populate `FS_Error` with the backend domain and available
native error text. A backend must not report success if it cannot apply the
requested state.

## Window Arena lifecycle

The main Window Arena window retains its system decoration and existing window
manager behavior. Each rift follows this creation sequence:

1. Create the window hidden.
2. Set decoration to `false` and opacity to `0.0f`.
3. Apply the world-approved geometry.
4. Show the window.
5. Animate opacity toward its stability-derived target over about 220
   milliseconds with an emphasized deceleration curve.

If decoration or opacity is unsupported, the demo reports the failure once and
continues. It keeps the system decoration or falls back to canvas-only visual
effects. A missing visual capability must not terminate the game.

During collapse, the existing `collapse_fraction` drives both the canvas
collapse and whole-window fade. The coordinator destroys a rift only after the
world approves destruction. The final visible frames approach zero opacity
before the App releases the native window.

## Stability and impact feedback

The existing rift `edge_pressure` value drives stability:

```text
stability = 1 - clamp(edge_pressure, 0, 1)
```

The presentation maps pressure through a smooth curve rather than a direct
linear fade. Healthy rifts stay near `1.0f`; critical rifts approach a minimum
steady opacity of `0.68f`. The game keeps enemies, projectiles, and HUD content
readable at minimum stability.

The normal B-style feedback combines:

- Whole-window opacity in the `0.68f` to `1.0f` range.
- A low-frequency breathing pulse that increases with pressure.
- A rift border that changes from cyan toward warm amber and danger red.
- A visible stability meter derived from the same pressure value.

The transient C-style feedback activates after an edge impact for about 120 to
180 milliseconds. It adds a short opacity notch and canvas-local scanline
offsets. Critical pressure can trigger a deterministic, low-frequency fault
pulse. The implementation does not jitter native window geometry because that
would change overlap routing, projectile coordinates, and pointer aiming.

The demo caches the last applied native opacity. It calls the App setter only
when the value changes by at least approximately `1/255`, or when a lifecycle
boundary requires an exact value. This prevents redundant compositor calls.

## Borderless dragging

Only rift windows support application-controlled dragging. Pressing the right
pointer button stores the source window ID, its starting geometry, and the
desktop pointer origin. Pointer-move events calculate a desktop delta and call
`fs_app_window_set_geometry` with unchanged logical width and height.

Observed geometry remains authoritative. Move and resize events continue to
publish the existing `WA_MESSAGE_WINDOW_GEOMETRY_OBSERVED` payload, so the
world updates overlap regions and cross-window projectile routing while the
user drags a rift.

Releasing the right pointer button ends the drag. Destruction, focus loss, or a
missing view also cancels it. The main window does not intercept right-button
dragging. The demo does not enable mouse passthrough, because a passthrough
window could not receive the drag gesture or normal game input.

## Scheduling

Spawn, breathing, impact, critical fault, dragging, and collapse request
animation frames through the existing App scheduler. A stable rift with no
active presentation transition does not call native setters every frame.

The game simulation retains its fixed 60 Hz update. Window geometry writes
remain bounded so pointer motion and world actions cannot flood the native
window manager. Static, hidden, minimized, or occluded windows keep the
existing on-demand rendering behavior.

## Verification

App and Mock tests must cover these cases:

- Set/get round trips for all three controls.
- Opacity validation for NaN, infinity, negative values, and values above one.
- Wrong-thread, wrong-owner, null-argument, and unsupported results.
- A truncated version 1.0 capability that safely rejects each new call.
- State that changes only after a successful backend setter.

Shared backend contract tests must set and read back decoration, opacity, and
mouse passthrough for GLFW and SDL3 on Win32. Cleanup must restore opacity to
`1.0f` and disable passthrough even if an assertion path fails. A platform that
cannot support an operation must return `FS_RESULT_UNSUPPORTED` rather than a
false success.

Window Arena tests must verify these properties:

- Pressure and lifecycle mapping stays finite, bounded, and monotonic.
- Drag calculations change desktop position but preserve logical size.
- Edge impact starts a finite transient and does not change native geometry.
- Existing collapse, overlap, projectile-transfer, and world-determinism tests
  continue to pass.
- GLFW and SDL3 automated game smoke tests create, move, overlap, fade, and
  destroy a borderless rift without stalling.

The architecture audit must confirm that root `include/` and `src/` contain no
new App, GLFW, SDL3, Win32, or Window Arena dependencies. The demo and tests
must not include native backend headers.

## Implementation sequence

Implement and validate the feature in this order:

1. Extend the public capability and App wrappers with structure-tail checks.
2. Add Mock state and unit tests for validation and compatibility.
3. Add GLFW control functions and shared backend contract coverage.
4. Add SDL3 decoration and opacity functions plus the Win32 passthrough
   adapter.
5. Add deterministic Window Arena presentation and drag helpers.
6. Integrate borderless creation, opacity lifecycle, B-style stability, and
   C-style transient feedback into the game.
7. Extend the automated game smoke path and run both backend build matrices.
8. Run all tests, `git diff --check`, and the architecture layer audit.

## Next steps

Review this specification, then create a file-by-file implementation plan.
Implementation begins with the App capability and Mock contract before any
gameplay changes.
