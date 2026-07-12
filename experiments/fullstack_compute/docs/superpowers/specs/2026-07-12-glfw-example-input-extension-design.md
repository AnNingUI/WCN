# GLFW example input extension design

This design removes direct GLFW input and window-state handling from regular
examples. The examples continue to use `FS_GlfwBackend` for window and WebGPU
ownership, while `fullstack_glfw_backend_ext.h` exposes normalized input,
timing, and viewport APIs that a future SDL3 backend can reproduce.

> **Note:** `wgpu_minimal_triangle.c` remains a low-level GLFW and WebGPU
> validation program. It is outside this migration.

## Goals

The extension creates one backend boundary for example application code.

The implementation must:

- Remove direct `glfw*` calls and `GLFW_*` constants from regular files
  under `examples/`.
- Preserve the current behavior of every migrated example.
- Provide polling APIs for simple C demos and event delivery for
  `ui/app.hpp`.
- Keep GLFW-specific callbacks, key translation, and window user-pointer
  ownership inside the extension implementation.
- Define normalized types that an SDL3 extension can reproduce without
  exposing SDL or GLFW constants.
- Keep window creation, WebGPU surface management, and presentation in the
  existing `fullstack_glfw_backend.h/.c` implementation.

The migration does not redesign demo controls, add new input features, or
change `wgpu_minimal_triangle.c`.

## Architecture

The implementation adds these files:

- `impl/fullstack_glfw_backend_ext.h` defines the public extension API and
  normalized input types.
- `impl/fullstack_glfw_backend_ext.c` owns GLFW polling, callbacks, key
  translation, and event dispatch.

`FS_GlfwBackendExt` attaches to an initialized `FS_GlfwBackend`. It does
not own the backend or its WebGPU resources. Callers initialize the extension
after `fs_glfw_backend_init()` and detach it before
`fs_glfw_backend_shutdown()`.

The extension becomes the only regular example layer that accesses
`backend.window`. The existing GLFW backend implementation and the excluded
minimal triangle can continue to access GLFW directly.

## Normalized types

The public header defines backend-neutral values with an `FS_Backend`
prefix. These names describe application semantics rather than GLFW values.

The header also provides a complete `FS_GlfwBackendExt` structure so C callers
can allocate it on the stack and `ui::App` can store it by value. The structure
contains normalized current and previous input state, the attached backend
pointer, the application event callback, and an opaque `platform_state`
pointer for saved GLFW callbacks and user-pointer data.

### Keys

`FS_BackendKey` includes only keys currently used by the examples and UI
framework:

- Letters `A` through `Z`.
- Digits `0` through `9`.
- Arrow keys.
- `Backspace`, `Enter`, `Escape`, `Space`, and `Tab`.
- An `Unknown` value for unmapped backend keys.

The enum can grow when an example requires another semantic key. Example code
must not depend on the enum's numeric values.

### Pointer and button state

`FS_BackendMouseButton` initially defines left, right, and middle buttons.
The API reports buttons as boolean down states and normalized pressed or
released events.

`FS_BackendPointerState` contains window-space cursor coordinates and the
current, pressed, and released button masks. Framebuffer scaling remains
available through viewport metrics so each example can convert coordinates as
required.

The extension also provides explicit `mouse_button_down()`,
`mouse_button_pressed()`, and `mouse_button_released()` queries. This removes
per-example previous-button variables without forcing callers to decode masks.

### Viewport and time

`FS_BackendViewport` contains:

- Window width and height.
- Framebuffer width and height.
- Framebuffer scale on each axis.

`fs_glfw_backend_ext_time_seconds()` returns monotonic seconds. Example code
must not call `glfwGetTime()`.

### Events

`FS_BackendEvent` supports:

- Pointer movement.
- Pointer button press and release.
- Scroll deltas.
- Key press, repeat, and release.
- UTF-32 text input.
- Viewport resize.

Events contain normalized values only. They don't expose `GLFWwindow*`,
GLFW key codes, scan codes, or modifier constants.

## Public API

The extension uses an explicit state object:

```c
typedef struct FS_GlfwBackendExt {
    FS_GlfwBackend* backend;
    void* platform_state;
    FS_BackendEventCallback event_callback;
    void* event_user_data;
    FS_BackendViewport viewport;
    FS_BackendPointerState pointer;
    uint8_t keys_current[FS_BACKEND_KEY_COUNT];
    uint8_t keys_previous[FS_BACKEND_KEY_COUNT];
    bool initialized;
} FS_GlfwBackendExt;

bool fs_glfw_backend_ext_init(
    FS_GlfwBackendExt* ext,
    FS_GlfwBackend* backend
);

void fs_glfw_backend_ext_shutdown(FS_GlfwBackendExt* ext);
void fs_glfw_backend_ext_poll_events(FS_GlfwBackendExt* ext);
double fs_glfw_backend_ext_time_seconds(const FS_GlfwBackendExt* ext);

bool fs_glfw_backend_ext_get_viewport(
    const FS_GlfwBackendExt* ext,
    FS_BackendViewport* out_viewport
);

bool fs_glfw_backend_ext_get_pointer(
    const FS_GlfwBackendExt* ext,
    FS_BackendPointerState* out_pointer
);

bool fs_glfw_backend_ext_key_down(
    const FS_GlfwBackendExt* ext,
    FS_BackendKey key
);

bool fs_glfw_backend_ext_key_pressed(
    const FS_GlfwBackendExt* ext,
    FS_BackendKey key
);

bool fs_glfw_backend_ext_key_released(
    const FS_GlfwBackendExt* ext,
    FS_BackendKey key
);

bool fs_glfw_backend_ext_mouse_button_down(
    const FS_GlfwBackendExt* ext,
    FS_BackendMouseButton button
);

bool fs_glfw_backend_ext_mouse_button_pressed(
    const FS_GlfwBackendExt* ext,
    FS_BackendMouseButton button
);

bool fs_glfw_backend_ext_mouse_button_released(
    const FS_GlfwBackendExt* ext,
    FS_BackendMouseButton button
);

void fs_glfw_backend_ext_take_scroll(
    FS_GlfwBackendExt* ext,
    float* out_x,
    float* out_y
);

void fs_glfw_backend_ext_set_event_callback(
    FS_GlfwBackendExt* ext,
    FS_BackendEventCallback callback,
    void* user_data
);

void fs_glfw_backend_ext_request_close(FS_GlfwBackendExt* ext);
```

The final names can change during implementation when required for consistency
with existing backend naming. The semantic boundary must remain unchanged.

## Frame lifecycle

Each frame follows one order:

1. Call `fs_glfw_backend_ext_poll_events()`.
2. Read edge states, current pointer state, viewport metrics, and scroll
   deltas.
3. Update application state.
4. Record and present rendering commands.

The extension stores previous and current key and button states. Pressed and
released queries remain valid until the next poll call. This removes
per-example `prev_key` and `prev_lmb` variables.

The extension installs GLFW callbacks during initialization. It owns the GLFW
window user pointer and stores the caller's event callback separately. This
prevents the user-pointer conflict previously seen between
`FS_GlfwBackend` and `ui::App`.

## UI framework integration

`ui/app.hpp` stores an `FS_GlfwBackendExt` beside its
`FS_GlfwBackend`. The App no longer declares GLFW callback functions or maps
GLFW key constants.

The extension callback sends `FS_BackendEvent` values to App. App converts
them into the existing `wcn_ui::Event` representation. This keeps UI element
event processing backend-neutral.

The App reads framebuffer size and time through the extension. No GLFW type or
constant remains in App implementation code.

## C example migration

Simple C examples keep their current imperative update loops. They replace
direct GLFW access as follows:

- Window and framebuffer queries use `FS_BackendViewport`.
- Cursor and mouse-button queries use `FS_BackendPointerState`.
- Key edge variables use `fs_glfw_backend_ext_key_pressed()` and
  `fs_glfw_backend_ext_key_released()`.
- Held-key behavior uses `fs_glfw_backend_ext_key_down()`.
- Time uses `fs_glfw_backend_ext_time_seconds()`.
- Scroll uses `fs_glfw_backend_ext_take_scroll()`.
- Close actions use `fs_glfw_backend_ext_request_close()`.

This migration applies to all regular examples that currently contain GLFW
symbols, including layout, game, memory, radius, text-layout, and UI demos.

## Error handling

Initialization fails when the extension receives a null backend, the backend
has no window, private callback storage cannot be allocated, or callback
installation cannot establish a valid association.

The existing GLFW backend owns a window user pointer and scroll callback before
the extension attaches. Extension initialization must capture the previous user
pointer and every callback it replaces. If any initialization step fails, it
must restore all captured values before returning `false`.

Query functions return safe zero or false values when they receive invalid
arguments. Event callback failures don't interrupt the render loop because
callbacks return no value.

Shutdown must restore the captured backend user pointer and callbacks before it
releases private storage. This lets callers detach the extension before backend
shutdown without leaving the existing backend scroll callback paired with an
incompatible or null user pointer. Calling shutdown on a zero-initialized,
partially initialized, or already shut down extension must be safe.

## Build integration

`fullstack_glfw_backend_ext.c` joins `FS_BACKEND_SOURCES` so every regular
example receives the implementation automatically. No target-specific source
lists are required.

The extension remains a GLFW-specific adapter. A future SDL3 implementation
can provide equivalent normalized behavior through its own adapter. A later
step can promote the normalized types into a common platform header without
changing example event semantics.

## Verification

The migration is complete when all these checks pass:

- Every regular example target compiles and links.
- Existing controls still work in representative C and C++ demos.
- `ui/app.hpp` receives pointer, scroll, key, and text-input events.
- Window resizing reports correct logical and framebuffer dimensions.
- Pressed events fire once per transition, while held queries remain true.
- This search returns no matches in regular example source files. It explicitly
  excludes the low-level minimal triangle and CMake configuration:

  ```powershell
  rg -n '\bglfw[A-Z][A-Za-z0-9_]*\b|\bGLFW[A-Za-z0-9_]*\b|<GLFW/' examples `
    --glob '!wgpu_minimal_triangle.c' --glob '!CMakeLists.txt'
  ```

- Direct GLFW symbols remain limited to `fullstack_glfw_backend.h/.c`,
  `fullstack_glfw_backend_ext.c`, and `wgpu_minimal_triangle.c`. The extension
  header contains normalized declarations only.

- `morrow_clock_demo`, `transition_demo`, `fs_text_layout_demo`, and the
  full example build complete successfully.

## Next steps

After this design is approved, create an implementation plan that introduces
the extension, migrates `ui/app.hpp`, migrates C examples in small groups,
and runs the full build and symbol audit.
