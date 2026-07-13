# Multi-window Window Arena design

This specification adds a playable multi-window game demo to the optional App
extension. The demo uses real desktop windows as game spaces, communicates
through stable window IDs, and keeps all window-system dependencies behind the
App backend contract. Root `include/` and `src/` remain independent of App.

> **Note:** This is an experimental demo and App API extension under active
> development.

## Goals

The implementation must prove that WCN can run a coordinated game across
multiple real windows without exposing GLFW or SDL3 to game code. The finished
demo must provide these behaviors:

- The main window contains a clickable **Create rift** button.
- Clicking the button creates a real child window on the desktop.
- The main window and up to three rift windows share one authoritative game
  world.
- Windows exchange versioned messages addressed by stable `FS_AppWindowId`
  values.
- Projectiles cross between overlapping windows while preserving desktop
  position and direction.
- Shooting a non-overlapping window edge expands that edge temporarily.
- The user can move, resize, minimize, restore, and close child windows without
  corrupting the game world.
- GLFW and SDL3 run the same game code through backend-neutral APIs.
- Static or occluded windows do not produce unnecessary CPU or GPU frames.

The design takes inspiration from Windowkill's use of real desktop windows as
the play area. It does not copy Windowkill source code, names, characters,
visual assets, audio, progression, or numerical balance.

## Repository boundary

Production additions are limited to `app/`:

```text
app/include/fullstack_app_message.h
app/include/fullstack_app_window_control.h
app/src/fullstack_app_message.c
app/src/fullstack_app_window_control.c
app/impl/backends/glfw/
app/impl/backends/sdl3/
app/impl/backends/mock/
```

The game and its private protocol live under `examples/`, and verification
lives under `test/`. Root `include/` and `src/` must not include App headers or
link App implementation code.

## Architecture

`WindowArenaWorld` is the sole authority for simulation, score, damage,
projectile ownership, overlap routing, and window lifecycle. A window view
owns rendering resources and a snapshot received from the world, but it cannot
mutate world state directly.

The major units are:

- `FS_AppMessageQueue`: Copies, orders, and releases backend-neutral messages.
- `FS_AppWindowControlCapability`: Gets and changes desktop window geometry.
- `WindowArenaProtocol`: Defines fixed-layout, pointer-free game messages.
- `WindowArenaWorld`: Runs deterministic fixed-step gameplay.
- `WindowArenaView`: Owns one `FS_AppWindow`, one render context, and one
  renderable snapshot.
- `multi_window_arena_demo`: Translates App events to protocol commands,
  advances the world, applies approved window operations, and renders dirty
  views.

## App message bus

The App extension exposes a same-process message bus with an IPC-compatible
wire shape:

```c
typedef struct FS_AppMessage {
    uint32_t struct_size;
    uint32_t protocol;
    uint32_t type;
    uint32_t flags;
    uint64_t sequence;
    uint64_t timestamp_ns;
    uint64_t correlation_id;
    FS_AppWindowId source_window_id;
    FS_AppWindowId target_window_id;
    FS_AppOwnedBytes payload;
} FS_AppMessage;
```

The public API provides posting, polling, payload copying, and release
operations. Posting assigns a global sequence and monotonic timestamp. Target
ID zero addresses the application coordinator, and `UINT64_MAX` represents a
broadcast. Other values address a specific window.

The queue is thread-safe and bounded. It deep-copies owned payloads. State
snapshot messages may coalesce by protocol, type, and target. Commands,
creation requests, damage, and lifecycle messages never coalesce. Posting to a
destroyed target returns or records a skipped delivery instead of dereferencing
window memory.

## Window Arena protocol

The example protocol uses version 1 and fixed-size payloads with no native
handles or pointers. It defines these message types:

- `CREATE_RIFT_REQUEST`
- `WINDOW_CREATED`
- `WINDOW_READY`
- `INPUT_COMMAND`
- `WINDOW_GEOMETRY_OBSERVED`
- `FIRE_PROJECTILE`
- `EDGE_IMPACT`
- `PROJECTILE_TRANSFERRED`
- `DAMAGE_APPLIED`
- `WINDOW_CLEAR`
- `WINDOW_CLOSE_INTENT`
- `WINDOW_DESTROY_APPROVED`
- `WORLD_SNAPSHOT`
- `GAME_OVER`

Each payload contains the protocol version, world tick, and source window
epoch. The coordinator rejects unknown versions, stale ticks where ordering is
required, and epochs that belong to a destroyed window generation.

## Window control capability

The App extension adds `FS_APP_CAPABILITY_WINDOW_CONTROL`. Its desktop geometry
uses signed logical coordinates so a window can occupy a monitor to the left or
above the primary monitor.

```c
typedef struct FS_AppWindowGeometry {
    uint32_t struct_size;
    int32_t desktop_x;
    int32_t desktop_y;
    uint32_t logical_width;
    uint32_t logical_height;
} FS_AppWindowGeometry;
```

The public helpers get and set geometry, change visibility, and raise a window.
GLFW implements them with its window position, size, visibility, and focus
functions. SDL3 implements them with the equivalent SDL window APIs. The Mock
backend stores geometry in memory and lets tests inject observed changes.

App adds `FS_APP_EVENT_WINDOW_MOVED`. A programmatic geometry request remains
an intent; the next observed backend geometry is authoritative because a
desktop window manager can constrain or alter the request.

Child windows are created hidden, positioned, initialized, rendered once, and
then shown. This prevents a visible jump from a backend-selected initial
position.

## Coordinates and overlap

Simulation uses window-local logical pixels. Window intersection and routing
use virtual-desktop logical pixels. Rendering converts local logical values to
framebuffer pixels with the current content scale.

For a local point `(local_x, local_y)`, the global point is:

```text
global_x = window.desktop_x + local_x
global_y = window.desktop_y + local_y
```

When a projectile reaches an owner window edge, the world converts the impact
to desktop coordinates. If that point lies in an overlapping target window,
the projectile changes owner and converts back to target-local coordinates. If
several windows qualify, the lowest stable window ID wins to keep simulation
deterministic. Without a target, the projectile produces an edge impulse and
is destroyed.

## Gameplay

The main window starts at 760 by 520 logical pixels. It contains the player,
HUD, and **Create rift** button. The play area contracts at a slow fixed rate.
Shots against its boundary expand the impacted edge. If the window reaches its
minimum safe size, continued pressure damages the player.

Each rift window starts near 380 by 280 logical pixels and contains tracking
enemies or one pulse enemy. Clearing a rift starts a 750 millisecond collapse
animation and then closes the window. Manually closing a child window safely
removes its entities but awards reduced or no clear reward.

Keyboard input from any focused game window controls the same player. Pointer
input is converted through the source window geometry, which lets the player
aim in desktop coordinates. The first release supports movement, aiming,
shooting, health, score, a restart action, two enemy behaviors, and up to four
windows. It does not add a roguelike shop, permanent upgrades, multiplayer, or
audio.

## Lifecycle

Each view moves through `CREATING`, `INITIALIZING`, `ACTIVE`,
`CLOSING_ANIMATION`, `DESTROY_PENDING`, and `DESTROYED`. A system close request
becomes `WINDOW_CLOSE_INTENT`; it does not immediately free the view. The world
stops new spawns, resolves owned entities, and emits
`WINDOW_DESTROY_APPROVED`. The coordinator then releases the render context,
destroys the App window, and drops remaining messages for its stable ID.

Closing the main window stops new messages and simulation, destroys every child
view, releases App and GPU resources, and unregisters the selected backend.
Creation failure leaves the main window alive and displays a transient error.

## Scheduling and rendering

The world advances with a 60 Hz fixed step and caps catch-up at four ticks per
App frame. Frame delta is clamped to 100 milliseconds. Programmatic desktop
geometry writes are limited to 30 Hz to avoid overwhelming the window manager.

Each view owns a lazy `FS_RenderContext` because Surface size, generation,
format, recovery, and visibility are independent. A view requests another
animation frame only while it contains active motion or a transition. Geometry,
input, HUD changes, exposure, and snapshots request one redraw. Minimized,
occluded, unavailable, and zero-size windows retain dirty state without
acquiring a frame.

## Visual assets

The demo uses an original procedural "desktop rift radar" style. Near-black
blue surfaces, cyan player geometry, amber enemies, magenta rifts, thin white
pressure lines, overlap scan grids, and short projectile trails are rendered
with existing Core rectangles, circles, paths, clips, transforms, and text.
No external sprite art is required.

The only required external asset is the Oxanium variable font from Google
Fonts. The implementation vendors the exact font file and its SIL Open Font
License under `examples/assets/window_arena/`. CMake copies the font and license
to the runtime asset directory. The demo does not download assets at runtime.

The first release does not add audio because App and Core do not expose a
backend-neutral audio contract. A later audio extension can use independently
licensed CC0 sound assets without coupling this demo to a platform audio API.

## Implementation sequence

The implementation proceeds in this order:

1. Capture the existing GLFW-only and GLFW plus SDL3 build and test baseline.
2. Add the App message types, bounded queue, App lifecycle integration, and
   unit tests.
3. Add window geometry types, public helpers, movement events, and Mock
   controls.
4. Implement the GLFW and SDL3 window-control capabilities and shared backend
   contract tests.
5. Add the Window Arena protocol and deterministic headless world tests.
6. Add the multi-window view registry and per-window rendering bridge.
7. Add gameplay, button hit testing, edge impulses, overlap routing, collapse
   transitions, HUD, and responsive small-window rendering.
8. Vendor the font and license, add CMake runtime asset copying, and mark
   licensing metadata correctly.
9. Add an `FS_WINDOW_ARENA_AUTOTEST=1` mode that automatically creates, moves,
   overlaps, and closes a child window before exiting.
10. Run both build matrices, all tests, runtime smoke tests, `git diff --check`,
    and the architecture layer audit.

## Verification

Unit and integration tests must prove message ordering, owned payload lifetime,
queue pressure, snapshot coalescing, stale target rejection, negative desktop
coordinates, DPI-independent overlap, deterministic projectile routing,
programmatic and observed geometry, close races, and world determinism.

Runtime verification must prove that the button creates a real window, GLFW
and SDL3 use identical game code, projectiles cross overlapping windows,
manual movement and resize affect routing, child close is safe, the main close
releases all resources, and static windows return to on-demand waiting.

The architecture audit must prove that root `include/` and `src/` contain no
App message, window-control, GLFW, SDL3, or game dependencies.

## Next steps

Implement the App message bus first, then add window control and its Mock tests.
Keep the game logic headless until both infrastructure layers pass their unit
tests.
