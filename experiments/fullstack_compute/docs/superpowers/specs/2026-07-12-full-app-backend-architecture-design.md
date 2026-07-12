# Full application backend architecture

## Summary

This specification replaces the GLFW-owned application path with a complete,
versioned C ABI for platform backends, GPU ownership, windows, events, frame
acquisition, rendering, presentation, and optional platform services. The
default path creates every required resource, while advanced hosts can inject
and borrow or transfer existing WebGPU and native-window resources.

The change is a full architectural refactor, not a compatibility wrapper. It
separates `FS_Core`, rendering, presentation, application lifecycle, and
platform backends so GLFW, SDL3, Android, and iOS can use the same upper layers.
Legacy GLFW APIs remain temporary facades over the new implementation.

## Goals

The architecture must:

- Support library-owned and host-injected GPU and native resources.
- Use a versioned C ABI vtable for built-in and custom platform backends.
- Provide both a convenient desktop run loop and externally driven step APIs.
- Support multiple windows that share a GPU context but own independent cores.
- Make events, input, viewport metrics, lifecycle, and time backend-neutral.
- Use explicit acquired-frame objects for encode, submit, present, and cancel.
- Recover from resize, suspension, outdated surfaces, and surface loss.
- Support asynchronous, capability-based platform services.
- Compile Core and offscreen rendering without GLFW, SDL3, or a window system.
- Provide real SDL3, Android, and iOS implementation paths.
- Remove platform, surface, and presentation responsibilities from `FS_Core`.

The architecture does not require a complex optimizing frame-graph compiler,
automatic device recreation, or unrestricted cross-thread window operations in
its first implementation.

## Architectural layers

The final ownership graph is:

```text
FS_App
|-- FS_AppBackend
|   |-- native lifecycle and windows
|   |-- surfaces
|   |-- platform event production
|   `-- optional platform services
|-- FS_GpuContext
|   |-- instance, adapter, device, and queue
|   |-- capability snapshot
|   `-- device-lost state
|-- FS_AppWindow (one or more)
|   |-- one surface
|   |-- one FS_RenderContext
|   |-- one independent FS_Core
|   `-- at most one acquired FS_AppFrame
|-- FS_EventQueue
`-- main-thread task queue
```

`FS_AppBackend` handles platform facts and presentation availability. It does
not perform UI layout or issue Core drawing commands. `FS_GpuContext` owns or
borrows the shareable WebGPU stack. Each visible window owns an independent
render context and Core because size-dependent textures, filters, and output
formats cannot safely be shared between windows.

`FS_Core` remains usable without `FS_App`. It records drawing semantics and
encodes a scene, but it does not own a surface, select a present mode, acquire a
surface texture, or call `wgpuSurfacePresent()`.

## Public modules

The public API is divided into focused headers:

```text
include/fullstack_result.h
include/fullstack_gpu.h
include/fullstack_core.h
include/fullstack_render.h
include/fullstack_present.h
include/fullstack_app_event.h
include/fullstack_app_backend.h
include/fullstack_app_services.h
include/fullstack_app.h
include/fullstack_legacy.h
```

Concrete GLFW, SDL3, Android, and iOS types never appear in the generic public
headers. Platform objects are represented by tagged opaque native handles.

Every extensible public descriptor starts with `struct_size`. Backend factories
and vtables also carry an ABI version. New fields are appended, and wrappers
validate the supplied size before accessing them.

## Backend ABI

`FS_AppBackend` uses a C ABI operations table. Applications call checked public
wrappers instead of dereferencing the operations table directly.

```c
#define FS_APP_BACKEND_ABI_VERSION 1u

typedef struct FS_AppBackendOps {
    uint32_t struct_size;
    uint32_t abi_version;

    FS_Result (FS_CALL *create)(FS_AppBackendHost*, const FS_AppBackendDesc*,
                        FS_BackendInstance**, FS_Error*);
    void (FS_CALL *destroy)(FS_BackendInstance*);
    FS_Result (FS_CALL *create_window)(FS_BackendInstance*, const FS_AppWindowDesc*,
                               FS_BackendWindow**, FS_Error*);
    void (FS_CALL *destroy_window)(FS_BackendInstance*, FS_BackendWindow*);
    FS_Result (FS_CALL *pump_events)(FS_BackendInstance*, FS_AppPumpMode,
                             uint64_t timeout_ns, FS_Error*);
    FS_Result (FS_CALL *acquire_frame)(FS_BackendInstance*, FS_BackendWindow*,
                               FS_AppFrame*, FS_Error*);
    FS_Result (FS_CALL *present_frame)(FS_BackendInstance*, FS_BackendWindow*,
                               FS_AppFrame*, FS_Error*);
    void (FS_CALL *cancel_frame)(FS_BackendInstance*, FS_BackendWindow*, FS_AppFrame*);
    double (FS_CALL *time_seconds)(FS_BackendInstance*);
    const FS_CapabilityHeader* (FS_CALL *query_capability)(FS_BackendInstance*,
                                                    uint32_t, uint32_t);
} FS_AppBackendOps;
```

Factories register a name, kind, ABI version, and operations table. Built-in
backends expose explicit registration functions. Static registration is
supported, but the API also lets a host register a custom backend. Dynamic
library loading is outside the first implementation; the ABI does not prevent
adding it later.

## Resource injection and ownership

The default descriptor creates the complete GPU stack, window, surface, Core,
and render context. Advanced descriptors may inject individual resources.

Each injected resource uses one of these ownership states:

```c
typedef enum FS_ResourceOwnership {
    FS_RESOURCE_NONE = 0,
    FS_RESOURCE_BORROWED,
    FS_RESOURCE_TRANSFERRED
} FS_ResourceOwnership;
```

- `NONE` means the library creates the missing resource.
- `BORROWED` means the library uses but never releases the resource.
- `TRANSFERRED` means ownership changes only after creation succeeds.
- A failed create operation leaves transferred inputs owned by the caller.
- Internally created resources are always library-owned.

Ownership is recorded per Instance, Adapter, Device, Queue, native window,
Surface, and Core. Parent ownership never implies child ownership.

Valid injection configurations include an Instance only, Instance plus
Adapter, Device plus its Queue, a complete GPU stack, a native window, an
existing Surface, or an existing compatible Core. Invalid combinations such as
a Queue without a Device, mismatched Device and Queue identities, or two window
objects managing one Surface fail before ownership transfer.

WebGPU cannot always reveal parent identities. Advanced injection descriptors
therefore carry identity information that the host contractually guarantees.
Debug builds retain identity tokens for validation.

## Transactional creation and destruction

Creation is transactional:

1. Validate descriptor sizes, enum values, and backend ABI.
2. Create generic App state, event storage, and task storage.
3. Create the platform backend.
4. Validate injected GPU resources and create missing resources.
5. Register GPU error and device-lost callbacks.
6. Create or wrap the native window.
7. Create or accept the Surface.
8. select and configure Surface capabilities.
9. Create the window's Core and RenderContext.
10. Create presentation resources for the selected output contract.
11. Publish initial window, surface, size, and scale events.
12. Commit transferred ownership and enter the ready state.

Failure rolls back completed steps in reverse order. Destruction first stops
new frames, cancels acquired frames, destroys per-window rendering and surfaces,
detaches platform callbacks, destroys native windows it owns, releases shared
GPU references, and finally destroys event and App storage.

Borrowed native windows are detached but never destroyed. A shared GPU context
outlives every window that references it.

## Complete Core and render refactor

The current Core mixes drawing, size-dependent resources, effects, filters,
presentation pipelines, and output-target assumptions. The new design splits
these concerns into `FS_Core`, `FS_RenderContext`, `FS_RenderFrame`,
`FS_RenderTarget`, a lightweight Frame Graph, and `FS_Presenter`.

### Core responsibility

`FS_Core` owns drawing semantics and scene resources:

- Canvas state, transforms, paths, text, images, clips, and recorded commands.
- Font, glyph, image, and atlas resources.
- Scene-pass encoding into an internal standard scene target.

It does not own or configure a Surface, acquire or present frames, select an
output format, or create a Surface-specific presentation pipeline.

### Render context

Each window owns an `FS_RenderContext` that references a shared GPU context and
owns one Core. It manages framebuffer-sized scene, resolve, filter,
post-processing, and readback resources. Resize is transactional: it creates a
complete replacement resource set before swapping it into use.

### Render target

All output destinations use an explicit `FS_RenderTarget` that describes its
kind, texture, view, format, extent, sample count, usage, generation, and
read/copy/present capabilities. Surface frames, offscreen textures, and
host-provided targets use the same rendering path.

### Render frame and Frame Graph

The replacement for the monolithic `fs_core_encode()` is:

```text
begin render frame
  -> encode Core scene
  -> built-in effects
  -> filter chain
  -> application custom passes
  -> color conversion and presentation pass
  -> finish command batch
  -> AppFrame submit
  -> AppFrame present
```

The lightweight Frame Graph supports ordered stages, temporary texture pooling,
declared pass reads and writes, conflict validation, debug labels, and command
encoder ownership. It does not reorder passes, schedule multiple queues, or
replicate Vulkan-style barriers.

Stages include `BEFORE_SCENE`, `SCENE`, `AFTER_SCENE`, `BEFORE_EFFECTS`,
`EFFECTS`, `AFTER_EFFECTS`, `BEFORE_PRESENT`, `PRESENT`, and `AFTER_PRESENT`.
The transition engine becomes a custom pass instead of acquiring and presenting
a Surface directly.

### Presentation and color

Presentation moves out of Effects into `fullstack_present`. The presenter owns
pipelines that convert the standard scene output to the target format and color
space. Its cache key includes Device, output format, alpha mode, source color
space, target color space, and tone-mapping policy.

This replaces the existing temporary downgrade from sRGB Surface formats to
non-sRGB formats with an explicit Linear/sRGB contract and creates an extension
path for Display P3 and HDR.

The old `fs_core_encode()` remains only as a deprecated adapter that calls the
new render path during migration.

## Explicit application frames

`FS_AppFrame` is a stack-friendly, single-use frame token containing frame ID,
Surface generation, extent, format, texture, view, and internal ownership data.
Its state machine is:

```text
EMPTY -> ACQUIRED -> SUBMITTED -> PRESENTED
                    `----------> CANCELLED
```

Only one frame may be acquired per window. Applications can submit command
buffers and present separately, which supports custom passes and external Queue
scheduling. A convenience wrapper performs acquire, render, submit, and present
for ordinary applications.

Every Surface reconfiguration increments a generation. Submit and present
reject stale frames; cancel remains legal so their resources can be released.
Double acquire, double submit, present before submit, and double present are
runtime errors in every build configuration.

Acquire results distinguish normal skip conditions from failures: zero-sized
framebuffers, timeout, occlusion, suspension, outdated Surface, Surface loss,
Device loss, and fatal errors.

## Surface lifecycle

The Surface state machine is:

```text
UNAVAILABLE -> UNCONFIGURED -> READY -> FRAME_ACQUIRED -> FRAME_SUBMITTED
      ^              ^          |                               |
      |              `---------- RECOVERY_PENDING <-------------'
      `---------------- suspend or native surface loss
```

The backend automatically reconfigures an outdated Surface, completes a
suboptimal frame before reconfiguration, avoids configuring zero-sized
framebuffers, and rebuilds a lost Surface when its native source remains
available. Android and iOS may release their Surface while retaining the GPU,
Core CPU state, and application state.

Device recreation is explicit rather than automatic. Device loss invalidates
every associated render context and emits an event. The architecture retains
CPU-side resource descriptions and supports rebuilding them against a newly
injected or newly created GPU context.

## Backend-neutral event system

Generic event and key definitions move out of the GLFW extension header. Event
categories cover quit, window state, logical and framebuffer resize, scale,
safe area, orientation, pointers, scroll, touch, keys, UTF-8 text input, IME
editing, file drop, suspend, resume, Surface availability, memory pressure,
Device loss, backend errors, and user events.

Events use stable App and Window IDs rather than object pointers. Each event has
a global sequence number and monotonic timestamp. Pointer and touch payloads
carry both logical and framebuffer coordinates. UI layout uses logical units;
Core output uses framebuffer pixels.

Physical/logical key events are separate from committed UTF-8 text and IME
pre-editing events. This is required for Chinese input, composed characters,
selection ranges, and mobile keyboards.

The event queue is multi-producer and single-consumer. Platform and worker
threads may push events, while the App thread consumes them. Move, resize,
scale, and safe-area events may coalesce; key, button, text, touch boundaries,
lifecycle, and error events never coalesce. Dynamic payloads are copied into
queue-owned storage and released with `fs_app_event_release()`.

Each window also maintains an input snapshot containing current keys, buttons,
pointer locations, touch points, modifiers, focus, one-frame pressed/released
edges, and accumulated scroll.

## Threading model

Platform event pumping, window mutation, Surface acquire, and present are bound
to the thread that created or attached the backend. GPU encoding defaults to the
same thread. The API validates this affinity.

Workers may perform CPU tasks and push application events or main-thread tasks.
`fs_app_post_task()` schedules a callback on the App thread and wakes the
platform event wait. This supports asynchronous image decoding and Monet color
extraction without introducing GLFW dependencies.

The architecture does not claim that arbitrary window or Surface calls are
thread-safe when the underlying platforms prohibit them.

## Main-loop modes

Advanced hosts use the step API:

```text
begin_frame
  -> pump_events
  -> consume events
  -> update
  -> acquire/render/submit/present each requested window
  -> end_frame
```

Desktop applications may call `fs_app_run()` with init, event, update, render,
and shutdown callbacks. The convenience loop uses the same step functions.

The event pump supports poll, wait, and timed wait. Continuous animation uses
polling; idle, occluded, or zero-sized windows wait; deadlines use timed waits.
Workers can request a wake or redraw. Closing a window first emits a request so
the application can accept or reject it.

Android and iOS use externally driven ticks. `fs_app_run()` returns an
external-loop-required error on platforms where the library cannot own the
process loop.

## Lifecycle events

The generic lifecycle includes suspend, resume, Surface available, Surface
unavailable, low memory, and quit requested. Resume does not imply that a
Surface is available. Suspend stops frame acquisition but does not implicitly
destroy the Device or Core. Low-memory events let applications trim caches;
backends do not silently destroy application resources.

Desktop minimization is represented by visibility and framebuffer state rather
than application suspension.

## Optional platform services

File dialogs, clipboard, URI opening, soft keyboards, cursors, window controls,
and power services are versioned capability vtables returned by
`query_capability()`. Adding a service does not change the primary backend ABI.

File dialogs are asynchronous. GLFW uses vendored tinyfiledialogs on an allowed
worker thread and returns completion through the event or task queue. Platforms
that require UI-thread dialogs use their native asynchronous APIs. Android uses
an Activity Result bridge, and iOS uses `UIDocumentPickerViewController`.

Unsupported capabilities return `NULL` or `FS_RESULT_UNSUPPORTED`; they do not
provide partial dummy behavior.

## Backend implementations

The GLFW backend is reorganized into window, event, Surface, service, and
factory modules. It no longer creates WebGPU devices, owns Core, creates Effects
presentation pipelines, or calls Core rendering functions.

SDL3 implements the same operations independently. Shared behavior such as
Window IDs, event sequencing, input snapshots, frame validation, Surface
generation, diagnostics, and loop policy resides in generic App modules.

Android supports `ANativeWindow`, lifecycle callbacks, safe areas, orientation,
IME, and Activity Result services. iOS uses an Objective-C++ adapter for UIKit,
Scene lifecycle, `CAMetalLayer`, safe areas, keyboard state, and document
pickers. Both preserve the public C ABI.

An explicit native-handle query is the only escape hatch for advanced hosts.
Core, UI, and ordinary examples cannot use it.

## UI and transition migration

The current `examples/ui/app.hpp` is separated into application, window,
renderer, event dispatcher, layout engine, and transition renderer units.
Only the application and renderer coordinate generic App and Render APIs.

UI code no longer stores `FS_GlfwBackend`, `FS_GlfwBackendExt`, or
`GLFWwindow*`. It receives generic events, logical dimensions, framebuffer
dimensions, scale, and time. The layout bridge remains platform-neutral.

Transitions render pages into offscreen `FS_RenderTarget` objects and register
their final composition in the Frame Graph. They no longer access Device,
Queue, Surface, Surface format, acquire, or present through GLFW fields.

All regular examples migrate to the new App API. As previously agreed,
`wgpu_minimal_triangle.c` remains a low-level example outside this abstraction.

## Compatibility strategy

Legacy GLFW symbols move to an optional legacy target. Their implementation is
a facade over `FS_App`, the GLFW backend, `FS_AppWindow`, and
`FS_RenderContext`; it never retains a second Surface, event queue, or recovery
path. New code cannot include the legacy header.

Legacy functions emit deprecation warnings for one migration cycle and are
scheduled for removal in the next major API version after all maintained
examples migrate.

## Error and diagnostics model

New APIs return `FS_Result`, which distinguishes success, skip, pending,
cancelled, unsupported operations, argument/state/thread errors, allocation
failures, backend and ABI errors, window and Surface errors, Device loss, stale
frames, GPU validation errors, and internal errors.

Optional caller-owned `FS_Error` values contain a result code, domain, native
code, operation, fixed-size message, sequence, and debug source information.
The library does not print directly to stderr. A diagnostic callback routes
logs to examples, logcat, os_log, or a host engine.

Debug validation checks ABI sizes, thread affinity, resource identities, frame
state transitions, Surface generations, callback re-entry, dynamic event
payload lifetime, Frame Graph conflicts, and leaked acquired frames. Release
builds retain all safety-critical state checks.

## Performance requirements

Steady-state frames must not allocate heap memory in begin/end frame, ordinary
event polling, input updates, acquire, frame-token handling, common Frame Graph
passes, submit, or present. Frame arenas, small vectors, pipeline caches,
temporary texture pools, and preallocated event storage provide this behavior.

Allocation is allowed during creation, resize, Surface format changes, first
pipeline creation, long dynamic event payloads, queue growth, and platform
service requests.

The backend does not synchronously wait for all GPU work during normal present.
Retired resources use submission completion tracking or a bounded compatibility
delay when the current wgpu-native API cannot expose the required callback.

## Build organization

The build exposes these targets:

```text
fullstack_result
fullstack_gpu
fullstack_core
fullstack_render
fullstack_present
fullstack_app
fullstack_backend_mock
fullstack_backend_glfw
fullstack_backend_sdl3
fullstack_backend_android
fullstack_backend_ios
fullstack_platform_dialogs
fullstack_legacy
```

Core and offscreen rendering build without a window backend. CMake options
control each backend, legacy support, tests, and examples. Target dependencies
enforce that Core cannot depend on App or a platform backend, backends cannot
depend on Core drawing APIs, and Effects cannot own presentation.

## Verification strategy

A Mock backend drives deterministic tests for window creation, event ordering,
thread affinity, wake behavior, lifecycle, Surface status injection, frame
state transitions, capability versions, and rollback.

Ownership tests cover created, borrowed, and transferred resources; creation
failure before transfer; multiple shared windows; and exactly-once release.

Frame tests cover the valid path, cancellation, every invalid transition,
stale generations, zero-sized windows, timeout, occlusion, Surface loss,
suboptimal reconfiguration, and multi-window isolation.

Event tests cover monotonic sequence and timestamps, queue pressure, coalescing,
critical event preservation, UTF-8 and Chinese IME, dynamic payloads, worker
producers, input edges, and logical/framebuffer coordinate conversion.

Render tests cover backend-free offscreen operation, independent window cores,
multiple Surface formats, Linear/sRGB screenshot baselines, transactional
resize, Frame Graph ordering, transitions, effects, filters, custom passes, and
explicit Device-lost rebuilding.

Platform smoke tests cover GLFW and SDL3 on Windows, Linux, and macOS, plus
Android and iOS creation, first frame, resize or rotation, suspension, Surface
recreation, input, IME, file-dialog cancellation, sustained rendering, and
shutdown.

Architecture audits reject GLFW or SDL symbols in UI and generic headers,
platform symbols in Core, Surface present calls in Core, Core drawing calls in
backends, presentation pipeline creation in Effects, and legacy includes in new
examples.

## Implementation phases

The phases control integration risk but do not reduce the final scope.

1. Add result, diagnostics, GPU context, event, input, task, registry, and Mock
   backend foundations.
2. Fully split Core, RenderContext, RenderTarget, Frame Graph, and Presenter.
3. Rewrite GLFW against the new backend ABI and migrate platform services.
4. Split UI application responsibilities and migrate transitions and examples.
5. Implement and contract-test the SDL3 backend.
6. Implement Android and iOS lifecycle and native service adapters.
7. Make legacy support optional and prepare its major-version removal.

## Completion criteria

The work is complete only when:

- Core, GPU, Render, Present, App, and backend responsibilities are separated.
- GLFW and SDL3 are peer implementations of one backend ABI.
- Android and iOS use real lifecycle and Surface adapters.
- UI and transitions contain no direct concrete-backend dependencies.
- Multiple windows share GPU resources but own independent Core instances.
- Default ownership and injected ownership pass exact-lifetime tests.
- Surface and Device failure behavior follows the specified state machines.
- Events, IME, scale, safe areas, and asynchronous services are backend-neutral.
- The legacy layer contains no independent implementation.
- Core and offscreen rendering build without a window backend.
- Contract, ownership, frame, event, render, architecture, and platform smoke
  tests pass.

Passing one GLFW demo or preserving the old implementation behind a generic
name does not satisfy these criteria.


## Normative ABI and lifecycle clarifications

This section is normative and resolves implementation details described at a
higher level above.

### Generic and backend object boundaries

`FS_AppWindow` is allocated by the generic App layer. It stores the stable ID,
metrics, input state, Surface state, RenderContext, Core, and one opaque
`FS_BackendWindow*`. A backend allocates and interprets only its backend window
and instance storage. It never allocates the generic App window. A factory and
its operations table remain valid until every instance is destroyed;
unregistering a live factory returns `FS_RESULT_INVALID_STATE`.

### Stable C ABI rules

Exports use `FS_API` visibility and the `FS_CALL` calling convention. Exported
structures use fixed-width values; public source enums are stored as `uint32_t`
fields rather than compiler-sized enums. Required vtable slots are non-null.
Optional slots are gated by `struct_size`, minor version, and capability flags.
Registration requires an exact ABI major and a supported minor range.

Capability queries return an immutable backend-owned `FS_CapabilityHeader`
containing size, capability ID, major/minor version, and flags. It remains valid
until backend destruction, and its operations declare thread affinity.

### Dependency-closed injection

Injected GPU resources form dependency-closed bundles. Device plus Queue is a
valid offscreen bundle, but it cannot create a new Surface-backed window unless
the host also supplies a compatible Instance, Adapter, and lineage token. An
injected Surface includes GPU lineage and native-source identity. Missing
parents are created only when their ancestry can be established.

Borrowed resources outlive the final App, window, callback, queued task,
submitted command, and deferred retirement that references them. Successful
transfer grants exclusive ownership. An injected Surface or Core has one
exclusive managing window; a borrowed Core cannot be rendered concurrently by
another owner.

### Submission and frame contract

Queue submission is a generic App/GPU operation, not a backend-vtable slot.
`fs_app_frame_submit()` accepts command buffers, submits them to the window GPU
Queue, and returns a monotonic submission ID. Command-buffer handles remain
caller-owned and may be released after Queue submission returns. External
schedulers call `fs_app_frame_mark_submitted()` with validated Queue identity
and submission token before present.

`FS_AppFrame` is initialized with `FS_APP_FRAME_INIT` or reset explicitly. Its
stable public layout contains value fields and a fixed-size opaque token, not
private pointers. Acquired texture and view references are frame-owned and are
released exactly once by present or cancel. Cancel is legal from `ACQUIRED`.
From `SUBMITTED`, it abandons presentation when supported or marks the frame
non-presented and defers release until submission completion. Presented and
cancelled frames can only be reset.

### Reconfiguration with outstanding frames

Surface reconfiguration never occurs while a Surface texture remains acquired.
Resize, scale, mode change, suspend, Surface loss, or native-source replacement
records a pending transition while a frame is acquired or submitted.

- An acquired frame must finish or cancel.
- A submitted frame retires through present, legal cancellation, or submission
  completion before reconfiguration.
- Suspend and native loss reject new acquire immediately but use the same frame
  retirement rule.
- Device loss releases references at the WebGPU-required boundary without
  attempting presentation.

Only after retirement can the backend reconfigure the Surface and increment its
generation.

### Thread roles and loop ordering

The App records App, backend-affinity, event-consumer, and platform-UI threads.
On GLFW and ordinary desktop SDL3 they coincide with the creation thread. A host
separates them only when the backend advertises that capability. The event queue
remains single-consumer.

Android and iOS callbacks update a backend lifecycle mailbox on their required
platform thread and wake the App thread. The App thread serializes mailbox
changes into unified lifecycle events. Platform callbacks never invoke update
or render directly.

The legal step order is `begin_frame`, event pump, event consumption, update,
acquire/encode/submit/present, and `end_frame`. Begin-frame drains prior tasks
and resets input edges. End-frame validates frames, drains destruction-safe
work, and commits pending Surface changes. Callback re-entry into pump,
destruction, or another tick fails. Destruction requested inside a callback is
deferred. Wake is thread-safe, idempotent until observed, and guarantees a
waiting pump returns.

### Asynchronous destruction safety

Destroy marks objects closing, rejects requests, detaches platform callbacks,
cancels services, and invalidates weak IDs. Device callbacks, worker events,
tasks, and service completions use reference-counted callback contexts.
Destruction completes only after they drain or acknowledge cancellation. Late
completion receives cancelled status and cannot access destroyed storage.
Borrowed hosts release resources only after destroy or asynchronous detach
completes.

### Presenter and Frame Graph semantics

Each RenderContext owns one `FS_Presenter`. Immutable pipelines may use a shared
GPU cache, but target bindings and output contract are per-window. Pending
frames retire before Presenter, RenderContext, Core, Surface, and the final GPU
reference are destroyed in that order.

The final GPU stage is `AFTER_PRESENT_ENCODE`, meaning after presentation-pass
encoding and before submission. Work after `wgpuSurfacePresent()` uses a CPU
post-present callback, not a Frame Graph render pass.

### Platform-service requests

Every asynchronous service has a versioned request descriptor and result,
stable request ID, cancellation operation, completion callback or event, and a
result-release function. Requests retain weak IDs rather than raw pointers.
Window destruction cancels window-bound requests; App destruction cancels and
drains all requests. Late native completion becomes cancellation.

The service selects its legal execution context. GLFW uses a worker only when
the dialog library and host platform permit it. Main-thread-only platforms
schedule native UI on the platform thread and report completion asynchronously.

### Event-queue overflow

The queue has a soft capacity and hard byte limit. It coalesces eligible events
before bounded growth. If a critical event cannot be stored, the producer
returns queue-pressure or out-of-memory, releases unqueued payload, writes a
fixed emergency diagnostic, and wakes the consumer. Workers may use configured
bounded backpressure but never while holding a platform lock.

Already queued payloads remain queue-owned until poll/drain and explicit event
release. Losing a key, button, text, touch-boundary, lifecycle, or error event is
a fatal input-state desynchronization; the App rebuilds the snapshot from
backend state when supported. Input edges and scroll reset once in
`fs_app_begin_frame()`, before pumping, and remain valid until the next call.

### Measurable legacy removal gate

`FS_BUILD_LEGACY` defaults to `ON` for the first release with the new API and to
`OFF` in the following minor release. CI prevents all files except the legacy
implementation, legacy tests, and one migration sample from including
`fullstack_legacy.h`.

Legacy symbols may be removed in the next major release only after two
consecutive releases where every maintained target uses the new API, audits find
no non-legacy use, GLFW and SDL3 contract suites pass, the migration guide maps
every legacy symbol, and the legacy target contains facades only.
## Final normative contract corrections

If an earlier example conflicts with this section, this section controls.

### Backend instance allocation

The backend vtable uses opaque backend objects, not generic App objects:

```c
FS_Result (FS_CALL *create)(FS_AppBackendHost* host,
                    const FS_AppBackendDesc* desc,
                    FS_BackendInstance** out_instance,
                    FS_Error* error);
FS_Result (FS_CALL *create_window)(FS_BackendInstance* instance,
                           const FS_AppWindowDesc* desc,
                           FS_BackendWindow** out_window,
                           FS_Error* error);
```

`FS_AppBackendHost` is a read-only generic-services table that supplies the
allocator, diagnostics, event enqueue, wake, monotonic clock fallback, and task
posting functions. A custom backend allocates its private instance through this
allocator and returns it as `FS_BackendInstance*`. Generic code never inspects
that storage. Destroy operations receive the same opaque instance and window.

### Fixed-width callable ABI types

Every enum-like callable ABI type is a fixed-width typedef, including function
returns and parameters:

```c
typedef uint32_t FS_Result;
typedef uint32_t FS_AppPumpMode;
typedef uint32_t FS_AppEventType;
typedef uint32_t FS_AppFrameState;
typedef uint32_t FS_ResourceOwnership;
```

Named constants are macros or anonymous-enum constants representable by the
underlying typedef. No exported function returns or accepts a compiler-defined
enum type. IDs, flags, sizes, timestamps, and versions also use explicit-width
types.

### Submission token ownership

All Queue submissions, including host-scheduled work intended for an App frame,
flow through `FS_GpuContext`:

```c
FS_Result fs_gpu_submit(FS_GpuContext* gpu,
                        uint32_t command_count,
                        const WGPUCommandBuffer* commands,
                        FS_SubmissionToken* out_token,
                        FS_Error* error);
FS_Result fs_gpu_submission_status(FS_GpuContext* gpu,
                                   FS_SubmissionToken token,
                                   FS_SubmissionStatus* out_status);
```

`FS_SubmissionToken` contains a GPU-context identity and monotonic serial. The
GPU context owns completion tracking. `fs_app_frame_mark_submitted()` accepts
only a token created by the frame's GPU context. A host that bypasses
`fs_gpu_submit()` cannot mark an App frame submitted and must instead use an
external render target that is not managed as an `FS_AppFrame`.

The concrete step API is:

```c
FS_Result fs_app_begin_frame(FS_App*, FS_Error*);
FS_Result fs_app_pump_events(FS_App*, FS_AppPumpMode,
                             uint64_t timeout_ns, FS_Error*);
FS_Result fs_app_poll_event(FS_App*, FS_AppEvent* out_event);
FS_Result fs_app_end_frame(FS_App*, FS_Error*);
```

`begin_frame` is legal only from `READY`; it enters `FRAME_ACTIVE`.
Pump and poll are legal only during `FRAME_ACTIVE`. Acquire/render operations
are legal after pump and before end. `end_frame` requires no unresolved
`ACQUIRED` frame, commits deferred work, and returns to `READY`. `SKIP` from
poll means the queue is empty; it is not an error.

### Nonblocking destruction and tombstones

Public destruction is two-phase and nonblocking:

```c
FS_Result fs_app_begin_destroy(FS_App*, FS_DestroyToken* out_token);
FS_Result fs_app_destroy_status(FS_DestroyToken, FS_DestroyStatus* out_status);
void fs_app_destroy_token_release(FS_DestroyToken);
```

Begin-destroy immediately detaches the live App API, invalidates weak IDs,
cancels requests, and transfers callback contexts to a reference-counted
`tombstone`. The caller can release the App allocation after begin-destroy
returns. Late native callbacks observe only the tombstone and can acknowledge
cancellation without touching App memory. The tombstone self-destructs after
all references retire. A synchronous convenience destroy pumps only legal local
work up to a caller-supplied timeout, then falls back to the tombstone path; it
never waits indefinitely or blocks a platform UI thread on itself.

### Guaranteed critical-event failure semantics

Critical events are never silently dropped. The queue reserves fixed emergency
slots and inline payload storage for a synthetic
`FS_APP_EVENT_FATAL_QUEUE_PRESSURE`. When normal storage and bounded growth fail,
the producer atomically transitions the App to `FATAL_EVENT_QUEUE`, stores the
first failure metadata in the reserved slot, wakes the consumer, and rejects
further non-destruction work. The App does not continue with reconstructed
input because committed text and lifecycle transitions are not recoverable.

Contract tests therefore verify either delivery of every critical event or a
guaranteed fatal queue-pressure event followed by deterministic shutdown. No
test assumes continuation after critical-event loss.

### Service completion affinity

Platform-service native work may execute on a required UI or worker thread, but
user-visible completion is always posted to the App task queue and delivered on
the App/event-consumer thread. Completion events and completion callbacks are
serialized with normal App dispatch. A request chooses exactly one delivery
form. Completion callbacks cannot pump events, enter another App tick, or
perform immediate destruction; they may post tasks or request deferred destroy.
The runtime validates these re-entry rules.
### Final frame, allocation, and calling-convention invariants

`fs_app_end_frame()` requires every frame acquired during the tick to be in
`PRESENTED` or `CANCELLED`. `SUBMITTED` is not an acceptable end-frame state.
Cross-tick presentation is not supported by this ABI version. An application
that needs host-controlled delayed presentation must render to an external
`FS_RenderTarget` and import its result into a later App frame.

`FS_App` is an opaque library allocation returned by `fs_app_create()`. The
nonblocking destruction sequence is:

```c
FS_Result fs_app_begin_destroy(FS_App* app,
                               FS_DestroyToken* out_token,
                               FS_Error* error);
void fs_app_release(FS_App* app);
```

Begin-destroy invalidates the live API and transfers late callback ownership to
the tombstone but does not free the opaque App allocation. The caller invokes
`fs_app_release()` exactly once after begin-destroy succeeds. Release frees the
App allocation immediately and does not wait for the tombstone. Apps embedded
in caller-owned storage are not supported; a future placement API would require
an independent contract.

`FS_CALL` applies to every function that crosses a module or plugin ABI
boundary: exported functions, every backend-vtable slot, every
`FS_AppBackendHost` slot, capability/service operations, diagnostics, event and
task callbacks, lifecycle callbacks, render-pass callbacks, and completion
callbacks. Public headers express these through function-pointer typedefs or an
explicit `FS_CALL` on each slot. A backend built with a mismatched calling
convention fails ABI registration where detectable and is unsupported
otherwise.