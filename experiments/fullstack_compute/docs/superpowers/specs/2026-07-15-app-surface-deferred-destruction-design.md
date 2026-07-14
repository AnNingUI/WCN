# App Surface deferred destruction design

This specification prevents device loss when an App repeatedly creates,
renders, and destroys native windows. A window becomes logically unavailable
as soon as you destroy it, but App keeps its backend Surface and native window
alive until the last GPU submission that used that Surface has completed.

The change belongs entirely to the optional `app/` extension, plus tests and
the Window Arena stress harness. Root `include/` and `src/` remain unchanged.

## Problem statement

Window Arena can create and destroy many Rift windows during one run. The
reported failure occurred after window ID 40 with this wgpu-native validation
error:

```text
Error in wgpuSurfaceGetCurrentTexture: Validation Error
Caused by:
  Parent device is lost
```

The game permits at most eight world windows at once, so the failure does not
come from 40 simultaneous windows. It comes from repeated window and Surface
turnover.

The current frame path records a submission token in `FS_AppFrame`, submits
render work, presents the Surface texture, and releases the acquired texture
handles. A successful present does not preserve that token on `FS_AppWindow`.
When `fs_app_destroy_window()` runs later, its Surface retirement wait often
has no token to inspect. The backend can then unconfigure and release the
Surface, followed by destruction of the GLFW or SDL3 native window, while GPU
work from the final present can still reference that Surface.

This ordering creates a driver-dependent race. Fast smoke tests can pass while
a long session eventually loses the shared WebGPU device.

## Goals

The implementation must provide these behaviors:

- Destroying a window removes it immediately from the logical App window set.
- A supported backend hides the window immediately.
- App does not release a Surface or native window until its last Surface
  submission completes.
- Normal destruction does not block unrelated windows or wait for the entire
  device to become idle.
- App shutdown drains pending destruction before it releases the backend and
  GPU context.
- GLFW and SDL3 follow the same backend-neutral lifetime rule.
- The fix does not reduce enemy spawn rates, simultaneous Rift capacity, or
  gameplay complexity.

## Non-goals

This change does not add device recreation after a genuine device loss. It
also does not pool game windows, share mutable RenderContexts between windows,
or change root GPU retirement APIs. Those options do not correct the Surface
ordering defect and require separate designs.

## Chosen approach

App uses two-phase window destruction:

1. Logical destruction happens synchronously in
   `fs_app_destroy_window()`.
2. Physical backend destruction happens asynchronously after the last Surface
   submission reaches a terminal state.

This approach avoids the frame hitch of a blocking `wgpuDevicePoll()` on every
Rift close. It also fixes the lifecycle contract for every App consumer rather
than hiding the problem inside Window Arena.

## Window lifetime state

`FS_AppWindow` gains private state for Surface use and deferred destruction:

```c
FS_SubmissionToken last_surface_submission;
FS_AppWindow* pending_destroy_next;
bool deferred_destruction_safe;
bool pending_destroy;
```

`FS_App` gains an intrusive pending-destruction list and a count:

```c
FS_AppWindow* pending_destroy_head;
uint32_t pending_destroy_count;
```

The intrusive list avoids an allocation during destruction. This matters
because cleanup must remain reliable under memory pressure.

`deferred_destruction_safe` is true for windows created by the backend and for
injected native windows whose ownership is transferred to App. It is false for
an injected `FS_RESOURCE_BORROWED` native window. App stores this decision when
it creates the `FS_AppWindow`.

`last_surface_submission` is distinct from the existing
`retiring_submission`. The existing token temporarily blocks Surface recovery
after a submitted frame is cancelled. The new token records the most recent
submission that can reference a successfully presented Surface and must not
change normal renderability or Surface state.

## Submission tracking

After `fs_app_frame_present()` successfully presents a submitted frame, App
copies `frame->submission` into `window->last_surface_submission` before it
completes the frame ticket.

If presentation fails after submission, App retains the existing cancellation
and recovery behavior. Destruction uses the newest valid token from
`last_surface_submission` and `retiring_submission`. Submission serials are
monotonic within one `FS_GpuContext`, so the greater serial represents the
later dependency.

Acquired texture and texture-view handles keep their current release behavior
after a successful present. The deferred window lifetime protects the Surface
and native platform object that own the presentation path.

## Logical destruction

`fs_app_destroy_window()` performs these operations on the App thread:

1. Validate that the window is active and belongs to the App.
2. Remove its redraw requests and scheduler state.
3. Cancel an active frame, if one exists.
4. Discard queued messages for the window.
5. Request `visible = false` through the optional window-control capability.
6. Quiesce backend event routing, callbacks, and asynchronous services for the
   window.
7. Remove the window from `app->windows`, so count, lookup, event delivery, and
   public iteration no longer expose it.
8. Select the final Surface submission token.
9. Finalize the backend window immediately if no GPU dependency remains.
10. If asynchronous destruction is safe, link the window into
    `pending_destroy_head` and return success.
11. If the native window is borrowed, drain its final Surface submission and
    release the backend Surface synchronously before returning.

Once the function returns, callers must treat the `FS_AppWindow*` as invalid,
matching the existing destruction contract. App retains the allocation only
as an internal tombstone until physical destruction completes.

Window Arena already fades a Rift to zero opacity before destruction. The App
visibility request makes immediate disappearance the generic behavior for
other consumers. If a backend does not provide visibility control, physical
destruction normally follows within the next few GPU polls.

The synchronous borrowed-window branch preserves the external ownership
contract. Once `fs_app_destroy_window()` returns, the external owner can safely
destroy its native handle because App no longer retains a Surface or backend
wrapper that references it. Window Arena uses backend-created windows, so its
normal path remains asynchronous.

## Backend quiescing

The backend-neutral ABI adds a required `quiesce_window` operation. Quiescing
does not release the Surface or native window. It performs only operations that
must happen at logical destruction:

- Cancel pending dialogs, file operations, and other window-scoped services.
- Disable native callbacks or mark the backend window as quiesced.
- Make backend event lookup ignore the window.
- Stop new exposure, input, resize, and close events from entering App.

GLFW and SDL3 keep the backend wrapper in their internal list until physical
destruction, but their lookup functions skip quiesced entries. Their existing
`destroy_window` operations remain safe after quiescing and must not repeat a
non-idempotent service completion.

Events that entered the backend-neutral `FS_EventQueue` before quiescing can
still remain queued. `fs_app_poll_event()` therefore discards a window-scoped
event when `fs_app_find_window()` no longer finds its ID, then continues polling
until it finds a live-window event, a global event, or an empty queue. Global
events with window ID zero remain valid.

## Physical destruction

App adds a private collector that runs immediately after
`fs_gpu_context_poll()` at the start of an App frame. For each pending window,
the collector queries its selected submission token:

- `FS_SUBMISSION_PENDING` keeps the window in the list.
- `FS_SUBMISSION_SUCCEEDED` permits backend destruction.
- `FS_SUBMISSION_FAILED` also permits backend destruction because no useful
  work remains to protect.
- A missing GPU context or a terminal GPU state permits best-effort backend
  destruction.
- An unexpected unknown token emits a diagnostic and keeps the window until
  shutdown, where the drain path resolves it conservatively.

Finalization calls the backend `destroy_window` operation, clears the backend
handle, and deallocates the private `FS_AppWindow` tombstone. Backend code
continues to own the required order:

```text
cancel backend services
unconfigure Surface
release Surface
destroy GLFW or SDL3 native window
free backend window state
```

The collector never runs from a WebGPU callback. It runs on the App thread, so
GLFW, SDL3, scheduler, and backend window-list operations keep their thread
ownership guarantees.

The public App diagnostics API adds this read-only query:

```c
uint32_t fs_app_pending_destroy_count(const FS_App* app);
```

The query lets runtime stress tests, diagnostics, and orderly shutdown checks
observe physical lifetime without exposing backend handles or mutable state.

## Shutdown behavior

`fs_app_begin_destroy()` and `fs_app_release()` must run on the App thread.
Shutdown uses this exact sequence:

1. Enter the closing state and logically destroy every active window.
2. If the GPU is ready and pending windows remain, call
   `wgpuDevicePoll(device, true, NULL)` once.
3. Call `fs_gpu_context_poll()` to process completion callbacks and collect
   root GPU retirements.
4. Collect every pending App window whose token is now terminal.
5. If a token remains unknown after the global drain, emit a diagnostic and
   force best-effort backend finalization.
6. Destroy the backend instance, then release the remaining App resources.

The blocking drain is restricted to full App shutdown. Normal Rift closure
remains asynchronous and does not stall rendering in the main window or other
Rifts.

If the device is already lost, shutdown skips waiting for successful work and
performs best-effort backend cleanup. It must not call Surface acquisition or
presentation after the GPU enters a lost state.

## RenderContext ordering

Window Arena keeps its current high-level destruction order:

```text
destroy Rift RenderContext logically
remove Rift from the game world
destroy App window logically
retire RenderContext GPU resources after submission completion
destroy Surface and native window after submission completion
```

`fs_render_context_destroy()` already schedules its owned Core and Presenter
resources against the RenderContext's last submission. The App change closes
the missing half of the contract by protecting the Surface and native window
against the same in-flight work.

## Error handling

The implementation follows these rules:

- Logical destruction succeeds once App removes the window from its active
  set, even when physical destruction is deferred.
- Failure to hide a window does not cancel destruction.
- No allocation is required to enqueue pending destruction.
- A failed submission is treated as terminal for resource cleanup.
- Device loss stops further acquisition and permits best-effort teardown.
- Shutdown reports diagnostics for unresolved tokens but does not leak backend
  windows intentionally.
- Borrowed native windows synchronously release their Surface dependency before
  `fs_app_destroy_window()` returns.
- A destroyed `FS_AppWindow*` is invalid immediately and must never be passed
  to another public API call.

## Verification

Verification covers both deterministic lifetime semantics and real backend
stress.

### App lifetime tests

Tests must prove these cases:

- A successful present records the last Surface submission.
- Destroying a window removes it immediately from `fs_app_window_count()` and
  `fs_app_find_window()`.
- A pending submission delays the backend `destroy_window` call.
- A completed or failed submission releases the pending backend window on the
  App thread.
- Destroying a window without a submission remains immediate.
- Active-frame cancellation selects the correct latest dependency token.
- App shutdown drains and frees every pending window before backend teardown.
- Backend quiescing stops new events and services before physical destruction.
- Queued events for logically destroyed window IDs are discarded.
- Backend instrumentation proves that each Surface and native wrapper is
  finalized exactly once across completion, failure, and shutdown paths.
- A borrowed native window uses synchronous Surface release and never enters
  the pending-destruction list.

### Runtime Surface churn test

Window Arena gains a noninteractive stress mode that repeatedly creates,
renders, presents, and destroys at least 256 Rift windows while preserving the
normal simultaneous-window ceiling. The test records logical active windows,
`fs_app_pending_destroy_count()`, successful presents, and GPU state.

The GLFW and SDL3 test runs must meet these conditions:

- The process exits normally.
- `FS_GPU_STATE_READY` remains active throughout the churn loop.
- No Surface acquisition validation error occurs.
- All 256 or more Rift lifetimes complete.
- Active logical windows never exceed the configured game ceiling.
- Pending physical windows return to zero before shutdown completes.

The full repository test suite, architecture audit, and portable package smoke
tests run after the churn test passes. The package is regenerated only after
both backends pass.

## Alternatives rejected

A blocking device drain before every backend window destruction is simple but
can pause all game windows whenever one Rift closes. It remains a fallback for
allocation or shutdown failure, not the normal path.

A Window Arena window pool reduces native and GPU churn but leaves the App API
unsafe for other demos. It can also retain hidden native windows and large
RenderContexts for the full session. Pooling is an optional later optimization,
not a correctness fix.

Increasing GPU retirement capacities or reducing Rift creation frequency only
changes when the failure appears. Neither option establishes correct Surface
ownership.

## Next steps

After approval, create an implementation plan that separates App lifetime
changes, deterministic tests, runtime churn instrumentation, dual-backend
verification, and portable package regeneration.
