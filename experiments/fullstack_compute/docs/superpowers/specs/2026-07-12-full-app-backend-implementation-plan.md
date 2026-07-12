# Full application backend implementation plan

## Purpose

This plan implements the approved
`2026-07-12-full-app-backend-architecture-design.md` as a complete replacement
architecture. It does not stop after wrapping GLFW or making one demo compile.
Each phase has an explicit verification gate, and later phases depend on the
earlier architectural boundaries being enforced.

## Working rules

- Preserve unrelated user changes in the current dirty worktree.
- Add new APIs beside legacy APIs until maintained callers migrate.
- Keep each change buildable at its phase gate.
- Add tests before migrating higher layers onto a new contract.
- Do not place platform symbols in Core, Render, Present, or generic App code.
- Do not add `wgpu_minimal_triangle.c` to the App abstraction.
- Treat Android and iOS adapters as required deliverables, not future notes.
- Commit each numbered phase separately after its verification gate passes.

## Phase 0: Baseline and dependency graph

### Tasks

1. Capture the current build and test baseline.
2. Record all direct users of `FS_GlfwBackend`, `FS_GlfwBackendExt`,
   `fs_core_encode()`, presentation-pipeline creation, and raw Surface access.
3. Add a dependency-audit script under `tools/architecture/`.
4. Add a CTest entry that runs the audit without requiring a display.
5. Document which existing Taffy warnings are baseline warnings.

### Files

- Modify `CMakeLists.txt`.
- Modify `examples/CMakeLists.txt` only to register the audit initially.
- Add `tools/architecture/check_layers.ps1`.
- Add `tests/CMakeLists.txt`.
- Add `tests/architecture_layer_test.cmake`.

### Gate

- `cmake -S . -B build` succeeds.
- `cmake --build build` matches the existing baseline.
- The audit reports existing violations as an explicit baseline list and fails
  on new violations.

## Phase 1: Result, diagnostics, ABI, and allocator foundation

### Tasks

1. Add fixed-width `FS_Result`, error domains, `FS_Error`, diagnostic records,
   log levels, `FS_API`, and `FS_CALL` definitions.
2. Add a versioned allocator interface used by every opaque public object.
3. Add ABI helpers for `struct_size`, major/minor negotiation, required vtable
   prefixes, and capability headers.
4. Remove direct stderr logging from new modules and route it through
   diagnostics.
5. Add compile-time layout assertions for exported ABI structures.

### Files

- Add `include/fullstack_result.h`.
- Add `include/fullstack_abi.h`.
- Add `src/fullstack_result.c`.
- Add `src/fullstack_abi.c`.
- Add `tests/result_abi_tests.c`.
- Modify root CMake targets.

### Gate

- C and C++ consumers compile the same headers.
- Windows, GCC/Clang, and MSVC calling-convention compile tests pass.
- ABI tests validate short descriptors, unknown tail fields, incompatible major
  versions, optional minor fields, and null required slots.

## Phase 2: GPU context and submission tracking

### Tasks

1. Extract Instance, Adapter, Device, and Queue creation from
   `impl/fullstack_glfw_backend.c` into `FS_GpuContext`.
2. Implement created, borrowed, and transferred ownership per resource.
3. Implement dependency-closed injection and lineage tokens.
4. Add capability snapshots, device-lost state, and uncaptured-error routing.
5. Add `fs_gpu_submit()` and `FS_SubmissionToken` completion tracking.
6. Add deferred resource retirement keyed by submission serial.
7. Implement explicit context recreation against a replacement Device/Queue.

### Files

- Add `include/fullstack_gpu.h`.
- Add `src/fullstack_gpu.c`.
- Add `src/fullstack_gpu_private.h`.
- Add `tests/gpu_context_tests.c`.
- Add `tests/gpu_ownership_tests.c`.
- Modify shared WGPU discovery in CMake.

### Gate

- Offscreen GPU context creation works without GLFW.
- Every ownership combination releases each handle exactly once.
- Failed transfer leaves ownership with the caller.
- Device-lost and submission completion tests pass.

## Phase 3: Event queue, input state, tasks, and tombstones

### Tasks

1. Move normalized event/key/pointer/viewport definitions out of the GLFW ext
   header into generic App headers.
2. Implement the multi-producer, single-consumer event queue.
3. Implement coalescing, dynamic payload ownership, reserved fatal pressure
   storage, and deterministic fatal shutdown.
4. Implement per-window input snapshots and begin-frame edge resets.
5. Implement the main-thread task queue and backend wake contract.
6. Implement weak IDs, callback contexts, two-phase destruction, destroy tokens,
   and reference-counted tombstones.

### Files

- Add `include/fullstack_app_event.h`.
- Add `src/app/fullstack_app_event.c`.
- Add `src/app/fullstack_app_input.c`.
- Add `src/app/fullstack_app_task_queue.c`.
- Add `src/app/fullstack_app_lifetime.c`.
- Add `tests/event_queue_tests.c`.
- Add `tests/input_state_tests.c`.
- Add `tests/app_lifetime_tests.c`.

### Gate

- UTF-8 and Chinese IME payload tests pass.
- Queue-pressure tests deliver all critical events or the guaranteed fatal
  pressure event.
- Worker push/wake tests pass under stress.
- Late callbacks cannot access released App/window memory.

## Phase 4: Backend registry and Mock backend

### Tasks

1. Add `FS_AppBackendHost`, `FS_BackendInstance`, `FS_BackendWindow`, factory
   registration, version negotiation, and capability queries.
2. Implement the generic `FS_App` and generic `FS_AppWindow` object boundary.
3. Implement App/backend/UI thread role tracking and affinity validation.
4. Implement the concrete step-loop state machine.
5. Implement a deterministic Mock backend with synthetic windows, lifecycle,
   events, Surface statuses, and services.
6. Write the reusable backend contract test suite.

### Files

- Add `include/fullstack_app_backend.h`.
- Add `include/fullstack_app.h`.
- Add `src/app/fullstack_app_backend_registry.c`.
- Add `src/app/fullstack_app.c`.
- Add `src/app/fullstack_app_window.c`.
- Add `src/app/fullstack_app_loop.c`.
- Add `impl/backends/mock/fullstack_mock_app_backend.c`.
- Add `tests/backend_contract_tests.c`.
- Add `tests/mock_backend_tests.c`.

### Gate

- Generic App tests run without GLFW or a display.
- Factory lifetime, unregister, thread affinity, loop ordering, wake, deferred
  destroy, and multi-window tests pass.
- The Mock backend passes the complete reusable backend contract suite.

## Phase 5: Split Core scene encoding from output presentation

### Tasks

1. Inventory size-dependent and presentation-dependent fields in
   `FS_Core`/`FS_EffectsResources`.
2. Define the standard internal scene format and color-space contract.
3. Move Surface-format presentation pipeline code out of Effects.
4. Make Core encode a scene without knowing a Surface or present mode.
5. Keep CPU-side drawing resource behavior stable.
6. Add a deprecated compatibility adapter for `fs_core_encode()`.

### Files

- Modify `include/fullstack_core.h`.
- Modify `src/fullstack_core.c`.
- Modify `src/fullstack_core_private.h`.
- Modify `include/fullstack_effects.h`.
- Modify `src/fullstack_effects.c`.
- Modify `src/fullstack_effects.h`.
- Add `tests/core_offscreen_tests.c`.

### Gate

- Core builds and renders offscreen without App or a window backend.
- Effects no longer create presentation pipelines.
- Core source contains no Surface acquire/present logic.
- Legacy encode output matches existing screenshot baselines.

## Phase 6: RenderContext, targets, Frame Graph, and Presenter

### Tasks

1. Implement `FS_RenderTarget`, `FS_RenderContext`, `FS_RenderFrame`, and
   `FS_CommandBatch`.
2. Move framebuffer-size resources out of Core into RenderContext ownership.
3. Implement transactional resize and delayed resource retirement.
4. Implement the ordered lightweight Frame Graph and pass validation.
5. Implement per-window `FS_Presenter` with a shared immutable pipeline cache.
6. Implement Linear/sRGB conversion and remove the Surface-format downgrade.
7. Add Display P3/HDR descriptors even if initial backends return unsupported.
8. Implement explicit resource recreation after Device replacement.

### Files

- Add `include/fullstack_render.h`.
- Add `include/fullstack_present.h`.
- Add `src/render/fullstack_render_context.c`.
- Add `src/render/fullstack_render_frame.c`.
- Add `src/render/fullstack_frame_graph.c`.
- Add `src/render/fullstack_render_target.c`.
- Add `src/present/fullstack_present.c`.
- Add `tests/render_context_tests.c`.
- Add `tests/frame_graph_tests.c`.
- Add `tests/presentation_color_tests.c`.

### Gate

- No steady-state heap allocations occur in the tested frame path.
- Transactional resize preserves the previous resources on allocation failure.
- Linear/sRGB screenshot baselines pass.
- Effects, filters, and custom passes compose in the specified order.

## Phase 7: AppFrame and Surface state machine

### Tasks

1. Implement the stable `FS_AppFrame` token and exact state transitions.
2. Implement acquire, generic GPU submission, present, cancel, reset, and
   exactly-once Surface texture/view release.
3. Implement Surface generation and stale-frame rejection.
4. Defer resize/reconfigure/suspend/loss while a frame is outstanding.
5. Implement zero-size, timeout, occlusion, suboptimal, outdated, lost, and
   Device-lost results.
6. Enforce that end-frame accepts only presented or cancelled frames.

### Files

- Add `src/app/fullstack_app_frame.c`.
- Add `src/app/fullstack_app_surface.c`.
- Add `tests/app_frame_tests.c`.
- Add `tests/surface_state_tests.c`.

### Gate

- Every valid and invalid frame transition is tested.
- Reconfiguration never occurs while a Surface texture remains acquired.
- Multi-window frame generations remain independent.

## Phase 8: Rewrite the GLFW backend

### Tasks

1. Split GLFW factory, native window, events, Surface, and services.
2. Use `FS_GpuContext`; remove GPU creation from the backend.
3. Remove Core, Render, Effects, and Presenter ownership from the backend.
4. Translate GLFW callbacks into the generic queue and input state.
5. Preserve and safely restore host callbacks/user pointers for wrapped windows.
6. Implement wake, clipboard, cursor, and native-handle capabilities.
7. Move vendored tinyfiledialogs into the async file-dialog service.
8. Run the reusable backend contract suite against GLFW.

### Files

- Add `impl/backends/glfw/fullstack_glfw_app_backend.c`.
- Add `impl/backends/glfw/fullstack_glfw_window.c`.
- Add `impl/backends/glfw/fullstack_glfw_events.c`.
- Add `impl/backends/glfw/fullstack_glfw_surface.c`.
- Add `impl/backends/glfw/fullstack_glfw_services.c`.
- Modify or replace `impl/fullstack_glfw_backend.c` with a legacy facade.
- Modify or replace `impl/fullstack_glfw_backend_ext.c` with a legacy facade.
- Add `tests/glfw_backend_tests.c`.

### Gate

- GLFW passes the same backend contract suite as Mock.
- GLFW backend source contains no Core drawing or presentation-pipeline calls.
- File-dialog completion is serialized on the App thread.

## Phase 9: Split and migrate the C++ UI framework

### Tasks

1. Split `examples/ui/app.hpp` into application, window, renderer, event
   dispatcher, layout engine, and transition renderer units.
2. Replace GLFW/ext members with `FS_App` and `FS_AppWindow`.
3. Use logical dimensions for layout and framebuffer dimensions for rendering.
4. Move focus, hover, capture, text input, and IME dispatch onto generic events.
5. Make responsive relayout depend only on generic window metrics.
6. Add UI tests using the Mock backend.

### Files

- Add `examples/ui/application.hpp`.
- Add `examples/ui/window.hpp`.
- Add `examples/ui/renderer.hpp`.
- Add `examples/ui/event_dispatcher.hpp`.
- Add `examples/ui/layout_engine.hpp`.
- Reduce `examples/ui/app.hpp` to the convenience composition layer.
- Modify `examples/ui/element.hpp`, `text.hpp`, and `button.hpp` as needed.
- Add `tests/ui_app_tests.cpp`.

### Gate

- `rg` finds no GLFW or SDL symbols in `examples/ui/`.
- Mock resize, scale, pointer, keyboard, text, and IME tests pass.
- Existing responsive text wrapping and centered text cases pass.

## Phase 10: Transition engine and maintained example migration

### Tasks

1. Convert transition page capture to offscreen `FS_RenderTarget` objects.
2. Register transition composition as a Frame Graph pass.
3. Remove direct Device, Queue, Surface, format, acquire, and present access.
4. Migrate MORROW clock, UI demo, text demos, memory demo, clay demo, orbital
   catcher, radius profile, and the main compute demo.
5. Keep `wgpu_minimal_triangle.c` unchanged and separately linked.
6. Replace per-executable copies of Core sources with shared library targets.

### Files

- Modify `examples/ui/transition.hpp`.
- Modify `examples/ui/router.hpp`.
- Modify `examples/transition_demo.cpp`.
- Modify maintained example sources and `examples/CMakeLists.txt`.

### Gate

- All maintained examples build against shared targets.
- Transition rendering handles resize and Surface regeneration.
- Architecture audit finds no concrete backend symbols in migrated examples.
- The clock remains responsive and its async Monet/file-dialog path works.

## Phase 11: SDL3 backend

### Tasks

1. Add optional SDL3 discovery and build configuration.
2. Implement SDL3 factory, window, Surface, events, and services.
3. Map touch, IME editing, clipboard, cursor, URI, and dialogs.
4. Pass the reusable backend contract suite.
5. Run the same UI and transition applications without source changes.

### Files

- Add `impl/backends/sdl3/*`.
- Add `tests/sdl3_backend_tests.c`.
- Modify root and example CMake configuration.

### Gate

- Switching GLFW to SDL3 requires a runtime/configuration choice only.
- UI/demo source code remains unchanged.
- SDL3 passes the full backend contract suite.

## Phase 12: Android backend and build

### Tasks

1. Add the Android native lifecycle adapter and external-loop tick path.
2. Support `ANativeWindow`, Vulkan Surface creation, rotation, scale, insets,
   suspend/resume, Surface replacement, and low-memory events.
3. Add touch, key, UTF-8/IME, soft keyboard, URI, and Activity Result services.
4. Add an Android CMake/Gradle sample that runs the clock and basic UI.
5. Add emulator/device smoke-test instructions and automated build checks.

### Files

- Add `impl/backends/android/*`.
- Add `platform/android/` project files.
- Add `tests/android_contract_tests.c` where host-testable.

### Gate

- arm64 Android builds complete.
- Clock and basic UI run, rotate, suspend, resume, and recreate their Surface.
- File selection and IME completion return through generic App events/tasks.

## Phase 13: iOS backend and build

### Tasks

1. Add the Objective-C++ Scene/View lifecycle adapter.
2. Support `CAMetalLayer`, drawable resize, safe areas, orientation, background,
   foreground, and Surface replacement.
3. Add touch, hardware keys, UTF-8/IME, soft keyboard, URI, and document picker.
4. Add an Xcode/CMake sample that runs the clock and basic UI.
5. Add simulator build and device smoke-test instructions.

### Files

- Add `impl/backends/ios/*`.
- Add `platform/ios/` project files.
- Add host-testable lifecycle adapter tests.

### Gate

- iOS simulator/device builds complete.
- Clock and basic UI handle orientation, safe areas, background/foreground, IME,
  and document selection through generic APIs.

## Phase 14: Legacy facade and enforcement

### Tasks

1. Move old declarations to `include/fullstack_legacy.h`.
2. Make the old GLFW backend/ext symbols pure facades over the new App stack.
3. Add deprecation annotations and `FS_BUILD_LEGACY`.
4. Default legacy on for the first release and add the documented next-minor
   default-off switch metadata.
5. Add a migration guide mapping every legacy symbol.
6. Enforce allowed legacy includes in CI.

### Files

- Add `include/fullstack_legacy.h`.
- Add `impl/legacy/*`.
- Add `docs/fullstack-app-migration.md`.
- Modify CMake options and architecture audit.

### Gate

- Legacy tests pass using only the facade.
- No maintained target includes the legacy header.
- The legacy implementation contains no independent event, Surface, Core,
  Render, or presentation logic.

## Phase 15: Final completion audit

### Tasks

1. Run requirement-by-requirement verification against the approved design.
2. Run all unit, contract, ownership, frame, event, render, UI, and architecture
   tests.
3. Run GLFW and SDL3 desktop smoke tests.
4. Verify Android and iOS build and lifecycle smoke evidence.
5. Audit steady-state allocation and resource-release diagnostics.
6. Confirm every maintained example is migrated and the minimal triangle remains
   intentionally independent.

### Required commands

```powershell
cmake -S . -B build -DFS_BUILD_TESTS=ON -DFS_BUILD_BACKEND_GLFW=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Equivalent SDL3, Android, and iOS configuration/build commands must be recorded
once their platform projects exist.

### Completion gate

The implementation is complete only when every completion criterion in the
approved architecture specification has direct build, test, audit, or runtime
evidence. A successful GLFW build alone is insufficient.
