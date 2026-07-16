# A2UI FS Design App implementation plan

This plan implements the approved A2UI FS Design App architecture through
test-first, dependency-ordered milestones. It builds a headless vertical slice
before the complete GUI, then adds collaboration, security, export, recovery,
and performance hardening.

> **Note:** This is an implementation plan for a preview feature under active
> development.

The plan preserves all unrelated worktree changes. Each commit must stage only
the files named by its task. Root `include/` and `src/` remain independent from
the new design, protocol, App, and third-party code.

## Implementation strategy

The implementation proceeds through three product gates.

1. Build a headless vertical slice that creates a document, imports A2UI,
   replays a `CanvasLayer`, renders an offscreen image, saves `.fsdesign`,
   reloads it, and reproduces the same document and command hashes.
2. Build the Figma-style GUI on the proven runtime, with selection,
   transforms, properties, i18n preview, history, and embedded MCP.
3. Add full CRDT collaboration, preview branches, secure archives, recovery,
   export breadth, performance limits, and backend packaging.

Do not begin the next gate until the previous gate's tests, architecture audit,
and narrow smoke commands pass.

## Baseline and guardrails

Before adding files, record the current repository state and verify the existing
App and Core baselines.

Run these commands from the repository root:

```powershell
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
powershell -NoProfile -ExecutionPolicy Bypass `
  -File tools/architecture/check_layers.ps1 -Root .
git status --short
```

Record pre-existing failures without repairing unrelated code. Keep the
existing `fullstack_result`, `fullstack_gpu`, `fullstack_app_foundation`,
`fullstack_app`, `fullstack_core_runtime`, `fullstack_render`, and backend
targets building throughout the work.

## Build options and target graph

Add design features behind explicit CMake options so existing consumers can
continue to build the root SDK and App without MCP, JSON, archive, ICU, or GUI
dependencies.

Add these options to `CMakeLists.txt`:

```text
FS_BUILD_DESIGN_RUNTIME       ON
FS_BUILD_DESIGN_A2UI          ON
FS_BUILD_DESIGN_PERSISTENCE   ON
FS_BUILD_DESIGN_MCP           ON
FS_BUILD_DESIGN_HEADLESS      ON
FS_BUILD_DESIGN_GUI           ON
FS_BUILD_DESIGN_COLLAB        ON
FS_BUILD_DESIGN_ICU           AUTO
FS_BUILD_DESIGN_FUZZ          OFF
```

During staged implementation, keep a new option disabled by default until its
target and tests pass. The final preview build enables the runtime, A2UI,
persistence, MCP, headless, GUI, and collaboration options by default.
`FS_BUILD_DESIGN_ICU` is a string cache value with `AUTO`, `ON`, and `OFF`.
The collaboration option controls Presence and multi-client synchronization;
the document's CRDT primitives are always part of the design runtime.

The implementation adds these targets in dependency order:

```text
fullstack_design_foundation
fullstack_design_document
fullstack_design_runtime
fullstack_canvas_vm
fullstack_design_a2ui
fullstack_design_persistence
fullstack_design_i18n
fullstack_design_mcp
fullstack_design_impl_json
fullstack_design_impl_archive
fullstack_design_impl_http
fullstack_design_impl_i18n
a2ui_design_headless
a2ui_design_app
```

Generic design targets may link root public targets and lower generic design
targets. Only `app/impl` targets may link concrete third-party libraries. The
GUI may link App backends, `examples/ui`, Taffy, and generic design targets.

## Third-party dependency policy

Vendor third-party source only under root `third_party/`. Record its upstream
URL, exact revision, license, and local modifications. Mark every vendored tree
as `linguist-vendored` in `.gitattributes`.

Use this dependency plan:

| Dependency | Location | Purpose |
| --- | --- | --- |
| A2UI schemas | `third_party/a2ui/` | Pinned protocol schemas |
| yyjson | `third_party/yyjson/` | Bounded JSON parsing and writing |
| miniz | `third_party/miniz/` | `.fsdesign` ZIP container support |
| CivetWeb | `third_party/civetweb/` | Local Streamable HTTP and SSE |
| utf8proc | `third_party/utf8proc/` | Default Unicode grapheme segmentation |
| SHA-256 | `third_party/crypto-algorithms/` | Content hashes |
| PDFio | `third_party/pdfio/` | PDF writing and object serialization |

Use ICU4C through `find_package(ICU COMPONENTS uc i18n data)` rather than
vendoring it. The portable package copies required ICU runtime libraries when
the ICU adapter is enabled. A compact built-in formatter keeps projects
readable when ICU is unavailable, while the ICU test configuration is required
for full plural and MessageFormat completion.

Before adding any dependency, verify its license, build warnings, Windows
toolchain compatibility, and source provenance in a standalone commit. Do not
combine vendored source and application behavior in one commit.

## Milestone 1: design foundation

This milestone establishes stable C ABI types, immutable snapshots, typed
values, retained nodes, and the owner-thread runtime without parsing JSON or
calling the GPU.

### Step 1: add design ABI and identity types

Define fixed-width, allocator-safe public types before implementing containers
or document behavior.

Files to create:

- `app/include/fullstack_design.h`
- `app/include/fullstack_design_types.h`
- `app/src/design/fullstack_design_types.c`
- `test/design_abi_tests.c`
- `test/design_id_tests.c`

Files to update:

- `CMakeLists.txt`
- `test/CMakeLists.txt`

Implement these contracts:

1. Define `FS_EntityId`, `FS_ActorId`, `FS_OpId`, `FS_HlcTimestamp`, and
   `FS_DesignRevision` with fixed-width storage.
2. Add canonical Base32 parse and format functions with caller-owned buffers.
3. Add comparison, zero, validity, and stable hashing helpers.
4. Add `FS_DESIGN_ABI_VERSION`, `struct_size` initializers, `FS_API`, and
   `FS_CALL` to all public declarations.
5. Reuse `FS_Result`, `FS_Error`, `FS_Allocator`, and `FS_DiagnosticSink`.
6. Add design error-domain constants without changing root result behavior.
7. Expose opaque forward declarations only from `fullstack_design.h`.

Write failing tests first for ABI size, C and C++ inclusion, malformed Base32,
round-trip IDs, ordering, and equal-ID hash stability. Build only
`fs_design_abi_tests` and `fs_design_id_tests` until they pass.

### Step 2: add owned values and immutable snapshots

Implement the value system and snapshot ownership needed by every higher
layer.

Files to create:

- `app/include/fullstack_design_value.h`
- `app/include/fullstack_design_snapshot.h`
- `app/src/design/fullstack_design_value.c`
- `app/src/design/fullstack_design_snapshot.c`
- `app/src/design/fullstack_design_arena.c`
- `app/src/design/fullstack_design_arena.h`
- `test/design_value_tests.c`
- `test/design_snapshot_tests.c`

Implement these value kinds:

```text
null
boolean
signed integer
unsigned integer
double
UTF-8 string
color
entity reference
resource reference
list
object
variable binding
localized message binding
```

Use allocator-owned immutable strings, lists, and objects. Reject invalid UTF-8,
non-finite doubles, duplicate object keys, excessive nesting, and size
overflow. Snapshots are immutable, reference-counted, and query-safe from
non-owner threads.

Write tests for deep clone, release order, allocator failure at every
allocation point, nesting limits, deterministic object key order, and snapshot
access after the live document advances.

### Step 3: add retained nodes and document queries

Create the semantic node tree with no CRDT merge yet. This establishes the
data model used by GUI, A2UI, persistence, and Canvas.

Files to create:

- `app/include/fullstack_design_document.h`
- `app/include/fullstack_design_node.h`
- `app/src/design/fullstack_design_document.c`
- `app/src/design/fullstack_design_document_private.h`
- `app/src/design/fullstack_design_node.c`
- `app/src/design/fullstack_design_index.c`
- `app/src/design/fullstack_design_index.h`
- `test/design_document_tests.c`
- `test/design_tree_tests.c`

Implement pages, frames, groups, shapes, paths, text, localized text, images,
icons, components, instances, slots, comments, and `CanvasLayer` placeholders.
Store properties in typed maps and children in stable placement order.

Provide snapshot queries for node by ID, children, parent, page, property,
component origin, instance overrides, and resource references. Reject duplicate
IDs, cross-document references, invalid parents, cycles, and malformed
component paths.

Write tests for empty documents, page creation, nested nodes, components,
instance overrides, deletion, restored identities, invalid cycles, and
deterministic tree serialization order.

### Step 4: add the owner-thread runtime and request queue

Serialize all mutation through one owner thread before adding transport or
worker code.

Files to create:

- `app/include/fullstack_design_runtime.h`
- `app/include/fullstack_task_provider.h`
- `app/src/design/fullstack_design_runtime.c`
- `app/src/design/fullstack_design_request.c`
- `app/src/design/fullstack_design_request.h`
- `app/impl/threading/fullstack_native_task_provider.c`
- `app/impl/threading/fullstack_native_task_provider.h`
- `test/design_runtime_tests.c`
- `test/design_threading_tests.c`
- `test/support/mock_task_provider.c`
- `test/support/mock_task_provider.h`

Implement runtime create, begin-destroy, release, owner-thread checks, request
posting, cancellation, immutable snapshot publication, and completion delivery.
Use the existing App allocator and task patterns. Generic runtime code submits
background work through `FS_TaskProvider`; platform thread creation and joins
remain in `app/impl`. Do not call a user completion from a worker thread.

Write tests for wrong-thread mutation, cross-thread snapshot reads,
cancellation, destroy with queued work, allocator failure, callback re-entry
rejection, and late worker completion after source revision changes.

## Milestone 2: transactions, history, and CRDT

This milestone makes every mutation atomic and convergent. The GUI, MCP, and
A2UI adapters must use this transaction path rather than private shortcuts.

### Step 5: add typed transactions and strict preconditions

Define a compact typed operation model and implement atomic local commits
before distributed merge.

Files to create:

- `app/include/fullstack_design_transaction.h`
- `app/src/design/fullstack_design_transaction.c`
- `app/src/design/fullstack_design_validate.c`
- `app/src/design/fullstack_design_validate.h`
- `test/design_transaction_tests.c`
- `test/design_transaction_rollback_tests.c`

Implement insert, delete, restore, set property, clear property, move, copy,
replace, set variable, set localized message, bind resource, and Canvas stream
reference operations. Support `merge`, `strict`, and `dryRun` modes.

Validate all operations, resource references, permissions, and preconditions
before publishing a new snapshot. Write allocation-failure tests that prove no
partial mutation escapes after any failed operation.

### Step 6: add operation IDs, HLC, and state vectors

Implement causal metadata and the deterministic total order required by every
CRDT domain.

Files to create:

- `app/include/fullstack_design_crdt.h`
- `app/src/crdt/fullstack_crdt_clock.c`
- `app/src/crdt/fullstack_crdt_state_vector.c`
- `app/src/crdt/fullstack_crdt_register.c`
- `app/src/crdt/fullstack_crdt_or_map.c`
- `app/src/crdt/fullstack_crdt_private.h`
- `test/crdt_clock_tests.c`
- `test/crdt_register_tests.c`
- `test/crdt_or_map_tests.c`

Implement HLC advancement, remote observation, state-vector comparison,
duplicate operation rejection, remove-wins node existence, LWW registers, and
multi-value registers. Apply the exact HLC, Actor ID, counter, and canonical
hash total order from the design specification.

Add tests that permute operation delivery, duplicate every operation, create
equal physical timestamps, and inject an invalid duplicate Op ID with a
different payload.

### Step 7: add atomic tree placement

Implement parent-scoped LSEQ positions and deterministic cycle recovery as one
atomic placement value.

Files to create:

- `app/src/crdt/fullstack_crdt_lseq.c`
- `app/src/crdt/fullstack_crdt_tree.c`
- `app/src/crdt/fullstack_crdt_tree.h`
- `test/crdt_lseq_tests.c`
- `test/crdt_tree_tests.c`

Write failing tests for insertion between adjacent nodes, repeated midpoint
insertion, concurrent moves, cross-page moves, incompatible parent positions,
cycle creation, fallback to a losing placement, and deterministic
**Recovered Layers** ordering.

Do not store parent and position in separate registers. One placement operation
must select both values.

### Step 8: add collaborative text and locale sequences

Implement grapheme-based text collaboration behind a versioned segmentation
provider.

Files to create:

- `app/include/fullstack_design_text.h`
- `app/src/crdt/fullstack_crdt_text.c`
- `app/src/crdt/fullstack_crdt_text.h`
- `app/include/fullstack_unicode_provider.h`
- `app/impl/unicode/fullstack_utf8proc_provider.c`
- `app/impl/unicode/fullstack_utf8proc_provider.h`
- `test/crdt_text_tests.c`
- `test/unicode_provider_tests.c`

Vendor utf8proc in a separate dependency commit. Test Latin combining marks,
Chinese, Arabic, emoji ZWJ sequences, flags, skin tones, concurrent insertion,
concurrent deletion, selection anchors, and explicit Unicode-version migration.

### Step 9: add compensating undo and redo

Build undo on operation provenance rather than snapshot rollback.

Files to create:

- `app/include/fullstack_design_history.h`
- `app/src/design/fullstack_design_history.c`
- `app/src/design/fullstack_design_undo.c`
- `test/design_history_tests.c`
- `test/design_undo_dependency_tests.c`

Implement local-Actor transaction history, `undo_of`, redo with new operation
IDs, property compensation, placement restoration, deletion restore, and
dependency-aware insertion undo.

Tests must prove that undo never erases causally later remote children,
properties, bindings, instances, or Canvas edits. Dependent insert undo must
move the retained node to **Recovered Layers** and expose an undo conflict.

### Step 10: add the convergence harness and log compaction

Create a reusable randomized replica harness before adding network transport.

Files to create:

- `test/support/design_replica_harness.h`
- `test/support/design_replica_harness.c`
- `test/crdt_convergence_tests.c`
- `test/crdt_offline_reconnect_tests.c`
- `app/src/crdt/fullstack_crdt_compaction.c`
- `test/crdt_compaction_tests.c`

Generate operations from multiple Actors, reorder and duplicate delivery,
disconnect replicas, reconnect from snapshots, remove peers, and compact only
after acknowledgement. Compare canonical document hashes after every complete
delivery schedule.

## Milestone 3: Canvas API coverage and VM

This milestone creates a complete, auditable mapping from the public Canvas API
to typed commands, queries, resources, or explicit runtime exclusions.

### Step 11: generate the Canvas API coverage manifest

Make unclassified root Canvas APIs a build failure before defining opcodes.

Files to create:

- `tools/design/generate_canvas_api_coverage.ps1`
- `app/src/canvas/canvas_api_coverage.json`
- `app/src/canvas/canvas_opcode_registry.json`
- `test/canvas_api_coverage_test.cmake`
- `test/canvas_api_coverage_tests.c`

Files to update:

- `test/CMakeLists.txt`
- `tools/architecture/check_layers.ps1`

Classify every public declaration in `include/fullstack_core.h` as `opcode`,
`query`, `resource`, or `runtime-excluded`. Record the root source revision,
payload version, validation rule, feature bits, and exclusion reason.

Register a CTest that regenerates the declaration inventory and fails for an
unclassified, duplicate, or silently reclassified API.

### Step 12: add Canvas command ABI and validator

Define stable opcodes and validate command blocks without an `FS_Core`.

Files to create:

- `app/include/fullstack_canvas_vm.h`
- `app/include/fullstack_canvas_command.h`
- `app/src/canvas/fullstack_canvas_command.c`
- `app/src/canvas/fullstack_canvas_validate.c`
- `app/src/canvas/fullstack_canvas_validate.h`
- `test/canvas_command_abi_tests.c`
- `test/canvas_command_validation_tests.c`

Implement command headers, payload readers, command versions, feature bits,
optional extension skipping, finite-number checks, state-stack balance, path
limits, clip limits, text limits, and exact payload-length checks.

Fuzz malformed byte streams in a bounded unit-test corpus before enabling the
optional libFuzzer target.

### Step 13: add resource IDs and generation-checked handles

Implement resource descriptions independently from decoding or GPU upload.

Files to create:

- `app/include/fullstack_canvas_resource.h`
- `app/src/canvas/fullstack_canvas_resource.c`
- `app/src/canvas/fullstack_canvas_resource_table.c`
- `app/src/canvas/fullstack_canvas_resource_table.h`
- `test/canvas_resource_tests.c`

Support gradients, patterns, images, fonts, Path2D, and pixel buffers. Use
stable external resource IDs and internal slot, generation, and type handles.

Test stale handles, type mismatch, slot reuse, content deduplication, missing
resource placeholders, and allocator failure.

### Step 14: replay commands into FS Canvas

Add the adapter that maps validated opcodes to the real public root API.

Files to create:

- `app/src/canvas/fullstack_canvas_vm.c`
- `app/src/canvas/fullstack_canvas_replay.c`
- `app/src/canvas/fullstack_canvas_replay.h`
- `app/src/canvas/fullstack_canvas_query.c`
- `test/canvas_vm_mock_tests.c`
- `test/canvas_vm_offscreen_tests.c`

Use a Mock replay sink for exact opcode tests. Use `FS_Core` only on its owning
render thread for offscreen integration tests. Keep text measurement, pixel
readback, and hit testing in query APIs rather than replay commands.

Generate coverage tests from `canvas_api_coverage.json` so every opcode and
query classification executes at least once.

### Step 15: add Canvas stream chunk CRDT

Integrate immutable content-addressed command chunks with document transactions.

Files to create:

- `app/src/crdt/fullstack_crdt_canvas_stream.c`
- `app/src/crdt/fullstack_crdt_canvas_stream.h`
- `app/src/canvas/fullstack_canvas_stream.c`
- `app/include/fullstack_canvas_stream.h`
- `test/crdt_canvas_stream_tests.c`
- `test/canvas_stream_cache_tests.c`

Test concurrent edits to different chunks, conflicts in the same range,
deterministic winner selection, branch preservation, undo, deduplication, and
compiled-stream cache invalidation.

## Milestone 4: A2UI normalization

This milestone vendors pinned schemas and converts both accepted A2UI versions
into the same typed design transaction model.

### Step 16: vendor A2UI schemas and JSON support

Add third-party content and schema manifests without changing runtime behavior.

Files to create or vendor:

- `third_party/a2ui/`
- `third_party/yyjson/`
- `app/impl/a2ui/schema_manifest.json`
- `app/impl/json/fullstack_yyjson_adapter.c`
- `app/impl/json/fullstack_yyjson_adapter.h`
- `app/include/fullstack_json_provider.h`
- `test/a2ui_schema_manifest_test.cmake`
- `test/json_provider_tests.c`

Files to update:

- `.gitattributes`
- `CMakeLists.txt`
- `test/CMakeLists.txt`

Pin A2UI commit `0190314c56eb136bb2b1541d8385d18c1131b9fe`.
Generate and verify SHA-256 for every accepted schema file. Configure yyjson
with bounded allocation, maximum depth, maximum string size, and duplicate-key
rejection at the adapter boundary. The CMake manifest test uses
`file(SHA256 ...)`, so the runtime SHA-256 provider remains independent and can
arrive with persistence.

### Step 17: add version-neutral A2UI IR

Define normalized messages before writing either version parser.

Files to create:

- `app/include/fullstack_a2ui.h`
- `app/include/fullstack_a2ui_catalog.h`
- `app/src/a2ui/fullstack_a2ui_ir.c`
- `app/src/a2ui/fullstack_a2ui_catalog.c`
- `app/src/a2ui/fullstack_a2ui_validate.c`
- `test/a2ui_ir_tests.c`
- `test/a2ui_catalog_tests.c`

Represent surface lifecycle, component updates, data updates, actions,
responses, bindings, and opaque optional components. Add FS Catalog component
descriptors for semantic nodes and `CanvasLayer`.

Test catalog negotiation, required and optional features, unknown fields,
opaque round-trip, default values, and unsupported required components.

### Step 18: implement v0.9.1 and v1.0 adapters

Parse each pinned wire version into the normalized IR and translate IR into
document transactions.

Files to create:

- `app/src/a2ui/fullstack_a2ui_v091.c`
- `app/src/a2ui/fullstack_a2ui_v1_candidate.c`
- `app/src/a2ui/fullstack_a2ui_controller.c`
- `app/src/a2ui/fullstack_a2ui_export.c`
- `test/fixtures/a2ui/v0.9.1/`
- `test/fixtures/a2ui/v1.0-candidate-2026-07-17/`
- `test/a2ui_v091_tests.c`
- `test/a2ui_v1_tests.c`
- `test/a2ui_roundtrip_tests.c`

Test `createSurface`, `updateComponents`, `updateDataModel`, `deleteSurface`,
v1 action request and response, malformed ordering, unknown optional data, and
loss reports. The adapters must submit normal `FS_DesignTransaction` values.

## Milestone 5: persistence, workspace policy, and i18n

This milestone makes the headless vertical slice durable and secure before MCP
or the GUI can write projects.

### Step 19: add filesystem, hash, and workspace provider interfaces

Keep generic persistence independent from platform paths and concrete crypto.

Files to create:

- `app/include/fullstack_workspace_policy.h`
- `app/include/fullstack_filesystem_provider.h`
- `app/include/fullstack_hash_provider.h`
- `app/include/fullstack_image_encode_provider.h`
- `app/src/io/fullstack_workspace_policy.c`
- `app/src/io/fullstack_design_io.c`
- `app/impl/filesystem/fullstack_native_filesystem.c`
- `app/impl/filesystem/fullstack_native_filesystem.h`
- `app/impl/hash/fullstack_sha256_provider.c`
- `app/impl/image/fullstack_stb_image_write_provider.c`
- `app/impl/image/fullstack_stb_image_write_provider.h`
- `test/workspace_policy_tests.c`
- `test/filesystem_provider_tests.c`
- `test/hash_provider_tests.c`
- `test/image_encode_provider_tests.c`

Vendor the selected SHA-256 source separately. Test traversal, Windows drive
and UNC paths, Junction and symlink escape, case folding, atomic replacement,
temporary cleanup, hash fixtures, allocator failure, and bounded PNG and JPEG
encoding through the existing stb implementation available to root adapters.

### Step 20: add canonical project persistence

Implement unpacked `.fsdesign` directories before ZIP containers.

Files to create:

- `app/include/fullstack_design_persistence.h`
- `app/src/persistence/fullstack_design_manifest.c`
- `app/src/persistence/fullstack_design_serialize.c`
- `app/src/persistence/fullstack_design_deserialize.c`
- `app/src/persistence/fullstack_design_snapshot_io.c`
- `app/src/persistence/fullstack_design_operation_log.c`
- `test/design_persistence_tests.c`
- `test/design_recovery_tests.c`
- `test/fixtures/fsdesign/`

Write canonical `manifest.json`, `document.json`, snapshot, operation log,
Canvas chunks, resources, locale catalogs, and preview metadata. Preserve
unknown extension fields.

Test round-trip hashes, interrupted save, stale temporary files, unknown
fields, corrupt resources, partial logs, migration dry run, and recovery
without replacing the source project.

### Step 21: add secure ZIP containers

Add archive support only after unpacked projects pass all recovery tests.

Files to vendor or create:

- `third_party/miniz/`
- `app/include/fullstack_archive_provider.h`
- `app/impl/archive/fullstack_miniz_archive.c`
- `app/impl/archive/fullstack_miniz_archive.h`
- `test/archive_security_tests.c`
- `test/fixtures/archive/`

Reject duplicate canonical names, absolute paths, traversal, symlinks,
hard links, unsupported encryption, excessive entry count, excessive expanded
size, excessive compression ratio, and hash mismatch. Extract only to a private
temporary location and publish atomically.

### Step 22: add autosave and migration

Implement incremental recovery journals and deterministic format migration.

Files to create:

- `app/src/persistence/fullstack_design_autosave.c`
- `app/src/persistence/fullstack_design_migrate.c`
- `app/include/fullstack_design_migrate.h`
- `test/design_autosave_tests.c`
- `test/design_migration_tests.c`

Use dirty revisions and debounce rather than periodic full saves. Test process
termination between every publication step, recovery comparison, retained
peer frontiers, tombstone compaction, and offline replica resynchronization.

### Step 23: add locale catalogs and formatters

Implement editor and document localization behind provider interfaces.

Files to create:

- `app/include/fullstack_design_i18n.h`
- `app/include/fullstack_i18n_provider.h`
- `app/src/i18n/fullstack_locale.c`
- `app/src/i18n/fullstack_locale_catalog.c`
- `app/src/i18n/fullstack_locale_fallback.c`
- `app/src/i18n/fullstack_i18n_basic.c`
- `app/impl/i18n/fullstack_icu_i18n.c`
- `app/impl/i18n/fullstack_icu_i18n.h`
- `test/i18n_catalog_tests.c`
- `test/i18n_fallback_tests.c`
- `test/i18n_icu_tests.c`
- `test/i18n_layout_diagnostics_tests.c`

Support exact and parent-locale fallback, project default, source fallback,
named arguments, plural branches, locale-sensitive formatting, LTR, RTL,
pseudolocalization, font overrides, and translation provenance.

Test `zh-Hans-CN`, Arabic RTL, Japanese, combining characters, missing glyphs,
text expansion, unsupported plural branches, and deterministic locale-matrix
ordering.

## Milestone 6: first headless vertical slice

This milestone proves the complete lower architecture before HTTP or GUI work.

### Step 24: create the headless executable

Build a C executable that composes the generic runtime and concrete local
adapters.

Files to create:

- `examples/a2ui_design_headless/main.c`
- `examples/a2ui_design_headless/design_headless_cli.c`
- `examples/a2ui_design_headless/design_headless_cli.h`
- `test/design_headless_smoke.cmake`

Files to update:

- `examples/CMakeLists.txt`
- `test/CMakeLists.txt`

Implement `validate`, `apply-a2ui`, `render`, `export`, and `generate-c`
command routing with no model-provider integration.

At this gate, `export` supports the PNG path used by the vertical slice.
Register `generate-c` and later export formats in the command parser, but return
`FS_RESULT_UNSUPPORTED` with a clear capability message until their milestones
land.

Add one deterministic fixture that performs this flow:

1. Create an empty document.
2. Import A2UI with a semantic frame and `CanvasLayer`.
3. Replay paths, text, a gradient, and an image placeholder offscreen.
4. Export PNG.
5. Save unpacked `.fsdesign`.
6. Reopen it.
7. Compare document, Canvas command, and output image hashes.

The vertical slice gate passes only when this flow works with
`FS_BUILD_DESIGN_MCP=OFF` and `FS_BUILD_DESIGN_GUI=OFF`.

## Milestone 7: MCP Session Hub and transports

This milestone exposes the proven runtime to agents without adding any direct
OpenAI, Anthropic, or Gemini client.

### Step 25: implement transport-neutral MCP JSON-RPC

Build the MCP server state machine against abstract byte-stream and JSON
providers.

Files to create:

- `app/include/fullstack_design_mcp.h`
- `app/include/fullstack_mcp_transport.h`
- `app/src/mcp/fullstack_mcp_session.c`
- `app/src/mcp/fullstack_mcp_jsonrpc.c`
- `app/src/mcp/fullstack_mcp_tools.c`
- `app/src/mcp/fullstack_mcp_resources.c`
- `app/src/mcp/fullstack_mcp_progress.c`
- `app/src/mcp/fullstack_mcp_sampling.c`
- `test/mcp_jsonrpc_tests.c`
- `test/mcp_tool_contract_tests.c`
- `test/mcp_session_resume_tests.c`

Implement initialization, capability negotiation, tools, resources,
subscriptions, progress, cancellation, logging, elicitation, and Sampling.
Map every mutation to `FS_DesignTransaction` and every query to an immutable
snapshot.

Test duplicate request IDs, cancellation races, reconnect with a state vector,
strict and merge transactions, dry runs, permission errors, and bounded JSON.

### Step 26: add Streamable HTTP

Add the local multi-client transport with explicit security defaults.

Files to vendor or create:

- `third_party/civetweb/`
- `app/impl/mcp/fullstack_civetweb_transport.c`
- `app/impl/mcp/fullstack_civetweb_transport.h`
- `app/impl/mcp/fullstack_mcp_auth.c`
- `test/mcp_http_integration_tests.c`
- `test/mcp_http_security_tests.c`

Bind loopback by default. Require an unpredictable session token, validate
`Origin`, cap request and response sizes, enforce timeouts, and support SSE
streaming. Remote bind configuration must fail without TLS, authentication,
and an Origin allowlist.

### Step 27: add the stdio bridge

Create a small executable that forwards stdio MCP messages to the local Session
Hub rather than duplicating server logic.

Files to create:

- `examples/a2ui_design_mcp_bridge/main.c`
- `app/impl/mcp/fullstack_mcp_stdio_bridge.c`
- `test/mcp_stdio_bridge_tests.c`

Test framing, partial reads and writes, process shutdown, invalid tokens,
server restart, cancellation, and no duplicate transaction after reconnect.

### Step 28: expose MCP from headless mode

Add `serve-mcp` to the headless executable and register end-to-end tests.

Files to update:

- `examples/a2ui_design_headless/main.c`
- `examples/a2ui_design_headless/design_headless_cli.c`
- `test/CMakeLists.txt`

Run a scripted MCP session that creates a Surface, edits semantic layers,
updates a Canvas stream, switches locale, renders a preview, saves the project,
and reopens it in another process.

## Milestone 8: Figma-style GUI shell

This milestone embeds the runtime and Session Hub in a responsive desktop
editor. It does not introduce a second document model inside C++ UI code.

### Step 29: create the GUI application skeleton

Create a thin C++ shell using `examples/ui`, Taffy, and backend-neutral App
events.

Files to create:

- `examples/a2ui_design_app/main.cpp`
- `examples/a2ui_design_app/design_application.hpp`
- `examples/a2ui_design_app/design_application.cpp`
- `examples/a2ui_design_app/design_workspace.hpp`
- `examples/a2ui_design_app/design_workspace.cpp`
- `examples/a2ui_design_app/design_theme.hpp`
- `examples/a2ui_design_app/design_commands.hpp`
- `examples/a2ui_design_app/locales/en-US.json`
- `examples/a2ui_design_app/locales/zh-Hans.json`
- `examples/a2ui_design_app/locales/ar-SA.json`
- `test/design_gui_shell_smoke.cmake`

Files to update:

- `examples/CMakeLists.txt`
- `test/CMakeLists.txt`

Recreate the approved Pencil shell: top toolbar, Pages and Layers panel,
infinite Canvas, property inspector, agent activity, runtime status, locale,
MCP status, preview, and share actions.

Route every visible editor string through the design i18n provider. The first
GUI commit includes complete English and Simplified Chinese catalogs plus an
Arabic smoke catalog for direction and fallback tests.

Use only generic App events and window metrics. Add GLFW and SDL3 automated
smoke paths that open one document, render one frame, resize through all editor
breakpoints, and exit.

### Step 30: add viewport navigation and selection

Implement the Canvas camera, hit testing, layer-tree synchronization, and
selection overlay.

Files to create:

- `examples/a2ui_design_app/design_viewport.hpp`
- `examples/a2ui_design_app/design_viewport.cpp`
- `examples/a2ui_design_app/design_selection.hpp`
- `examples/a2ui_design_app/design_selection.cpp`
- `examples/a2ui_design_app/design_overlay.hpp`
- `examples/a2ui_design_app/design_overlay.cpp`
- `test/design_selection_tests.cpp`
- `test/design_viewport_tests.cpp`

Support pan, zoom, fit-all, fit-selection, deep selection, marquee selection,
Shift multi-select, isolation, smart guides, rulers, distance labels, and high
DPI coordinate conversion.

Commit one transaction on pointer release rather than one operation per motion
event. Add tests for logical and framebuffer scaling, nested transforms,
locked and hidden layers, and `CanvasLayer` whole-node selection.

### Step 31: add creation and transform tools

Implement the first editing tool set against design transactions.

Files to create:

- `examples/a2ui_design_app/design_tools.hpp`
- `examples/a2ui_design_app/design_tools.cpp`
- `examples/a2ui_design_app/design_transform_gizmo.hpp`
- `examples/a2ui_design_app/design_transform_gizmo.cpp`
- `test/design_tool_tests.cpp`
- `test/design_transform_tests.cpp`

Support Select, Frame, Section, Rectangle, Ellipse, Polygon, Line, Pen, Text,
Image, Icon, Component, Slot, `CanvasLayer`, and Comment. Implement move,
eight-direction resize, rotation, aspect lock, center resize, duplicate drag,
align, and distribute.

### Step 32: add panels and responsive behavior

Bind Pages, Layers, Assets, Components, Design, Prototype, Inspect, Agent, and
Localization panels to immutable snapshots and transactions.

Files to create:

- `examples/a2ui_design_app/panels/pages_layers_panel.hpp`
- `examples/a2ui_design_app/panels/pages_layers_panel.cpp`
- `examples/a2ui_design_app/panels/properties_panel.hpp`
- `examples/a2ui_design_app/panels/properties_panel.cpp`
- `examples/a2ui_design_app/panels/agent_panel.hpp`
- `examples/a2ui_design_app/panels/agent_panel.cpp`
- `examples/a2ui_design_app/panels/localization_panel.hpp`
- `examples/a2ui_design_app/panels/localization_panel.cpp`
- `test/design_responsive_shell_tests.cpp`

Implement the approved width classes: full three columns, overlay right panel,
one side panel, and touch review sheets. Add mixed property values, numeric
scrubbing, variable bindings, component overrides, locale switching, RTL,
pseudolocale, and Locale Matrix previews.

### Step 33: add history, command palette, and accessibility

Expose the generic history and action model through editor interactions.

Files to create:

- `examples/a2ui_design_app/design_history_ui.hpp`
- `examples/a2ui_design_app/design_history_ui.cpp`
- `examples/a2ui_design_app/design_command_palette.hpp`
- `examples/a2ui_design_app/design_command_palette.cpp`
- `examples/a2ui_design_app/design_accessibility.hpp`
- `examples/a2ui_design_app/design_accessibility.cpp`
- `test/design_shortcut_tests.cpp`
- `test/design_accessibility_tests.cpp`

Implement the approved shortcuts, focus order, semantic accessibility tree,
high contrast, reduced motion, command search, history provenance, undo
conflicts, and recoverable error presentation.

### Step 34: integrate on-demand rendering

Use the existing App frame scheduler and separate editor dirty domains.

Files to update:

- `examples/a2ui_design_app/design_application.cpp`
- `examples/a2ui_design_app/design_workspace.cpp`
- `examples/ui/application.hpp`
- `test/design_gui_on_demand_smoke.cmake`

Add Canvas, panel, overlay, agent, layout, paint, and animation invalidation.
Idle windows must use `FS_APP_PUMP_WAIT`. Animations request another frame with
`fs_app_window_request_animation_frame()`.

Add smoke instrumentation that proves zero present calls during a stable idle
interval and bounded redraw after a document transaction.

## Milestone 9: agent preview and collaboration

This milestone adds multi-client visual collaboration after local editing and
MCP mutation are stable.

### Step 35: add preview branches and visual diffs

Implement isolated large-change previews without modifying the live document.

Files to create:

- `app/include/fullstack_design_branch.h`
- `app/src/design/fullstack_design_branch.c`
- `app/src/design/fullstack_design_diff.c`
- `app/include/fullstack_design_diff.h`
- `examples/a2ui_design_app/design_diff_overlay.cpp`
- `examples/a2ui_design_app/design_diff_overlay.hpp`
- `test/design_branch_tests.c`
- `test/design_diff_tests.c`

Support create, refine, accept, reject, and expire. Render additions, changes,
deletions, conflicts, and active agent work with the approved visual language.

### Step 36: add Presence and live subscriptions

Keep high-frequency collaboration state outside the persistent CRDT.

Files to create:

- `app/include/fullstack_design_presence.h`
- `app/src/collab/fullstack_design_presence.c`
- `app/src/collab/fullstack_design_subscription.c`
- `examples/a2ui_design_app/design_presence_overlay.cpp`
- `test/design_presence_tests.c`
- `test/design_subscription_tests.c`

Support cursors, selections, viewports, active pages, drag previews, text
carets, agent work regions, and progress. Add expiry, rate, and payload limits.

### Step 37: add replica synchronization

Connect the convergence-tested CRDT log to MCP resources and subscriptions.

Files to create:

- `app/src/collab/fullstack_design_sync.c`
- `app/include/fullstack_design_sync.h`
- `test/design_multi_client_tests.c`
- `test/design_sync_recovery_tests.c`

Test simultaneous human and Agent moves, concurrent translation, same-chunk
Canvas edits, offline reconnect, peer removal, snapshot fallback, undo, and
deterministic conflict rendering.

## Milestone 10: export and generated C

This milestone completes the approved output formats using the same snapshot
and Canvas validation path as preview rendering.

### Step 38: add raster and locale-matrix export

Implement bounded offscreen export for PNG, WebP, and JPEG.

Files to create:

- `app/include/fullstack_design_export.h`
- `app/src/export/fullstack_export_plan.c`
- `app/src/export/fullstack_export_raster.c`
- `app/src/export/fullstack_export_locale_matrix.c`
- `app/impl/image/fullstack_libwebp_provider.c`
- `app/impl/image/fullstack_libwebp_provider.h`
- `test/design_raster_export_tests.c`
- `test/design_locale_matrix_export_tests.c`

Test scale, transparent background, output limits, cancellation, all locales,
themes, viewports, and partial failure reports.

Use the existing stb adapter for PNG and JPEG. Discover libwebp through CMake
for WebP and require one CI and packaging lane with that adapter enabled. The
generic exporter depends only on `FS_ImageEncodeProvider`.

### Step 39: add SVG and PDF export

Preserve supported semantic layers as vectors and rasterize only unsupported
subtrees.

Files to create:

- `third_party/pdfio/`
- `app/include/fullstack_pdf_provider.h`
- `app/impl/pdf/fullstack_pdfio_provider.c`
- `app/impl/pdf/fullstack_pdfio_provider.h`
- `app/src/export/fullstack_export_svg.c`
- `app/src/export/fullstack_export_pdf.c`
- `app/src/export/fullstack_export_fallback.c`
- `test/design_svg_export_tests.c`
- `test/design_pdf_export_tests.c`

Test gradients, paths, text, clipping, components, RTL text, filters, blend
modes, local raster fallback, and deterministic export reports.

Vendor PDFio and its license in a separate dependency commit before adding the
adapter. The generic PDF exporter writes through a provider interface and does
not include PDFio headers from `app/src`.

### Step 40: generate compilable FS Canvas C

Generate source, resources, locale catalogs, manifest, and CMake files from a
validated snapshot.

Files to create:

- `app/src/export/fullstack_export_c.c`
- `app/src/export/fullstack_export_c_writer.c`
- `app/src/export/fullstack_export_c_writer.h`
- `test/design_c_export_tests.cmake`
- `test/fixtures/generated_c/`

Compile generated output in CTest with no GUI, MCP, A2UI, JSON, network, or App
backend dependency. Render the generated project and compare its logical
command and image hashes with the source project.

## Milestone 11: recovery, security, and performance

This milestone validates production failure paths and the measurable limits in
the approved specification.

### Step 41: implement GPU and Surface recovery in the GUI

Preserve CPU document state while rebuilding render resources.

Files to update:

- `examples/a2ui_design_app/design_application.cpp`
- `examples/a2ui_design_app/design_workspace.cpp`
- `app/src/canvas/fullstack_canvas_resource_table.c`
- `test/design_device_loss_smoke.cmake`
- `test/design_surface_recovery_smoke.cmake`

Inject Device Lost, Surface Lost, resize, suspend, and resume. Stop acquire and
present on loss, rebuild resources, replay visible streams, and expose a
rendering-unavailable mode if recovery fails.

### Step 42: add parser and archive fuzzing

Enable fuzz targets only when `FS_BUILD_DESIGN_FUZZ=ON`.

Files to create:

- `test/fuzz/a2ui_jsonl_fuzz.c`
- `test/fuzz/mcp_jsonrpc_fuzz.c`
- `test/fuzz/fsdesign_manifest_fuzz.c`
- `test/fuzz/canvas_stream_fuzz.c`
- `test/fuzz/svg_path_fuzz.c`
- `test/fuzz/locale_catalog_fuzz.c`

Seed corpora from valid fixtures and every historical parser failure. Require
bounded memory, bounded recursion, cancellation, and no process crash.

### Step 43: add performance and idle benchmarks

Measure the approved capacity and latency targets in repeatable programs.

Files to create:

- `test/design_large_document_benchmark.c`
- `test/design_canvas_stream_benchmark.c`
- `test/design_transaction_benchmark.c`
- `test/design_idle_benchmark.cmake`
- `test/design_pan_zoom_smoke.cmake`

Cover 100,000 nodes, 1,000,000 Canvas commands across visible and cached
streams, ordinary transaction latency, snapshot publication, pan and zoom,
idle CPU and GPU work, and memory growth.

Performance tests report measurements on every run. CI can use generous
regression thresholds first, then tighten them after stable baselines exist.

### Step 44: extend architecture audits

Make the final target graph and include boundaries mechanically enforceable.

Files to update:

- `tools/architecture/check_layers.ps1`
- `test/architecture_layer_test.cmake`
- `CMakeLists.txt`

Audit forbidden source includes, root-to-App links, generic App-to-concrete
adapter links, direct third-party includes outside `app/impl`, GUI
backend-specific event code, and generated C dependencies.

Configure dedicated Core-only, App-without-design, headless-without-GUI, and
complete-GUI build matrices.

### Step 45: package the desktop editor

Create a portable package only after both backends and recovery tests pass.

Files to create:

- `tools/packaging/package_a2ui_design_app_win64.sh`
- `tools/packaging/verify_a2ui_design_app_package.ps1`
- `tools/packaging/a2ui-design-app-third-party-notices.txt`

Files to update:

- `CMakeLists.txt`

Collect wgpu-native, backend runtime libraries, optional ICU libraries,
licenses, default locale catalogs, and required assets. Verify the package in
an isolated directory with GLFW and SDL3, then create the ZIP and record its
SHA-256.

## Verification commands

Run verification from the narrowest target to the complete product after each
milestone.

Use these representative commands:

```powershell
cmake -S . -B build -G Ninja `
  -DFS_BUILD_DESIGN_RUNTIME=ON `
  -DFS_BUILD_DESIGN_HEADLESS=ON `
  -DFS_BUILD_DESIGN_GUI=ON

cmake --build build --target `
  fullstack_design_document fullstack_canvas_vm a2ui_design_headless

ctest --test-dir build --output-on-failure -R "fs_design|fs_crdt"
ctest --test-dir build --output-on-failure -R "fs_canvas|fs_a2ui"
ctest --test-dir build --output-on-failure -R "fs_mcp|fs_i18n"

$env:FS_APP_BACKEND = "glfw"
build\a2ui_design_app.exe --autotest

$env:FS_APP_BACKEND = "sdl3"
build\a2ui_design_app.exe --autotest

ctest --test-dir build --output-on-failure

powershell -NoProfile -ExecutionPolicy Bypass `
  -File tools/architecture/check_layers.ps1 -Root .

git diff --check -- <files-touched-by-the-current-task>
```

Run an unscoped `git diff --check` only after separating or documenting
pre-existing unrelated worktree failures.

Also configure these reduced builds:

```powershell
cmake -S . -B build-core-only -G Ninja `
  -DFS_BUILD_DESIGN_RUNTIME=OFF

cmake -S . -B build-headless -G Ninja `
  -DFS_BUILD_DESIGN_GUI=OFF `
  -DFS_BUILD_DESIGN_HEADLESS=ON
```

The reduced builds must not discover or link CivetWeb, ICU, GLFW, SDL3, Taffy,
or design GUI sources unless their feature requires them.

## Commit sequence

Use small commits that preserve bisectability and never include unrelated
worktree files.

1. Add build options and design ABI tests.
2. Add values, snapshots, retained nodes, and the owner-thread runtime.
3. Add transactions, CRDT primitives, tree placement, text, and history.
4. Add Canvas coverage generation, command ABI, resources, and replay.
5. Vendor A2UI and yyjson, then add adapters and fixtures.
6. Add workspace policy, persistence, archives, autosave, and i18n.
7. Add and verify the headless vertical slice.
8. Add MCP core, CivetWeb transport, and the stdio bridge.
9. Add the GUI shell and responsive panels.
10. Add selection, tools, properties, history, and accessibility.
11. Add preview branches, Presence, and replica synchronization.
12. Add exports, generated C, recovery, fuzzing, and performance tests.
13. Add packaging and third-party notices.

Commit each vendored dependency and license separately from adapters that use
it. Before every commit, inspect `git diff --cached --name-status` and stage
only the task's files.

## First implementation session

Begin with Steps 1 through 4 only. This session creates ABI types, values,
snapshots, retained nodes, and owner-thread requests without introducing any
third-party dependency.

Stop the session after these conditions pass:

- `fs_design_abi_tests`
- `fs_design_id_tests`
- `fs_design_value_tests`
- `fs_design_snapshot_tests`
- `fs_design_document_tests`
- `fs_design_runtime_tests`
- `fs_design_threading_tests`
- `fs_architecture_layers`
- Existing App and root tests affected by CMake changes

Review the public C ABI and allocator behavior before proceeding to CRDT or
Canvas commands. Changing identity or ownership types after persistence and
wire protocols exist would create unnecessary migration work.

## Next steps

After this plan is approved, begin the first implementation session at Step 1.
Do not add A2UI, JSON, MCP, GUI, or third-party code until the foundation gate
passes and its public ABI receives review.
