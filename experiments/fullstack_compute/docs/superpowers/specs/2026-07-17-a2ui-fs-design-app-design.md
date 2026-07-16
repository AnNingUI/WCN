# A2UI FS Design App architecture

This specification defines an agent-native, Figma-style design application
for Fullstack Compute. The application uses a layered C runtime, embeds its MCP
server in the GUI by default, and can run the same document and rendering
engine without a window in headless workflows.

> **Note:** This is a preview feature currently under active development.

The design keeps the root rendering SDK independent from application,
protocol, persistence, and platform concerns. It adds the design runtime to the
optional `app/` extension and places complete GUI and headless programs under
`examples/`.

## Problem statement

Fullstack Compute exposes a broad Canvas-style rendering API, but it does not
currently provide an editable design document, a professional visual editor,
or a stable protocol surface for agents. Exposing every `fs_cmd_*`, style,
path, image, font, and query function as a separate MCP tool would create a
large, fragile tool surface without atomic editing, undo, collaboration, or
document persistence.

The new application must let human designers and agents edit the same retained
document. Agents must reach the application through MCP rather than
provider-specific OpenAI, Claude, or Gemini protocols. A2UI must remain a
supported semantic interchange format without becoming the low-level mutation
or Canvas command protocol.

## Goals

The system must provide these capabilities:

- Run one reusable C design runtime in GUI and headless modes.
- Provide a responsive, Figma-style editor with pages, layers, an infinite
  canvas, properties, components, and agent activity.
- Let MCP clients read, mutate, validate, preview, and export design documents.
- Parse A2UI v0.9.1 and the v1.0 Candidate into one version-neutral IR.
- Expose every relevant FS Canvas capability through a typed, validated
  `CanvasLayer` command model.
- Support atomic transactions, multi-agent CRDT collaboration, undo, redo,
  presence, and offline convergence.
- Support App UI localization and localized design content, including RTL and
  pseudolocalization.
- Persist projects as `.fsdesign` bundles and export A2UI, raster, vector,
  PDF, and compilable FS Canvas C output.
- Preserve the existing root SDK dependency boundary.

## Non-goals

The first implementation does not need to duplicate every advanced Figma
feature. It does not initially require Boolean vector networks, multiplayer
voice or chat, browser-based hosting, third-party plugin execution, or direct
model-provider clients.

The system does not move MCP, A2UI, JSON, networking, CRDT, filesystem, or
window code into root `include/` or `src/`. It does not expose native pointers,
WGPU handles, platform window handles, or unvalidated file paths through MCP.

## Chosen architecture

The selected architecture is a layered C core that is embedded in the GUI by
default and is also available in an optional headless process. The GUI shell is
a thin C++ composition layer over the reusable C ABI.

```text
Agent or MCP host
        |
        +-- Streamable HTTP
        +-- local stdio bridge
                |
+---------------v----------------+
| FS Design App runtime, pure C   |
|                                 |
| MCP session and policy          |
| A2UI controller                 |
| Design document and CRDT        |
| Canvas VM and resources         |
| Persistence and export          |
+---------------+----------------+
                |
        FS Canvas and WGPU core
          +-----+-----+
          |           |
     GUI Surface   Headless target
```

The GUI starts the design runtime and a local MCP Session Hub in the same
process. A lightweight bridge supports hosts that can only start stdio MCP
servers. Headless mode starts the same runtime with an offscreen target and no
window or input backend.

## Directory and dependency boundaries

The implementation follows the existing root SDK and optional App split.

```text
include/ and src/
  Existing root FS Canvas and WebGPU SDK.
  Depend only on WebGPU and the language runtime.

app/include/ and app/src/
  Backend-neutral C APIs and implementations for the App runtime, design
  document, A2UI normalization, CRDT, Canvas VM, and export orchestration.
  Depend only on root public APIs, WebGPU, and the language runtime.

app/impl/
  Replaceable JSON, HTTP, stdio, filesystem, compression, image, font,
  localization, and platform implementations.

examples/a2ui_design_app/
  Complete C++ GUI shell using `examples/ui`, Taffy, and the App runtime.

examples/a2ui_design_headless/
  Complete headless executable using the same C design runtime.
```

Root `include/` and `src/` must not include or link App, A2UI, MCP, JSON,
network, window, font, image, or filesystem dependencies. `app/include/` and
`app/src/` define provider interfaces for services implemented in `app/impl/`.

## C ABI, ownership, and threading

The reusable runtime follows the root project's stable C ABI conventions.
Every exported function uses `FS_API` and `FS_CALL`. Public structures start
with `struct_size` and use fixed-width fields. Enum-like ABI values use
`uint32_t` typedefs, and long-lived objects remain opaque handles.

Creation accepts an `FS_Allocator` or inherits the owning App allocator. Memory
must be released by the module that allocated it. Input strings and byte spans
remain valid only for the duration of a call unless the API explicitly copies
them. Returned variable-size data uses caller-provided buffers or owned result
handles with matching release functions. Public APIs do not return borrowed
pointers into mutable document storage.

Each `FS_DesignRuntime` and `FS_DesignDocument` has one owner thread. Only the
owner thread commits transactions, mutates CRDT state, changes selection, or
publishes a new document snapshot. MCP transport threads parse bounded input
and enqueue immutable transaction requests. They never mutate the document or
call `FS_Core` directly.

Queries from other threads use immutable, reference-counted snapshots. Worker
completion posts a result to the owner thread, which validates that the source
revision is still applicable before publication. User callbacks, MCP
completion, and diagnostics declare their delivery thread and cannot re-enter
a document commit.

Destroy first closes the runtime to new requests, cancels workers and MCP
operations, drains owner-thread completions, releases snapshots, and then frees
the runtime through its allocator. ABI major versions reject incompatibility;
minor versions use `struct_size` and capability flags.

## Runtime modules

The runtime separates protocol adaptation, document state, drawing execution,
and external services into independently testable modules.

### `FS_A2UIController`

`FS_A2UIController` accepts A2UI v0.9.1 and v1.0 Candidate messages. It
negotiates catalogs, validates message order, normalizes messages into internal
operations, and maps document changes back to A2UI when exporting or
subscribing.

The controller never mutates Canvas or document memory directly. It creates
the same typed transactions used by the GUI and MCP tools.

### `FS_DesignDocument`

`FS_DesignDocument` owns pages, frames, layers, components, instances, slots,
assets, variables, localized messages, interactions, Canvas streams, operation
history, and CRDT state. It is the single authoritative persistent state.

Selection, cursors, viewports, drag previews, and active tool state are
ephemeral presence and remain outside the persistent document.

### `FS_CanvasVM`

`FS_CanvasVM` validates typed command streams, resolves stable resource IDs,
compiles commands into root FS Canvas API calls, and replays a `CanvasLayer`
deterministically. It owns no platform window, Surface, or present loop.

### `FS_DesignExport`

`FS_DesignExport` coordinates project, A2UI, raster, vector, PDF, localization
matrix, and generated C exports. Encoders and filesystem writes are provided
through `app/impl/` service interfaces.

### `FS_McpSessionHub`

`FS_McpSessionHub` owns authenticated MCP sessions, Actor IDs, progress,
cancellation, logging, resources, subscriptions, elicitation, and Sampling
integration. Network and stdio transports remain replaceable adapters.

## GUI workspace

The GUI uses a dominant central Canvas with subordinate editing panels. It
follows the approved Pencil design in `a2ui.pen` without depending on Pencil at
runtime.

```text
+-------------------------------------------------------+
| File and state | tools | locale | MCP | preview | share|
+-------------+--------------------------+---------------+
| Pages       |                          | Design        |
| Layers      |     Infinite Canvas      | Prototype     |
| Assets      |                          | Inspect       |
| Components  |                          +---------------+
|             |                          | Agent activity|
+-------------+--------------------------+---------------+
| Runtime, revision, locale, zoom, and diagnostics       |
+-------------------------------------------------------+
```

The top toolbar exposes document state, drawing tools, locale, MCP status,
preview, and sharing. The left panel switches between pages, layers, assets,
and components. The right panel combines context-sensitive properties with a
compact agent activity surface.

## Selection and Canvas interaction

The editor supports professional direct manipulation while committing only
meaningful document transactions.

- Click selects one node, and Shift-click adds or removes a node.
- Dragging on empty Canvas creates a marquee selection.
- Alt or Ctrl enables deep selection through nested frames.
- Double-click enters a group, frame, component, or `CanvasLayer` isolation
  mode.
- Escape exits one isolation level at a time.
- The layer tree and Canvas always share one selection model.
- Hidden, locked, read-only, and remotely edited states remain visible.
- Multi-selection exposes a shared bounding box and mixed property values.

Space-drag or middle-button drag pans the Canvas. Ctrl or Command with the
wheel zooms. Touchpads support two-finger pan and pinch zoom. The editor also
provides fit-all, fit-selection, 100 percent, rulers, guides, layout grids,
pixel grids, smart guides, distance labels, and configurable snapping.

Dragging transforms updates an ephemeral preview. Pointer release submits one
document transaction. This rule prevents high-frequency pointer movement from
polluting CRDT state and undo history.

## Creation and properties

The first complete editing milestone includes selection, frames, sections,
rectangles, ellipses, polygons, lines, paths, text, images, icons, components,
slots, `CanvasLayer` nodes, and comments.

The property panel exposes context-sensitive sections for transforms, layout,
constraints, appearance, typography, effects, components, localization,
interactions, Canvas VM state, and export. It supports mixed values, numeric
scrubbing, variables, tokens, component overrides, property search, and
copying or pasting property sets.

Frames support free, horizontal, and vertical layout. Sizing supports fixed,
fit-content, fill-container, minimum, maximum, and aspect-ratio policies.
Viewport profiles render the same document for desktop, tablet, mobile, and
custom dimensions. Locale, theme, data, and profile changes are preview inputs,
not implicit document copies.

## Responsive editor shell

The editor preserves the Canvas hierarchy as its own window becomes smaller.

| Window width | Editor behavior |
| ---: | --- |
| At least 1200 px | Show the complete three-column workspace. |
| 900 to 1199 px | Collapse the right panel into an on-demand surface. |
| 640 to 899 px | Show one side panel at a time over the Canvas. |
| Less than 640 px | Use touch review mode with bottom sheets. |

Touch review mode retains selection, basic properties, comments, agent
requests, locale previews, and approval flows. Complex path-node editing
requires tablet or desktop space.

## Keyboard and accessibility

The editor follows familiar design-tool shortcuts while keeping all operations
available through the command palette.

```text
V  Select
F  Frame
R  Rectangle
O  Ellipse
P  Pen
T  Text
C  Comment
K  CanvasLayer
Ctrl or Command + K  Command palette
Ctrl or Command + Z  Undo
Shift + Ctrl or Command + Z  Redo
```

The GUI maintains a semantic accessibility tree that mirrors pages, tools,
selected nodes, properties, dialogs, errors, and agent status. It provides
visible focus, keyboard navigation, configurable shortcuts, high-contrast
themes, and a reduced-motion mode.

## Agent editing experience

Agents work through MCP and use the current selection, page, viewport, locale,
theme, component context, design variables, recent history, and diagnostics as
context. The application does not embed provider-specific chat protocols.

Small deterministic changes can commit immediately and remain undoable. Large
or broad changes create a preview branch. The Canvas renders additions in
green, modifications in blue, deletions in red, conflicts in amber, and active
agent work with a moving dashed outline.

The user can accept, reject, or refine a preview branch. Accepting it creates
one normal CRDT transaction. Privileged file, font, URL, pixel, and export
operations always follow the workspace permission policy.

When an MCP client supports `sampling/createMessage`, the App prompt surface
can request model output through MCP Sampling. If Sampling is unavailable, an
external host remains in control and the GUI offers context-copying and
activity feedback.

## MCP tool surface

The MCP server exposes a small set of coarse tools instead of one tool per
Canvas function. Typed batch operations provide atomicity and keep the model
context manageable.

Mutation tools include:

- `batch_design`
- `batch_canvas`
- `set_variables`
- `set_localized_messages`

Read-only and validation tools include:

- `get_editor_state`
- `batch_get`
- `inspect_canvas_layer`
- `snapshot_layout`
- `hit_test`
- `measure_text`
- `render_preview`
- `get_translation_status`

Privileged tools include project import, file access, external font loading,
image decoding, URL retrieval, pixel readback, and export writes. Each tool
returns structured errors, progress, cancellation state, and the resulting
document revision or state vector.

MCP uses the official 2025-11-25 protocol. Streamable HTTP is the default
multi-client transport. A lightweight stdio bridge connects stdio-only hosts
to the embedded Session Hub.

## A2UI versions and catalog

The controller accepts A2UI v0.9.1 and the v1.0 Candidate. It normalizes both
versions into an internal IR so document code does not branch on wire version.

The initial compatibility profile is pinned to the authoritative
`a2ui-project/a2ui` repository commit
`0190314c56eb136bb2b1541d8385d18c1131b9fe`, observed on July 17, 2026.
Implementations vendor the v0.9.1 and v1.0 schema trees from that commit under
`third_party/a2ui/specification/`. The adapter and a manifest containing each
schema file's SHA-256 hash live under `app/impl/a2ui/`. Builds and tests never
consume a floating `main` branch or fetch schemas at runtime. The vendored tree
is marked in `.gitattributes` with the same policy as other root third-party
sources.

Each adapter version has normative fixtures for message ordering, validation
order, unknown fields, unsupported components, default values, action mapping,
and round-trip loss. Supporting a later Candidate revision requires a new
named compatibility profile and regression fixtures; it does not silently
change the existing `1.0-candidate-2026-07-17` profile.

The application defines this independently versioned custom catalog:

```text
Catalog URI: urn:wcn:fs-design-catalog
Catalog version: 1.0
```

The first catalog contains `Surface`, `Frame`, `Group`, `Rectangle`,
`Ellipse`, `Path`, `Text`, `LocalizedText`, `Image`, `Icon`, `Component`,
`ComponentInstance`, `Slot`, and `CanvasLayer`.

The A2UI adapter applies these mappings:

- `createSurface` creates a page or independent design Surface.
- `updateComponents` creates typed semantic document operations.
- `updateDataModel` updates variables, bindings, data, and localization
  arguments.
- `deleteSurface` removes the corresponding design Surface.
- A2UI v1 `actionId` maps to `FS_InteractionBinding`.
- Unsupported optional components remain opaque and round-trip safely.
- Unsupported required components reject the transaction.

Catalog negotiation reports supported A2UI versions, catalog versions, Canvas
command versions, and optional features such as CRDT, RTL, headless rendering,
and `CanvasLayer` support.

## Protocol separation

MCP is the agent control plane, the Design Document is the editable authority,
A2UI is a semantic interchange protocol, and the Canvas VM is the drawing
execution layer.

A2UI messages cannot call `fs_cmd_*` directly. A `CanvasLayer` references a
Canvas stream by stable ID. The command stream is transferred through a typed
MCP transaction or project resource rather than embedded as an unbounded A2UI
component property.

## Design document model

The document stores semantic design state and references immutable binary or
content-addressed resources.

```text
FS_DesignDocument
+-- metadata
+-- pages
+-- nodes
+-- component definitions and instances
+-- variables and themes
+-- locale catalogs
+-- resources
+-- Canvas streams
+-- operation log
+-- snapshots
```

Every persistent entity has a globally unique `FS_EntityId`. Serialized IDs
use a compact Base32 representation. Deleted IDs are never reused.

## CRDT identity and causality

Every operation has an Actor ID and monotonically increasing Actor-local
counter.

```c
typedef struct FS_OpId {
    FS_ActorId actor;
    uint64_t counter;
} FS_OpId;
```

Transactions carry a causal state vector, a hybrid logical clock timestamp,
the Actor ID, a permission context, and the source transaction ID. The state
vector distinguishes causal ordering from concurrency. The HLC provides stable
display ordering.

All replicas use one total winner order: HLC physical time, HLC logical time,
Actor ID bytes, and Actor-local counter. The first unequal field decides. This
order resolves LWW registers, the rendered winner of multi-value registers,
Canvas conflict branches, and cycle-recovery candidates. Multi-value registers
still retain every concurrent value even though rendering selects one winner.
If every operation field is equal, the canonical value hash is the final
defensive tie-break; equal Op IDs with different payloads are a protocol error.

## Domain-specific CRDTs

The document uses a CRDT suited to each data domain.

| Data | CRDT |
| --- | --- |
| Node existence | Remove-wins observed-remove map |
| Ordinary scalar properties | LWW register |
| High-value content properties | Multi-value register |
| Tree placement | LWW placement register with parent and LSEQ position |
| Collaborative text | Unicode grapheme sequence |
| Canvas commands | Immutable chunk sequence |
| Variables and localized messages | OR-map with typed registers |

Deletion wins against concurrent updates that did not observe the deletion.
This rule prevents a remote property update from unintentionally reviving a
deleted node. Restore is an explicit operation.

Colors, dimensions, opacity, and similar scalar properties use LWW. Text,
component references, bindings, and Canvas stream references preserve
concurrent values in a multi-value register. The renderer uses a deterministic
winner while the inspector exposes every conflicting value.

## Tree moves and ordering

Parent and sibling position form one atomic `FS_TreePlacement` value containing
`parent_id`, a parent-scoped LSEQ position, and the move Op ID. A concurrent
move selects the complete placement from one operation; it cannot combine the
parent from one move with the position from another. Losing concurrent
placements remain available as conflict values.

LSEQ positions are valid only in their recorded parent. When a placement wins,
its parent and position always move together. If a malformed or migrated
position names another parent, validation rejects it before commit.

Concurrent moves can form a cycle. Replicas consider placement candidates in
the shared total winner order. A candidate is accepted only when adding its
edge preserves a forest. If the winning candidate creates a cycle, resolution
tries the next retained placement for that node. A node with no valid
candidate enters the page's **Recovered Layers** group at a deterministic LSEQ
position derived from its Entity ID. The inspector preserves every suppressed
placement and reports the conflicting operations.

## Collaborative text

Collaborative text uses Unicode grapheme clusters rather than bytes or code
points. The sequence must not split emoji, combining marks, or complex scripts.
Each locale owns an independent text sequence.

Agent whole-message replacements remain atomic. A replacement can include an
expected property version so automatic translation never overwrites a newer
manual edit silently.

Unicode grapheme segmentation is a provider interface. `app/src` owns the
sequence CRDT and calls a versioned segmenter supplied by `app/impl/`. Project
metadata records the Unicode data version used to create sequence boundaries.
A migration must resegment explicitly rather than changing boundaries during
ordinary loading.

## Canvas stream collaboration

Canvas streams are sequences of immutable command chunks rather than one CRDT
entry per byte or command. Each chunk has a content hash and can be reused
across revisions and component instances.

Concurrent changes to distinct chunks merge automatically. Concurrent changes
to the same logical range preserve both branches. Rendering uses a
deterministic winner, and the inspector offers compare, merge, and restore
actions.

## Transactions

All mutation sources submit the same typed transaction format. A transaction
supports `merge` and `strict` modes.

- `merge` applies CRDT semantics and is the normal collaborative mode.
- `strict` rejects the entire transaction when a declared precondition no
  longer matches.

Transactions can run as `dryRun`. A dry run returns validation errors,
required permissions, conflicts, layout changes, dirty regions, and Canvas VM
diagnostics without committing document state.

The commit pipeline is:

```text
parse
-> schema validation
-> permission validation
-> resource resolution
-> command validation
-> CRDT operation generation
-> atomic commit
-> dirty marking
-> layout and Canvas compilation
-> repaint request
```

Failure before commit leaves the document unchanged.

## Undo and redo

Undo does not restore a global snapshot because that would erase remote work.
Undo emits a new compensating transaction that removes or reverses only the
selected Actor transaction's contribution.

Property undo removes or compensates the transaction's register version. Move
undo restores the prior atomic placement. Canvas undo restores the prior chunk
sequence. Redo emits the forward intent again with new operation IDs.

Insertion undo is dependency-aware. If no later operation from another Actor
depends on the node, undo removes the insertion tag and the node becomes
deleted. If remote children, bindings, references, component instances, or
property edits depend on it, automatic undo must not erase them. It creates an
undo conflict, removes only safe local property contributions, and keeps the
node under **Recovered Layers** with provenance that explains why it remains.
Deleting the node and its remote dependents requires a separate explicit
destructive transaction.

Delete undo emits Restore only for the deleted identity generation. Dangling
references remain diagnostics until their target is restored or replaced.
Every compensating operation records `undo_of`, and tests verify that undo does
not remove causally later remote contributions.

## Presence

Presence is an independent, ephemeral channel for cursors, selections,
viewports, active pages, drag previews, text carets, agent work regions, and
agent progress. Presence is not persisted, exported, or included in undo.

Drag previews remain in Presence until pointer release. Agents can announce a
work region so the GUI can warn about likely human-agent overlap without
creating a hard lock.

## `CanvasLayer` model

`CanvasLayer` is a retained document node. It participates in layout,
transforms, clipping, opacity, ordering, selection, components, CRDT, undo,
localization bindings, and hit testing.

```json
{
  "type": "canvasLayer",
  "id": "layer-signal-field",
  "width": 330,
  "height": 330,
  "stream": "canvas-stream-42",
  "resources": ["gradient-1", "path-7", "font-ui"],
  "bindings": {
    "accent": "$color.accent",
    "label": "$i18n.hero.title"
  },
  "timePolicy": "explicit",
  "hitTest": "painted"
}
```

A `CanvasLayer` can bake a static snapshot without losing the source stream.
It can also expand supported paths, shapes, and text into ordinary layers.
Effects that cannot round-trip remain inside a nested `CanvasLayer` with an
export diagnostic.

## Canvas command representation

The logical command model is independent from its wire, memory, and project
encodings. MCP uses typed JSON, the runtime uses C structures, and `.fsdesign`
can store a compact binary stream.

```c
typedef struct FS_CanvasCommandHeader {
    uint16_t opcode;
    uint16_t flags;
    uint32_t payload_size;
} FS_CanvasCommandHeader;
```

Opcodes cover these groups:

- State save and restore.
- Translation, rotation, scale, transform, set-transform, and reset-transform.
- Fill, stroke, alpha, composition, line, shadow, filter, and text styles.
- Path begin, close, line, quadratic, cubic, arc, ellipse, rectangle, rounded
  rectangle, and Path2D composition.
- Clear, fill, stroke, clip, rectangle, path, and shape paint operations.
- Fill text, stroke text, and localized text.
- Image draw variants and validated pixel-buffer drawing.
- Gradient, pattern, image, font, Path2D, and pixel resource binding.
- Layer, opacity, blend, filter, and effect composition.

The VM maps these operations to the current public FS Canvas functions,
including `fs_cmd_*`, style APIs, Path2D APIs, gradients, patterns, images,
fonts, transforms, clipping, effects, and filters.

Window creation, Surface acquisition, submission, presentation, and the close
loop are not Canvas opcodes. They remain App and backend responsibilities.

## Canvas API coverage manifest

Canvas coverage is defined by a normative generated manifest rather than the
phrase "relevant Canvas API." The implementation generates
`app/src/canvas_api_coverage.json` from public declarations in
`include/fullstack_core.h` at the source revision recorded by the project.

Every public declaration must have exactly one classification:

- `opcode`, with a command version and opcode ID.
- `query`, with the corresponding read-only runtime operation.
- `resource`, with creation, ownership, and release mapping.
- `runtime-excluded`, with a reason such as Core lifecycle, encoding,
  submission, presentation, backend injection, or diagnostics.

The opcode registry never reuses a numeric ID. Each entry defines payload
layout, required feature bits, validation rules, and the first command-stream
version that supports it. CI regenerates the declaration inventory and fails
when a public API is missing, duplicated, or changes classification without an
explicit manifest update. Coverage tests instantiate every opcode and query
mapping against a Mock Core or deterministic offscreen target.

## Queries and mutations

Queries do not enter the deterministic command stream. Text measurement,
pixel readback, Canvas pixel readback, path hit testing, and diagnostics are
RPC-style queries.

Pixel writes use a validated `PixelBuffer` resource. File decoding, image
loading, font loading, URL access, and pixel readback require policy checks and
cannot occur as an implicit opcode side effect.

## Resource IDs and runtime handles

The protocol uses stable resource IDs rather than native pointers or WGPU
handles.

```text
image:sha256:<hash>
font:project/inter-regular
gradient:hero-accent
path:logo-outline
```

The VM resolves stable IDs to compact, generation-checked handles.

```c
typedef struct FS_CanvasResourceHandle {
    uint32_t slot;
    uint16_t generation;
    uint16_t type;
} FS_CanvasResourceHandle;
```

Generation checking prevents use-after-delete when a slot is reused. Missing
or stale resources produce a node diagnostic and a safe placeholder rather
than undefined behavior.

## Canvas validation and determinism

The VM validates every stream before replay. Validation includes finite
numbers, balanced state stacks, bounded clip depth, bounded path complexity,
bounded command and text sizes, valid resource type and generation, and valid
opcode payload length.

Command streams cannot read files, network state, pointers, or system time.
Animation time, pointer state, viewport metrics, data, locale, and theme arrive
through an explicit frame context. The same document, resources, and frame
context must produce the same logical render output.

Unknown required opcodes reject a stream. Unknown optional extension opcodes
can be skipped with a diagnostic.

## Internationalization

Internationalization covers both the editor UI and design content. The editor
uses stable message IDs and can switch locale without restarting. Design text
can be literal or reference a localized message.

```json
{
  "content": {
    "type": "localized",
    "key": "hero.primary_action",
    "source": "Start building",
    "args": {}
  }
}
```

Locale fallback follows exact locale, parent locale, project default, and
source text. For example, `zh-Hans-CN` falls back through `zh-Hans`, `zh`, the
project default, and the source value.

The localization system supports plural and formatting arguments, LTR, RTL,
font overrides, text expansion, accented pseudolocales, and mirrored RTL
previews. The formatter is a provider interface in `app/src`; CLDR and message
format implementations belong in `app/impl/`.

The GUI exposes locale, direction, translation state, source locale, message
key, completion, and missing-glyph diagnostics. Locale Matrix renders multiple
locales side by side and reports overflow, clipping, missing translations,
missing glyphs, unsupported plural branches, unmirrored icons, and hard-coded
text.

## Workspace security

Every external capability is classified as normal, workspace-scoped, or
approval-required. Deterministic document edits and harmless queries are
normal. Workspace files, project writes, local imports, and project-relative
resources require workspace capability. External paths, URLs, system fonts,
pixel readback, file overwrite, clipboard-sensitive data, and external process
launch require explicit approval.

All paths pass through `FS_WorkspacePolicy`. The implementation canonicalizes
the path, resolves relative segments, follows symlinks or Windows Junctions,
checks the final real path, matches the capability, and only then performs the
operation.

The policy must reject traversal, Junction escape, case-folding bypass, UNC or
device path bypass, disguised absolute paths, resource collisions, and unsafe
export overwrite.

## Archive security

Single-file `.fsdesign` containers are untrusted archives. Import validates the
complete central directory before extracting or exposing any entry.

The importer rejects absolute names, traversal segments, alternate path
separators that escape normalization, drive or UNC prefixes, duplicate
canonical entry names, symlink and hard-link entries, encrypted entries, and
unsupported compression methods. It enforces limits for entry count,
individual expanded size, total expanded size, nesting, metadata size, and
compression ratio.

Entries stream into a private temporary directory or bounded memory store.
Every resource hash is verified before publication. Failure removes partial
temporary data and leaves the destination project unchanged. Archive contents
are never executed, and project-relative links are resolved only after the
workspace policy approves the final extracted path.

## MCP transport security

Streamable HTTP binds to loopback by default. Each session uses an
unpredictable token, an Actor ID, an independent policy context, request size
and JSON-depth limits, cancellation, timeout, and rate controls.

The HTTP adapter validates `Origin`. Remote listening requires explicit TLS,
authentication, and an Origin allowlist. The stdio bridge connects only to the
local Session Hub by default.

## Structured errors

No exception crosses the C ABI. Public operations return the repository's
existing `FS_Result` and optional `FS_Error`. The design extension adds stable
error-domain constants for Document, CRDT, A2UI, Canvas VM, Resource, Layout,
Text, i18n, MCP, Filesystem, Export, and Security without defining a second
error ABI.

Entity IDs, operation IDs, recovery hints, JSON Pointers, and Canvas opcode
locations travel in an owned `FS_DesignDiagnosticContext` attached to an
operation result or emitted through `FS_DiagnosticSink`. Its lifetime is
explicit and it has a matching release function. The GUI displays the error on
the affected node, in the property panel, in the status bar, and in agent
activity. MCP returns the same structured information.

## Failure recovery

Transaction failure rolls back the complete transaction and identifies the
failing operation, JSON Pointer, node, or opcode. An MCP disconnect does not
stop local editing. Reconnection resumes from the authenticated session state
vector and never repeats a confirmed transaction.

On GPU device loss, the App stops acquisition, preserves the CPU document,
rebuilds the device, Queue, Surfaces, and cached resources, and replays visible
`CanvasLayer` streams. If GPU recovery fails, the editor remains open in a
CPU-editable, rendering-unavailable state.

Surface loss or staleness reconfigures only the affected window. Project
corruption recovery loads the last valid snapshot, replays valid operations,
quarantines corrupt resources, and writes a recovery report without replacing
the original project.

## `.fsdesign` bundle

The project format is a logical bundle that can remain an unpacked directory
for Git workflows or use the same contents in a ZIP container for
distribution.

```text
project.fsdesign/
+-- manifest.json
+-- document.json
+-- history/
|   +-- snapshot.fsdoc
|   +-- operations.fslog
+-- canvas/
|   +-- <stream-id>.fscmd
+-- resources/
|   +-- sha256/<content-hash>
+-- locales/
|   +-- en-US.json
|   +-- zh-Hans.json
|   +-- ar-SA.json
+-- previews/
|   +-- thumbnail.webp
+-- recovery/
    +-- autosave.fslog
```

JSON uses UTF-8 and canonical key ordering. Binary blocks include version,
length, byte order, and content hash. Saving preserves unknown extension
fields. Project, document IR, Canvas stream, A2UI catalog, A2UI adapter, and
MCP tool versions evolve independently.

Saving writes new data to temporary paths, flushes it, and atomically replaces
the manifest or target file. The writer never overwrites the only valid
snapshot in place.

## Autosave and migration

Autosave starts when a transaction makes the document dirty. It appends an
incremental journal, debounces resource work, writes changed resources, builds
a background snapshot, and atomically publishes the new manifest.

After an abnormal exit, the GUI offers to restore autosave, open the last
formal save, compare both states, or save a recovery copy.

Migrations are deterministic pure transformations. They do not modify the
source project, preserve unknown extensions, support dry runs, create a backup,
and emit a migration report.

## Log compaction and tombstones

Snapshots carry the complete CRDT state vector. An operation or tombstone can
be compacted only after every replica in the document's retained peer set has
acknowledged a state vector that observes it. Removing a peer from that set is
an explicit administrative operation recorded in history.

Compaction writes a new snapshot and operation log, verifies their canonical
hashes, and atomically publishes them. Offline replicas older than the retained
frontier must resynchronize from a full snapshot instead of replaying an
incomplete log. Remove-wins tombstones, undone operation provenance, and
resource generations remain until this acknowledgement rule permits removal.

## Export formats

The export subsystem supports `.fsdesign`, A2UI v0.9.1 JSONL, A2UI v1.0
Candidate JSONL, FS Catalog extensions, normalized debugging JSON, PNG, WebP,
JPEG, SVG, and PDF.

SVG and PDF keep supported semantic layers as vectors. A `CanvasLayer` effect
that cannot map to the target format rasterizes only the affected subtree and
records the reason in the export report.

Localization export supports one locale, every locale, locale by theme,
locale by viewport, and the complete locale by theme by viewport matrix.

## Generated FS Canvas C output

The C exporter produces a standalone project that depends only on root FS
Canvas APIs and WebGPU.

```text
generated/
+-- design.c
+-- design.h
+-- design_resources.c
+-- design_i18n.c
+-- assets/
+-- manifest.json
+-- CMakeLists.txt
```

Generated code does not depend on GUI, MCP, A2UI, JSON, networking, or a
specific App backend.

## Headless CLI

The headless program manipulates designs and serves MCP without embedding any
model-provider client.

```text
fs-design-headless validate project.fsdesign
fs-design-headless apply-a2ui input.jsonl
fs-design-headless render --locale zh-Hans --theme dark
fs-design-headless render --all-locales
fs-design-headless export --format pdf
fs-design-headless generate-c
fs-design-headless serve-mcp
```

GUI and headless processes call the same C runtime and produce equivalent
document hashes and logical render commands.

## Rendering and frame scheduling

The GUI uses the App extension's on-demand frame scheduler. Idle windows do not
redraw. Canvas, panel, overlay, and agent-activity domains maintain independent
dirty flags. Animations explicitly request another frame through a
`requestAnimationFrame`-style App API.

The renderer culls offscreen nodes, reuses compiled `CanvasLayer` results,
shares immutable resources, and reuses intermediate textures during pan and
zoom where safe. Agent parsing, CRDT merge, filesystem work, image decoding,
snapshot generation, layout preparation, and logical Canvas validation can run
away from the UI thread and publish complete results atomically. Replay into an
`FS_Core`, GPU resource creation, encoding, submission, and presentation remain
on the App or render thread required by the owning GPU context.

## Verification strategy

Verification combines deterministic unit tests, CRDT property tests, protocol
fuzzing, image baselines, backend integration tests, recovery tests, and
performance benchmarks.

### Unit tests

Unit tests cover the Design Document, typed properties, A2UI normalization,
Canvas opcode validation, resource generations, i18n fallback, workspace path
policy, compensating undo, project migration, and export planning.

### CRDT property tests

Property tests reorder, duplicate, delay, and merge operations across offline
and reconnecting replicas. Every replica must converge on the same document
hash, layer tree, variables, locale catalogs, Canvas streams, and logical
render command hash.

The suite includes concurrent delete and update, concurrent cross-page move,
cycle creation, collaborative text, same-chunk Canvas edits, and undo when
remote operations depend on local state.

### Fuzz tests

Fuzz targets include A2UI JSONL, MCP JSON-RPC, `.fsdesign` manifests, Canvas
binary streams, SVG paths, image and font metadata, locale catalogs, and deeply
nested or malicious document trees.

### Image regression tests

Image tests cover the editor shell and representative designs across DPI,
GLFW, SDL3, light and dark themes, LTR, RTL, pseudolocales, desktop, tablet,
mobile profiles, filters, and blending.

GPU verification combines an exact logical command hash with a tolerant image
comparison so small backend floating-point differences do not create false
failures.

### Integration and recovery tests

Integration tests cover embedded MCP, the stdio bridge, Streamable HTTP session
recovery, GUI and headless project equivalence, device-loss recovery, Surface
rebuild, autosave crash recovery, multiple windows, backend switching,
asynchronous large-file loading, and cancellation.

### Performance targets

The initial implementation targets these measurable workloads:

- 100,000 retained design nodes.
- 1,000,000 Canvas commands across visible and cached streams.
- 60 frames per second during ordinary pan and zoom.
- Less than 16 ms input feedback latency.
- Less than 8 ms for an ordinary document transaction before asynchronous
  layout or rendering work.
- No UI-thread blocking for background snapshots.
- Near-zero CPU and GPU work for an unchanged idle window.

## Architecture audits

Automated audits enforce dependency direction. Root `include/` and `src/` must
not include App, MCP, A2UI, JSON, networking, platform, font, image, or
filesystem headers. Generic `app/include/` and `app/src/` must not include GLFW,
SDL3, Win32, Cocoa, Android, UIKit, or a concrete JSON or HTTP library.

Adapters in `app/impl/` may depend on the generic App API, root APIs, root
adapters, and their selected third-party or platform libraries. Complete GUI
and headless executables remain in `examples/`.

The audit also inspects the configured CMake target graph. It must fail when a
root target links any `app/` target, when generic App targets link a concrete
backend or third-party implementation, or when an adapter dependency points
against the allowed direction. `tools/architecture/check_layers.ps1` or its
successor checks both source includes and generated target dependencies in
every maintained build configuration.

## Implementation phases

The implementation proceeds in dependency order so each phase produces a
testable artifact.

1. Add C document IDs, typed values, transactions, operation logs, and a Mock
   design service layer.
2. Add A2UI normalization, the FS Catalog, and deterministic protocol tests.
3. Add `CanvasLayer`, resource IDs, Canvas command validation, and headless
   replay.
4. Add project persistence, autosave, migration, and basic export.
5. Add MCP Session Hub, Streamable HTTP, and the stdio bridge.
6. Add the desktop GUI shell, selection, transforms, properties, and history.
7. Add components, variables, localization, responsive profiles, and agent
   preview branches.
8. Add domain-specific CRDT collaboration, Presence, offline recovery, and
   compaction.
9. Complete vector, PDF, generated C, locale matrix, security, recovery, and
   performance verification.

## Completion criteria

The design application is complete when these conditions hold:

- GUI and headless modes use one C design runtime.
- MCP clients can read, mutate, validate, preview, and export documents without
  provider-specific protocols.
- A2UI v0.9.1 and v1.0 Candidate normalize into one internal IR.
- `CanvasLayer` covers the complete relevant root Canvas API through validated
  typed commands.
- Transactions are atomic, undo is compensating, and replicas converge under
  the defined CRDT rules.
- The GUI provides the approved Figma-style editing, responsive shell, agent
  preview, and i18n experience.
- `.fsdesign`, A2UI, raster, vector, PDF, localization, and generated C exports
  pass round-trip and regression tests.
- Device loss, Surface loss, MCP reconnect, autosave recovery, and corrupt
  projects produce explicit recoverable states rather than blank output or a
  process crash.
- Root `include/` and `src/` remain independent from App and all new protocol,
  document, platform, and third-party dependencies.

## Next steps

After this specification passes review and receives final approval, create a
detailed implementation plan with file-level tasks, test-first milestones,
dependency additions, migration checkpoints, and verification commands. The
`writing-plans` skill is not available in the current environment, so the plan
must follow the same structure manually.
