# Canvas API Coverage (fullstack_compute)

## Implemented command primitives (already wired to GPU pipeline)
- rect / roundRect-like fill: `fs_cmd_rect` (`FS_CMD_RECT`)
- rect stroke: `fs_cmd_rect_stroke` (`FS_CMD_RECT_STROKE`)
- image draw: `fs_cmd_image`, `fs_cmd_image_handle` (`FS_CMD_IMAGE`)
- text draw: `fs_cmd_text_glyph`, `fs_cmd_text_utf8` (`FS_CMD_TEXT`)
- line segment: `fs_cmd_line` (`FS_CMD_LINE`)
- path segment (explicit segment): `fs_cmd_path_segment` (`FS_CMD_PATH_SEGMENT`)
- circle: `fs_cmd_circle` (`FS_CMD_CIRCLE`)
- arc stroke primitive: `fs_cmd_arc` (`FS_CMD_ARC`)
- quadratic bezier stroke: `fs_cmd_bezier_quad` (`FS_CMD_BEZIER_QUAD`)
- cubic bezier stroke: `fs_cmd_bezier_cubic` (`FS_CMD_BEZIER_CUBIC`)
- ellipse fill: `fs_cmd_ellipse` (`FS_CMD_ELLIPSE`)

## Public API layout
- UPDATE: all public headers are now under `experiments/fullstack_compute/include/`.
- UPDATE: `fullstack_core.h` is now public-only API surface; `FS_Core` is opaque.
- UPDATE: clip diagnostics/tuning APIs were split into `fullstack_core_debug.h` to keep core API noise low.
- UPDATE: context-wide state reset API is now available via `fs_context_reset`.

## Canvas 2D API gap status

### Path state / retained path API
- PARTIAL: `beginPath/moveTo/lineTo/quadraticCurveTo/bezierCurveTo/closePath/stroke` is available through
  `fs_path_begin/fs_path_move_to/fs_path_line_to/fs_path_quadratic_curve_to/fs_path_bezier_curve_to/fs_path_close/fs_path_stroke`.
- PARTIAL: path `fill` is available through `fs_path_fill` with GPU clip-mask route:
  path -> `fs_clip_path` job -> clip compute writes mask -> fullscreen fillRect constrained by path mask.
- UPDATE: `fs_path_fill` is now strict GPU-only: SDF-based clip-mask solve in clip compute for edge AA.
- UPDATE: fill-rule state added via `fs_style_set_fill_rule` (`FS_FILL_RULE_NONZERO` / `FS_FILL_RULE_EVENODD`).
- Current fill limitation:
  - path fill depends on clip-mask layer budget (`FS_CLIP_MASK_LAYERS`) and clip job capacity.
  - if GPU clip path creation fails, `fs_path_fill` now returns `false` (no CPU recovery path).
- UPDATE: clip failure diagnostics/introspection is now available:
  - enum + snapshot API:
    - `FS_ClipFailureReason`
    - `FS_ClipDiagnostics`
    - `fs_core_get_clip_diagnostics`
  - per-frame counters:
    - clip requests / cache hits / jobs enqueued / failures / layers used
  - last failure payload:
    - reason + path segment count + generated edge count
  - current reason classes:
    - `INVALID_INPUT`, `LAYER_EXHAUSTED`, `EMPTY_PATH`, `INVALID_BOUNDS`, `EDGE_ALLOC`, `JOB_ALLOC`
- UPDATE: clip layer capacity policy now includes safe in-frame reuse:
  - when `clip_mask_next_layer` is exhausted, allocator now attempts to reuse an unprotected layer
    (not referenced by emitted commands, current clip stack, or saved state stack chains).
  - victim choice now prefers least-recently-used protected-safe layer (LRU among reusable candidates).
  - near-capacity threshold policy: allocator can prefer reuse before hard exhaustion to reduce tail-risk.
  - overwritten layer hash/parent metadata is invalidated before reuse.
  - diagnostics exposes `layer_reuses_this_frame` to observe policy effectiveness.
  - tuning API:
    - `fs_core_set_clip_layer_reuse_reserve`
    - `fs_core_get_clip_layer_reuse_reserve`
- UPDATE: clip mask layer capacity has been raised from 16 to 64 in the experiment path
  to reduce stress-scene layer exhaustion artifacts during heavy nested clip playback/scroll.
- PARTIAL: clip baseline is available as axis-aligned rect clip via `fs_clip_rect` with `save/restore` scoping.
  (fragment-stage uniform clip test, applies to all primitives/text/image).
- UPDATE: clip state is baked per command at submit time; clip changes after command emission do not retroactively mutate prior commands.
- UPDATE: `fs_clip_path` now flattens path contour to edge lists and submits clip jobs to GPU compute (`FS_CLIP_MASK_WGSL`) to write clip-mask texture layers.
- UPDATE: per-call clip fill-rule API is available via:
  - `fs_clip_path_with_fill_rule`
  - `fs_clip_path2d_with_fill_rule`
  - default wrappers (`fs_clip_path` / `fs_clip_path2d`) remain style-driven.
- UPDATE: `save/restore` now snapshots clip state together with transform state.
- UPDATE: path helpers `arcTo`, `rect`, `roundRect` are now available through
  `fs_path_arc_to`, `fs_path_rect`, `fs_path_round_rect`.
- UPDATE: path helpers `arc` and `ellipse` are now available through
  `fs_path_arc`, `fs_path_ellipse`, `fs_path2d_arc`, `fs_path2d_ellipse`.
- UPDATE: hit-test APIs are now available (current path + `Path2D`):
  - fill hit-test:
    - `fs_is_point_in_path`
    - `fs_is_point_in_path_with_fill_rule`
    - `fs_is_point_in_path2d`
    - `fs_is_point_in_path2d_with_fill_rule`
  - stroke hit-test:
    - `fs_is_point_in_stroke`
    - `fs_is_point_in_stroke_path2d`
  - current implementation notes:
    - fill hit-test uses transformed-edge winding / evenodd solve with implicit subpath close.
    - stroke hit-test follows current line style state (`lineWidth/lineCap/lineJoin/miterLimit/dash`) with curve flatten approximation.
- UPDATE: clip masking is now compute-native (no CPU per-pixel fill loop and no per-clip texture upload path).
- UPDATE: clip dispatch scheduling now uses size-bucket multi-dispatch (small/medium/large clear-region buckets).
  - implementation strategy: one frame upload of ordered jobs + per-bucket bind-group range slicing (no per-dispatch uniform mutation).
  - this keeps correctness while reducing large sparse dispatch waste from single max-extent scheduling.
  - diagnostics now exposes per-frame dispatch telemetry:
    - dispatch batch count, valid job count
    - ideal clear pixels vs estimated dispatched pixels
    - dispatch waste pixels/ratio
    - per-bucket job distribution (6 buckets)
    - oriented-quad command count and oriented-quad-with-clip count
  - demo stress panel adds automatic Gate-C status line (`PASS/WARN/FAIL`) from failure count + dispatch waste ratio thresholds.
  - Gate-C now also exposes rolling-window status (`win N frames`) to smooth single-frame noise and judge sustained stability.
  - Gate-C rolling window now tracks oriented-quad clip coverage; sustained `oq>0 && oqClip==0` is downgraded to `WARN`.
- UPDATE: nested path clips use layer-parent chain sampling in fragment stage (current layer plus ancestor layers), preserving stacked clip semantics.
- UPDATE: fragment-stage path clip now includes per-layer bounds reject (AABB from clip compute output) before clip-mask texture sampling.
  - implementation detail: binding 7 now carries parent chain + per-layer min/max bounds in one uniform block.
  - goal: cut invalid clip texture fetches for out-of-bounds fragments while preserving exact clip semantics.
- UPDATE: command expansion compute stage now also intersects command AABB with clip-layer parent-chain bounds.
  - this reduces generated raster quad area before fragment stage, lowering downstream clip sampling pressure.
- UPDATE: `clip(path)` now has strict rect fast-path for axis-aligned rectangle line-paths under affine transform.
  - path requirements: single closed 4-line contour, axis-aligned edges, transformed edges still axis-aligned.
  - execution path: skip clip-mask job enqueue and directly intersect current clip rect (`fs_clip_intersect_aabb`).
- UPDATE: `clip(path)` now has analytic roundRect fast-path (compute clip job without edge list flattening).
  - path requirements: strict `line/cubic` round-rect contour shape with axis-aligned transform and near-uniform scale.
  - execution path: enqueue `CLIP_FILL_MODE_ROUND_RECT` job using `fill rect + radius`, skip winding-edge generation.
- UPDATE: `fs_clip_path` now has layer-hash cache (path+transform+fillRule+parent-clip hash), identical clips can reuse existing mask layer without reraster/upload.
- UPDATE: resize now drops only stale pending uploads referencing the old clip-mask texture (prevents dead-texture copy panic while preserving other uploads).
- UPDATE: retained `Path2D` object model is now available:
  - build/edit lifecycle: `fs_path2d_create/fs_path2d_reset/fs_path2d_destroy`
  - path commands: `fs_path2d_move_to/line_to/quadratic_curve_to/bezier_curve_to/arc/ellipse/arc_to/rect/round_rect/close`
  - path composition: `fs_path2d_add_path` / `fs_path2d_add_path_with_transform` (2x3 affine matrix)
  - draw/clip bridge: `fs_path_fill_path2d/fs_path_stroke_path2d/fs_clip_path2d`
  - demo MDN scene #2/#3 now uses real `Path2D` objects instead of immediate path replay.
- Render quality note:
  - bezier stroke SDF now uses segment-distance sampling (not point-distance) with adaptive sample count in fragment path.
  - rounded-rect fill/stroke AA now uses derivative-driven edge width (`fwidth`) instead of fixed 1px threshold, improving rotated edge stability.
  - oriented rounded-rect path now adds analytic 4-tap subpixel coverage in fragment stage to further suppress rotation stair-step artifacts.
  - path clip mask compute now uses adaptive supersampling (coarse 2x2 classify + boundary 4x4) to reduce clip edge jaggies.
  - clip edge AA is now scale-aware: boundary supersampling auto-selects `2x2/4x4/8x8` by viewport scale + clip transform scale hint.
  - manual override available via `fs_core_set_clip_aa_mode` (`-1` auto, `0` low, `1` medium, `2` high, `3` ultra).
  - demo scene keeps irregular clip/hole examples and adds `fs_path_arc_to`, `fs_path_rect`, `fs_path_round_rect` examples side-by-side.
  - clip write path now clears `union(old_layer_bounds, new_fill_bounds)` (+1px guard band) instead of full layer clear.
  - this preserves no-residual correctness while reducing clip compute dispatch area.
  - clip job `fill rect` now intersects parent clip-layer bounds chain before enqueue/dispatch.
  - if parent-chain intersection is empty, clip result is committed as an empty clip (no broad AABB fallback).
- demo now provides rule-clip A/B/C toggle via keyboard `V`: `RAW / CLIPPED / COMPARE`.
- demo now supports mouse-wheel / touchpad scrolling to pan long pages when examples overflow the viewport.
  - vertical scroll clamp now keeps a fixed preview margin of +/-20 screen px at top/bottom extremes.
  - demo miterLimit panel now includes interactive slider (LMB drag) for real-time join-tip behavior inspection.
  - API coverage page now includes a live `Path2D.addPath + addPath(path, transform)` sample with `isPointInPath/isPointInStroke` hit-test marker feedback.
  - demo main scene now includes a `GLOBAL ALPHA + GLOBAL COMPOSITE` comparison panel
    (auto-cycles across the currently wired subset).
  - path fill AA uses SDF distance solve in clip compute; generic `fs_clip_path` keeps coverage supersampling path.
  - clip cache is now frame-bounded (cross-frame hash entries are invalidated at frame begin) to avoid page-switch stale artifacts.
  - clip hash-cache reuse is disabled by default in demo/core run path for correctness; jobs are regenerated every frame.
  - clip layer reuse reserve is fixed to `0` in demo playback (fresh allocation priority); stress tuning hotkeys were removed from demo UI.
  - clip failure path now includes internal AABB fallback to keep rendering user-visible and avoid explicit failure markers.
  - MDN clip page now includes a conformance status line (`clip#1`, `clip#2-evenodd`, `clip#3-intersection`, `save/restore`).

### Transform stack
- PARTIAL: `save/restore/translate/rotate/scale/transform/setTransform/resetTransform` implemented by
  `fs_state_save/fs_state_restore/fs_translate/fs_rotate/fs_scale/fs_transform/fs_set_transform/fs_transform_reset`.
- UPDATE: transform readback API is now available via `fs_get_transform` (returns current 2x3 affine matrix).
- Current status: CPU submits local command payload + state snapshots; compute resolves transforms/bounds for most draw primitives.
- UPDATE: oriented-quad expansion is now wired for rect-like commands (`rect/rectStroke/image/text`) in compute stage.
  - command payload now carries quad basis (`origin + du + dv`) and compute emits vertices from quad parametric mapping.
  - this removes AABB-shape distortion under rotation/shear/mirror for these command families.
- UPDATE: text glyph pixel-snap now auto-disables under oriented transforms (rotate/shear/mirror) to prevent per-glyph jitter/tearing.
- FIX: text command flag packing now preserves internal oriented-quad bit while appending text-mode flags.
- UPDATE: compute pre-clamp now derives oriented-quad UV window from clipped AABB and emits cropped quad UV/world.
  - this reduces overdraw vs full-quad fallback while keeping clip correctness in fragment stage.
- UPDATE: render path now resolves from 4x MSAA color target (`compute-expanded vertices -> multisample render -> resolve`), reducing residual rotated-edge aliasing beyond single-sample SDF AA.
- UPDATE: oriented `rect/rectStroke` now uses compute-coverage mask path by default (`clip(path)` coverage build + clipped axis-aligned fill), replacing direct fragment-edge solve for these transformed cases.
- UPDATE: oriented `ellipse` and `arc` now route to compute-coverage path (path clip mask solve + clipped fill submit) instead of direct analytic fragment-edge solve.
- UPDATE: oriented `path stroke` routes segment/join/cap through compute-coverage primitives (polygon/circle-equivalent coverage).
  - bezier non-dash stroke now emits native bezier commands (no oriented CPU flatten loop).
  - dash mode still uses CPU-side segment expansion.
- UPDATE: command state preprocessing is now split from draw commands:
  - CPU submits command payload + per-command state snapshot (`command_states`) only.
  - compute stage consumes state snapshots for clip rect/path bounds and `globalAlpha` application.
  - render submission chooses blend pipeline from per-command state snapshot instead of baked command flags.
- UPDATE: `rect/rectStroke` non-oriented path now submits local-space geometry (`FS_RENDER_FLAG_LOCAL_SPACE`).
  - compute stage applies per-command affine transform matrix from state snapshot and derives world AABB there.
  - this removes CPU-side `rect` transform/AABB baking from the normal path.
- UPDATE: `image/text` non-oriented path now also submits local-space geometry.
  - CPU no longer bakes transformed AABB for these two families on the normal path.
  - compute stage applies state transform matrix, derives world bounds, and preserves existing UV/text sampling semantics.
- UPDATE: `line/path_segment` non-oriented path now submits local-space endpoints.
  - compute stage transforms segment endpoints and computes world-space AABB there.
- UPDATE: `circle/ellipse/arc` non-oriented path now submits local-space centers.
  - compute stage applies transform to center point and performs bounds derivation in compute.
  - radius/thickness scale for these primitives is resolved in compute from per-command transform state.
- UPDATE: `bezier quad/cubic` non-oriented path now submits local-space control points.
  - compute stage applies state transform to control points/endpoints and derives world-space bounds there.
  - stroke width scale is now resolved in compute from per-command transform state.
- UPDATE: `triangle` now submits local-space vertices.
  - compute stage applies state transform to triangle vertices and derives world-space bounds there.
- UPDATE: scalar/radius scaling for local-space primitives is now compute-side.
  - `rect/rectStroke`, `line/path_segment`, `circle/ellipse/arc`, `bezier quad/cubic` now submit raw local scalar values from CPU.
  - compute stage applies metric/x/y scale resolve from state transform and writes final render parameters.
- UPDATE: `fs_path_stroke` bezier non-dash path now skips oriented CPU flattening.
  - both quadratic and cubic segments now directly emit bezier commands for compute+render expansion under transform.
  - dash mode still uses CPU segment expansion by design in current step.
- FIX: removed unintended self-recursion in `fs_emit_styled_line_segment_compute_coverage` oriented path.
  - oriented line coverage path now executes deterministic polygon/cap emission instead of recursive re-entry.
- UPDATE: text stroke thickness scaling now resolves in compute for local-space text commands.
  - CPU text command payload now keeps stroke width in local units; transform-dependent scaling is applied in compute.
  - removed C-side transform-metric helper usage from text submit path.
- UPDATE: clip path edge build reduces per-segment transform work for quadratic/cubic flattening.
  - flatten loops now transform only the newly generated curve point each step and reuse prior transformed endpoint.
  - this removes duplicate point transforms in clip edge submission while preserving identical geometry.
- UPDATE: clip path edge build now also reuses transformed contour endpoints across connected segments.
  - line segment start-point transform and contour close-edge transforms reuse cached transformed endpoints when available.
  - this further reduces CPU-side transform calls in clip edge emission without changing clip geometry.
- UPDATE: clip curve flatten now evaluates in local path space with reduced CPU transform overhead.
  - quadratic/cubic clip edge expansion no longer performs per-sample transform calls on CPU.
  - local edge segments are transformed by compute pre-pass before clip-mask solve.
- UPDATE: clip edge transform pre-pass is now compute-driven.
  - CPU builds clip edges in local path space and enqueues per-job transform matrices.
  - `FS_CLIP_EDGE_TRANSFORM_WGSL` transforms local edges to device edges on GPU before clip-mask solve dispatch.
  - clip-mask compute now consumes transformed edge buffer; CPU no longer does per-edge point transform for clip jobs.
- Remaining limitation: UV window is derived from AABB corner inverse-map (conservative for some skewed intersections), not exact polygon clipping.

### Style/compositing state
- PARTIAL: per-command color + line style state exists (`lineWidth`, `lineCap`, `lineJoin`, `fillRule`, dash pattern + offset)
  via `fs_style_set_line_width/fs_style_set_line_cap/fs_style_set_line_join/fs_style_set_fill_rule/fs_style_set_dash`.
- UPDATE: dash readback API is now available via
  `fs_style_get_dash` (returns dash list copy + active offset).
- UPDATE: `miterLimit` state is now available via
  - `fs_style_set_miter_limit` / `fs_style_get_miter_limit`
  - `lineJoin=miter` now applies miter tip expansion with limit clamp, and auto-falls back to bevel when over-limit.
- UPDATE: `globalAlpha` and `globalCompositeOperation` extended common subset are now wired:
  - `fs_style_set_global_alpha` / `fs_style_get_global_alpha`
  - `fs_style_set_global_composite_operation` / `fs_style_get_global_composite_operation`
  - supported composite subset:
    `source-over`, `copy`, `lighter`, `destination-over`,
    `source-in`, `source-out`, `destination-in`, `destination-out`,
    `xor`, `source-atop`, `destination-atop`
- UPDATE: minimal `shadowColor + shadowBlur` style subset is now wired:
  - `fs_style_set_shadow_color` / `fs_style_get_shadow_color`
  - `fs_style_set_shadow_blur` / `fs_style_get_shadow_blur`
  - current implementation notes:
    - shadow emission is command-duplicated in core submit path (shadow pass first, then main pass).
    - blur is quantized to render flag payload for shader-side soft-edge widening.
    - shadow is currently implemented as a soft mask expansion path (not full physically-accurate Gaussian convolution).
- UPDATE: minimal solid `fillStyle/strokeStyle` state subset is now wired:
  - `fs_style_set_fill_color` / `fs_style_get_fill_color`
  - `fs_style_set_stroke_color` / `fs_style_get_stroke_color`
  - styled draw entry points:
    - `fs_fill_rect` / `fs_stroke_rect`
    - `fs_fill_text_utf8` / `fs_stroke_text_utf8`
    - `fs_fill` / `fs_stroke`
  - explicit-color path APIs (`fs_cmd_*`, `fs_path_fill`, `fs_path_stroke`) remain available for direct control.
- UPDATE: gradient object + style binding subset is now wired:
  - `fs_linear_gradient_create` / `fs_linear_gradient_add_color_stop` / `fs_linear_gradient_destroy`
  - `fs_radial_gradient_create` / `fs_radial_gradient_add_color_stop` / `fs_radial_gradient_destroy`
  - `fs_conic_gradient_create` / `fs_conic_gradient_add_color_stop` / `fs_conic_gradient_destroy`
  - `fs_style_set_fill_linear_gradient` / `fs_style_set_stroke_linear_gradient`
  - `fs_style_set_fill_radial_gradient` / `fs_style_set_stroke_radial_gradient`
  - `fs_style_set_fill_conic_gradient` / `fs_style_set_stroke_conic_gradient`
  - current behavior:
    - fill-path/fill-rect can consume linear/radial/conic gradient styles as grid-cell sampled draw under existing clip semantics.
    - text gradient now samples per-glyph (glyph-center) in styled `fillText/strokeText`; not per-fragment parity yet.
    - path `stroke` gradient now samples per-path-segment/per-join (and per-subsegment on dashed curve flattening); still not full per-fragment parity.
- UPDATE: image pattern object + style binding subset is now wired:
  - `fs_pattern_create_image` / `fs_pattern_destroy`
  - `fs_pattern_set_transform` (affine matrix subset)
  - `fs_style_set_fill_pattern` / `fs_style_set_stroke_pattern`
  - current behavior:
    - `fillRect` / `fill(path)` / `stroke(path)` and styled `fillText/strokeText` now use render-stage per-fragment pattern sampling on command paths that carry pattern metadata.
    - stroke path includes oriented/compute-coverage joins/caps/segments pattern propagation (no separate oriented fallback branch).
    - repeat subset supports: `repeat`, `repeat-x`, `repeat-y`, `no-repeat`.
    - `CanvasPattern.setTransform` affine subset (`a,b,c,d,e,f`) is consumed in fragment sampling path (no cell-sampled fallback branch in current fill pipeline).
    - W3C API coverage scene now includes a visual `pattern stroke (plain / rotate / rotate+dash)` comparison card for oriented-path regression checks.
- UPDATE: image smoothing style subset is now wired:
  - `fs_style_set_image_smoothing_enabled` / `fs_style_get_image_smoothing_enabled`
  - `fs_style_set_image_smoothing_quality` / `fs_style_get_image_smoothing_quality`
  - current behavior:
    - applies to `drawImage` and pattern sampling paths.
    - subset quality mapping: `low/medium -> linear`, `high -> minification-only multi-sample linear filter`.
    - demo W3C API coverage scene now includes `imageSmoothingEnabled + imageSmoothingQuality` visual card.
- Current status:
  - `fs_cmd_line` and `fs_path_stroke` can consume style state when width is 0.
  - dash is CPU-side command expansion before compute pass.
  - `lineJoin=round` is approximated by join circles at segment corners.
  - `lineCap=square` is supported by endpoint extension; `butt/round` share current capsule primitive behavior.
- NOT implemented yet: advanced `globalCompositeOperation` modes, browser-level shadow/filter parity.

### Rect command semantics
- UPDATE: `clearRect`-equivalent API is now available via `fs_cmd_clear_rect`.
  - implementation strategy: temporary `composite=copy` + transparent source draw, with shadow disabled during clear pass.

### Text API compatibility
- PARTIAL: UTF-8 text rendering + fallback + emoji image-font/native color glyph path.
- PARTIAL: text metrics + stroke API now available:
  - `fs_measure_text_utf8` (`FS_TextMetrics` width + actual bbox + em bbox estimate + glyph/line count)
  - `fs_cmd_stroke_text_utf8` (SDF stroke flag + stroke width, single text command path -> compute -> fragment SDF ring)
- Known limitation:
  - `strokeText` currently targets SDF glyphs; color glyph (RGBA emoji) does not run SDF stroke and is skipped in stroke pass.
  - `measureText` reflects current glyph layout path but not full Canvas `TextMetrics` parity.
- PARTIAL: text state knobs now include `textAlign` + `textBaseline` + `direction` + `fontKerning` + `textRendering` + `fontStretch` + `fontVariantCaps` + spacing subset:
  - `fs_style_set_text_align` / `fs_style_get_text_align`
  - `fs_style_set_text_baseline` / `fs_style_get_text_baseline`
  - `fs_style_set_text_direction` / `fs_style_get_text_direction`
  - `fs_style_set_font_kerning` / `fs_style_get_font_kerning`
  - `fs_style_set_text_rendering` / `fs_style_get_text_rendering`
  - `fs_style_set_font_stretch` / `fs_style_get_font_stretch`
  - `fs_style_set_font_variant_caps` / `fs_style_get_font_variant_caps`
  - `fs_style_set_letter_spacing` / `fs_style_get_letter_spacing`
  - `fs_style_set_word_spacing` / `fs_style_get_word_spacing`
  - applied by `fs_cmd_text_utf8` and `fs_cmd_stroke_text_utf8` for anchor-based text placement.
  - `textAlign=start/end` now honors `direction=ltr/rtl` anchor behavior (bidi shaping still out of scope).
  - current subset behavior:
    - `fontKerning=none` disables pair-kerning advance in draw + measure paths.
    - `textRendering=optimizeSpeed` also disables pair-kerning as a speed-biased hint.
    - `textRendering=geometricPrecision` disables text pixel-snap to reduce geometric drift.
    - `fontStretch` applies synthetic horizontal metric scaling (`ultra-condensed ... ultra-expanded`) in draw + measure paths.
    - `fontVariantCaps` currently provides ASCII `small-caps` approximation (lowercase->uppercase with reduced glyph size); other caps variants are accepted but not fully differentiated.
- NOT implemented yet: full browser-equivalent `fontStretch` face-selection and full-script `fontVariantCaps` typographic feature parity.

### Pixel IO / image data
- PARTIAL: image decode/upload pipeline is implemented.
- PARTIAL: atlas-handle pixel IO is available:
  - `fs_core_put_image_data_rgba8` (CPU RGBA writes queued to GPU upload path)
  - `fs_core_get_image_data_rgba8` (reads from image-atlas CPU shadow)
- UPDATE: canvas-rect ImageData subset APIs are now available:
  - `fs_core_create_image_data_rgba8`
  - `fs_core_put_canvas_image_data_rgba8`
  - `fs_core_get_canvas_image_data_rgba8`
- UPDATE: canvas-rect `getImageData` now includes a GPU-present readback path:
  - during `fs_core_encode`, current target texture is copied to an internal readback buffer (`CopyTextureToBuffer`).
  - backend now uses `wgpuQueueSubmitForIndex`, and core tracks submission index for readback synchronization.
  - `fs_core_get_canvas_image_data_rgba8` first tries to refresh canvas shadow from this readback buffer (best-effort), then falls back to existing shadow behavior.
- Known limitation:
  - readback source is the most recently submitted/presented frame copy (not full browser same-call immediate parity in all call orders).
  - fallback path still exists via core-managed canvas shadow (covers `put_canvas_image_data` tracked writes).
  - on platforms/surfaces without `CopySrc` usage support, framebuffer readback path is disabled and shadow fallback is used.
  - arbitrary framebuffer draw-result parity is improved but still not full browser-level edge-case parity.
  - format fixed to RGBA8.
- NOT implemented yet: full browser-equivalent framebuffer `ImageData` parity across all draw sources and blending paths.

### Context lifecycle/introspection
- PARTIAL: context introspection APIs are now available:
  - `fs_core_is_context_lost`
  - `fs_core_get_context_attributes` (`FS_ContextAttributes`)
- Current behavior:
  - `context_lost` flips to true on key encode/resize resource failures.
  - context attributes currently expose fixed backend capabilities (`alpha/premultiplied_alpha/antialias/depth/stencil/preserve_drawing_buffer`).
- NOT implemented yet: full browser-parity context loss recovery lifecycle and event model.

## Recommended next implementation order
1. Path fill robustness under strict GPU mode
  - done: failure diagnostics
  - done: safe layer reuse policy under exhaustion
  - done: stress-scene diagnostics + policy tuning hooks
    - MDN page adds a clip stress panel with high clip-path pressure draw workload.
    - stress workload is now default-on in demo playback (diagnostics panel remains visible).
    - live counters shown via `fs_core_get_clip_diagnostics`:
      requests / cache hits / jobs / layer reuses / failures / layers used / last failure payload.
    - reserve policy in demo is now fixed to `0` (favor fresh layer allocation for correctness; no end-user tuning hotkeys).
2. Path-based clip performance/feature parity
  - pending: clip job size-bucket scheduling rework (needs command-encoded per-dispatch uniform strategy)
  - remaining: Path2D sharing/reuse heuristics and further scheduling refinements
  - research-aligned next steps:
    - promote rect/rounded-rect clip into explicit scissor/analytic fast-path before mask sampling
    - evaluate depth-only clip prepass prototype (Graphite-like) for nested clip chains with many overdrawn draws
    - keep shader variant count stable when clip state changes (clip as data, not pipeline specialization)
3. Oriented-quad command for rotate/shear exactness on image/text/rect
4. Compositing state (`globalAlpha`, `blend/composite`, shadow/filter)
5. Text state parity (`textAlign/textBaseline/...`)

## Clip Exit Criteria (Definition of Done)
- Gate A: Canvas clip semantics correctness
  - pass: MDN-style clip#1 (basic region), clip#2 evenodd hole, clip#3 intersection, save/restore stack behavior.
  - pass: nested clip parent-chain does not leak outside ancestor clip bounds.
  - pass: `clip(path)` with rect/roundRect fast-path is pixel-consistent with generic path mode.
- Gate B: Runtime robustness
  - pass: no frame-stale / residual artifacts during page switch + scroll + resize cycles.
  - pass: no invalid-resource panics when resize happens during pending uploads/clip jobs.
  - pass: layer exhaustion scenarios degrade predictably (diagnostics + stable fallback behavior).
- Gate C: Performance floor (engineering target)
  - pass: clip dispatch uses size-bucket scheduling; large-scene sparse jobs no longer forced by single global max extent.
  - pass: strict rect/roundRect analytic fast-path hit rate is observable in diagnostic runs.
  - pass: no correctness regression when clip cache is on/off.
- Gate D: Exit to next subsystem
  - pass: public API behavior for clip path + fillRule + Path2D is stable for current demo corpus.
  - pass: remaining clip work only has "incremental optimization" items, not semantic blockers.
  - action: once Gate A/B/C/D all pass, move focus to non-clip roadmap items (oriented quad, compositing expansion, text parity).

## Current Clip Status
- Baseline assessment: clip stage is considered stable for current demo corpus (no obvious runtime bug under active validation flow).
- Execution policy: clip work moves to maintenance mode (regression fixes only), roadmap focus shifts to non-clip milestones.

## Demo Paging Policy
- Current status: demo page 1 is considered full; page 2 still has spare capacity.
- Permission rule: for future example additions, it is explicitly allowed to open a new page instead of forcing more content into page 1.
- Layout rule: prioritize readability and spacing; avoid overcrowding existing pages when adding new scenarios.

---

## Incremental Update (2026-04-06)

### New Public Headers

#### `fullstack_effects.h` — ENTIRELY NEW
Physical shadow/filter pipeline with separable Gaussian convolution:
- `FS_GAUSSIAN_KERNEL_MAX_SIZE` (63u) / `FS_GAUSSIAN_BLUR_RADIUS_MAX` (95.0f)
- `FS_PhysicalShadowParams` struct — full params: blur_radius, color, offset_x/y, kernel_size/sigma, flags
- `fs_effects_init/destroy/resize` — lifecycle
- `fs_gaussian_kernel_compute/compute_for_blur` — kernel computation
- `fs_style_set_physical_shadow` / `fs_style_get_physical_shadow` — param get/set
- `fs_style_set_gaussian_shadow` — convenience wrapper
- `fs_style_disable_physical_shadow` / `fs_style_is_physical_shadow_enabled`
- `fs_effects_render_physical_shadow` — internal render helper
- `fs_core_get_effects_resources` / `fs_core_set_effects_resources` — core integration
- Note: `fs_effects_render_physical_shadow` is currently a stub; shadow still uses command-duplication path.

#### `fullstack_core_gpu_layout.h` — ENTIRELY NEW (private layout, public-facing)
- `FS_CLIP_MASK_LAYERS` raised to 64 (was 16)
- `FS_CLIP_FILL_MODE_COVERAGE / SDF / ROUND_RECT` — clip fill mode enum
- `FS_ClipEdgeGPU` — local-space clip edge (16 bytes)
- `FS_ClipJobGPU` — clip job with fill_mode, scale_hint_bits, parent info (64 bytes)
- `FS_ClipJobTransformGPU` — per-job transform (32 bytes)
- `FS_ClipDispatchUniforms` — dispatch uniforms with aa_mode (32 bytes)
- `FS_ClipLayerUniforms` — parent chain + bounds for all 64 layers
- `_Static_assert` compile-time size validations

### New Source Files

- **`src/fullstack_core_path2d.c`** — dedicated Path2D subsystem file (path helpers, lifecycle)
- **`src/fullstack_core_transform.c`** — dedicated transform/math utilities:
  - `fs_transform_requires_oriented_quad()` — detects rotate/shear/mirror (checks b|c != 0 or a|d < 0)
  - `fs_affine_set_identity_2d()` / `fs_affine_try_invert_2d()` / `fs_affine_apply_point_2d()`
  - `fs_vec2_normalize()` / `fs_eval_quad_point()` / `fs_eval_cubic_point()` / `fs_distance_sq_point_segment()`
- **`src/fullstack_core_text_api.c`** — dedicated text rendering/measurement logic
- **`src/fullstack_core_clip_resources.c`** — clip GPU resources: dual-buffer edge system (local+device), per-job transform GPU buffers, range-sliced bind groups
- **`src/fullstack_core_upload.c`** — texture upload pipeline: `fs_queue_write_texture_2d`, `fs_flush_pending_texture_uploads`, `fs_discard_pending_uploads_for_texture`
- **`src/fullstack_core_canvas.c`** — canvas shadow/readback: `fs_ensure_canvas_shadow`, `fs_encode_canvas_readback_copy`, `fs_refresh_canvas_shadow_from_readback`

### Path State / Path2D (incremental updates)

- Path state temporary binding API: `fs_path_state_begin_temporary`, `fs_path_state_borrow`, `fs_path_state_restore`, `fs_path_state_bind_empty`
- Styled wrapper APIs: `fs_fill(core)` and `fs_stroke(core, width)` — consume current style state
- `fs_arc_resolve_delta` helper for arc angle resolution
- `FS_CMD_TRIANGLE = 11` added; `fs_cmd_triangle()` public API added
- Clip GPU resources: dual-buffer edge system (local+device), per-job transform GPU buffers, range-sliced bind groups
- Clip hash now includes fill_mode and parent layer hash
- `fs_clip_release_uncommitted_layer` — resets layer on partial job failure
- `fs_clip_apply_path_aabb_fallback` — internal AABB fallback on GPU clip failure

### Transform Stack (incremental updates)

- **`fs_set_transform(FS_Core*, a, b, c, d, e, f)`** — direct 6-component transform setter (replaces, unlike `fs_transform` which multiplies)
- **`fs_context_reset(FS_Core*)`** — full context state reset: stack clear, identity transform, style reset, clip reset, path reset
- `fs_ensure_state_stack_capacity` / `fs_state_stack_clear` — state stack management
- GPU helper functions in compute shader: `transform_point`, `transform_metric_scale`, `transform_length_x/y`, `oriented_uv_from_world`
- Command state snapshot (`FS_CommandStateGPU`, 112 bytes) bakes at submit time: xform0[4], xform1[4], clip_rect[4], clip_meta[4], pattern metadata
- T009/T010 (GPU round-join commandization + compute-side angle derivation) completed but gated OFF by default due to regression

### Style / Compositing (incremental updates)

- **`fs_style_set_shadow_offset`** / `get_shadow_offset_x` / `get_shadow_offset_y` — independent X/Y offset
- Shadow blur max raised from 15.0 to 95.0 (matching `FS_GAUSSIAN_BLUR_RADIUS_MAX`)
- `fs_radial_gradient_sample_rgba8` now uses full quadratic discriminant formula instead of linear approx
- `fs_fill_rect` / `fs_stroke_rect` wired through `fs_style_resolve_fill/stroke_color_at`
- `fs_draw_linear_gradient_rect_cells` does grid-cell gradient subdivision (up to 64x64 cells)
- `gradient_add_color_stop` now does sorted insertion
- `fs_style_set_fill/stroke_pattern` validates atlas origin before setting paint type
- `fs_style_pattern_sample_rgba8` does full per-fragment sampling with inverse transform + repeat modes

### Text API (incremental updates)

New helper functions in `fullstack_core_style.c`:
- `fs_resolve_text_vertical_metrics` — computes font ascent/descent/line-height (defaults 0.8/0.2/1.25 of font_size_px)
- `fs_text_align_offset` — bidi-aware anchor offset for textAlign=start/end with direction
- `fs_is_word_spacing_codepoint` — full Unicode space codepoint list (31 codepoints)
- `fs_text_baseline_offset` — 6-mode baseline offset
- `fs_is_text_kerning_enabled` — respects fontKerning=none AND textRendering=optimizeSpeed
- `fs_is_text_geometric_precision` — disables pixel-snap when textRendering=geometricPrecision
- `fs_text_stretch_scale` — maps 9 stretch enum values to numeric scales (0.5 to 2.0)
- `fs_is_text_small_caps_enabled` / `fs_text_variant_map_codepoint` — small-caps ASCII mapping with 0.82x scale

Style color mode system for text:
- `fs_cmd_text_utf8_internal` accepts `style_color_mode` (`FS_TEXT_STYLE_COLOR_NONE/FILL/STROKE`)
- `fs_resolve_text_draw_color` — resolves style color at glyph center for gradient text
- Color glyphs (emoji) bypass gradient/pattern application

Public API additions:
- **`fs_fill_text_utf8(core, x, baseline_y, font_size_px, utf8, max_width)`** — styled fill entry
- **`fs_stroke_text_utf8(core, x, baseline_y, font_size_px, utf8, max_width, stroke_width)`** — styled stroke entry

`fs_measure_text_utf8` improvements:
- Now computes `actualBoundingBoxLeft/Right/Ascent/Descent` from per-glyph layout (previously only width + glyph/line counts)
- Respects `max_width` — breaks iteration when pen exceeds max_width

Expanded `FontVariantCaps` enum: `PETITE_CAPS`, `ALL_PETITE_CAPS`, `UNICASE`, `TITLING_CAPS` (accepted but mostly pass-through except small-caps/all-small-caps)

Experimental shaping path (`FS_ENABLE_EXPERIMENTAL_SHAPING = 0`, disabled by default):
- Uses `font_backend->shape_text_utf8` from extended `FS_FontBackend` with `shape_text_utf8 / free_shaped_text / free_glyph_pixels` callbacks
- Preconditions: single font, no word spacing, kerning enabled, no stretch, no small caps, no image fonts, no newlines

### Pixel IO / Image Data (incremental updates)

- Dual staging buffer architecture: `upload_staging_cpu` (malloc) + `upload_staging_gpu` (WGPUBuffer CopyDst), batched flush at encode time, 256-byte row alignment
- Image atlas shadow mirroring on write: `fs_queue_write_texture_2d` updates `image_atlas_shadow_rgba` in the same call
- Canvas readback pipeline: `canvas_readback_buffer` + `canvas_readback_serial` monotonic counter + `fs_core_notify_submission` for submission index tracking
- `fs_core_put_canvas_image_data_rgba8` now does triple write: canvas shadow + atlas upload + image draw command
- `fs_core_get_canvas_image_data_rgba8` flow: refresh shadow from GPU readback first, then read from shadow with bounds clamping
- `FS_PendingTextureUpload` struct tracks deferred uploads: texture/layer/xy/width/height/padded_row/src_offset

### Public API Surface (incremental updates)

**`fullstack_core.h`:**
- `fs_core_notify_submission(FS_Core*, WGPUSubmissionIndex)` — track GPU submission index for readback sync
- `fs_core_get_missing_image_glyph_count` / `fs_core_get_missing_image_glyph` / `fs_core_clear_missing_image_glyphs` — missing glyph introspection
- `fs_core_get_image_backend_name` / `fs_core_get_font_backend_name` — backend introspection

**`fullstack_backend_api.h` (new types):**
- `FS_FontGlyphBitmap` — glyph bitmap with SDF params (sdf_radius_px, sdf_onedge, sdf_pixel_dist_scale)
- `FS_FontGlyphPixelFormat` enum: `FS_FONT_GLYPH_PIXEL_FORMAT_SDF_R8`, `FS_FONT_GLYPH_PIXEL_FORMAT_RGBA8`
- `FS_ShapedGlyph` / `FS_ShapedTextRun` — text shaping types
- `FS_FontVerticalMetrics` — font metrics (ascent/descent/line_height)
- `FS_FontBackend` — extended with shape_text_utf8 / free_shaped_text / free_glyph_pixels callbacks

**`fullstack_core_debug.h`:**
- `fs_core_set_clip_cache_enabled` / `fs_core_get_clip_cache_enabled` — toggle clip hash-cache (disabled by default)
- `fs_core_set_clip_aa_mode(FS_Core*, int32_t mode)` — override clip AA mode: -1 auto, 0 low, 1 medium, 2 high, 3 ultra

**`fullstack_core_private.h` (new internal state):**
- `FS_HitFillContext` — fill hit-test context
- `clip_aa_mode_override` / `clip_cache_enabled` / `clip_layer_reuse_reserve` tuning params
- `canvas_readback_submission` / `_submission_valid` / `_mapped` — enhanced readback tracking
- `image_atlas_shadow_rgba` / `image_atlas_shadow_size` — per-layer atlas shadow
- `canvas_shadow_rgba` / `canvas_shadow_size` / `canvas_image_data_handle` — canvas pixel shadow + atlas slot
- `FS_MapReadbackContext` — volatile done/success flags for async map callback

### CSS Filter Pipeline (WebGPU compute-based)

Full 10-filter CSS filter pipeline implemented via WebGPU compute + ping-pong textures:

- **`FS_EFFECTS_FILTER_WGSL`** — single compute shader with switch on `filter_type`:
  - brightness (1), contrast (2), grayscale (3), hue-rotate (4), invert (5), opacity (6), saturate (7), sepia (8), blur (9), drop-shadow (10)
- **`FS_EFFECTS_GAUSSIAN_BLUR_WGSL`** — separable H+V Gaussian blur (256-wide workgroups)
- **`FS_EFFECTS_FILTER_COPY_WGSL`** — passthrough render shader for presentation
- **`FS_EFFECTS_VERT_WGSL`** — overdrawn triangle vertex shader for fullscreen passes
- **`FS_EFFECTS_SHADOW_COMPOSITE_WGSL`** — drop-shadow composite shader
- Ping-pong architecture: scene → A → [filter/gaussian H/V] → B → [filter] → A → scene
- Presentation: scene_texture → canvas via `presentation_pipeline` (RGBA8Unorm → swap chain format)
- Public API: `fs_style_set_filter(core, "brightness(1.5) contrast(1.8) grayscale(1) ...")`
- Filter chain: `fs_filter_chain_parse` → `fs_filter_chain_execute` in `fs_core_encode`
- Parser supports: unitless values, % suffix, deg/rad suffix, px suffix, hex/rgba colors

**Bug fixes (2026-04-06):**

1. **`fs_vs_main` diagonal split rendering** — Fullscreen quad vertex shader using `vi%2*2-1` generated 4 vertices creating 2 triangles with a diagonal seam. Fixed by switching to overdrawn triangle technique: `x=f32(vi&1u)*4.0-1.0`, `y=f32(vi>>1u)*4.0-1.0` (3 vertices covering entire [-1,1] clip space). Updated all `Draw(4,...)` calls to `Draw(3,...)` in presentation and shadow composite passes.

2. **Simple filter uniform buffer overwrite** — `wgpuQueueWriteBuffer` is synchronous (writes immediately, not queued). B→A passthrough used `writeBuffer(NONE)` which overwrote the preceding filter's uniform parameters before GPU execution, making all non-blur filters identity copies. Fixed by removing B→A compute passthrough entirely and using direct texture copy based on result location (`write_to_a`).

3. **Double-copy texture conflict** — Original flow did B→A copy then A→scene copy within the same command encoder (write A then read A), causing WebGPU validation conflict / GPU hang. Fixed by directly copying from result location (A or B) to scene_texture in one operation, eliminating intermediate state.

4. **Gaussian blur V-pass bind group swap** — V-pass BGLs were identical to H-pass (both read B, write A). Fixed by swapping: V_bg_a reads B→writes A, V_bg_b reads A→writes B.

5. **Drop-shadow dispatch too small** — Dispatch only covered source bounds, not shadow offset region. Fixed by expanding dispatch dimensions by `offset_x`/`offset_y`.

6. **Filter name length parsing** — `hue-rotate` (10 chars) parsed as 9, `drop-shadow` (11) as 10. Fixed in `fs_filter_type_from_name`.

7. **Shader fixes** — grayscale `mix` direction corrected, sepia matrix transposed (WGSL is column-major), hue-rotate matrix coefficients fixed.

8. **Drop-shadow px suffix** — `8px 8px` parsing consumed `8` then left `px` unconsumed, breaking subsequent parameter parsing. Fixed by adding px suffix consumption in `fs_filter_parse_radius`.

**Filter pipeline data flow (corrected):**
1. `scene_texture` → `wgpuCommandEncoderCopyTextureToTexture` → `ping_pong_A`
2. For each filter: compute dispatch (read A/B, write opposite) — no `wgpuQueueWriteBuffer` between chained filters
3. After chain: direct copy from result texture (A, B, or shadow_composite) → `scene_texture`
4. `presentation_pipeline` samples `scene_texture` → canvas swap chain

**Key architectural insight:** `wgpuQueueWriteBuffer` executes immediately on the CPU timeline, not the GPU timeline. Any `writeBuffer` call between command recording and submission overwrites the buffer before the GPU reads it. The filter chain must write the uniform buffer immediately before each dispatch, and there must be no subsequent `writeBuffer` calls before the GPU executes that dispatch.
