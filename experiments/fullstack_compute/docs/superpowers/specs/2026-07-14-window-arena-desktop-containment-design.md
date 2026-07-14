# Window Arena desktop containment design

This design adds three desktop-level behaviors to Window Arena: enemies can
cross overlap seams into the main window, a background window retains empty
desktop clicks during a run, and game windows rebound inward when expansion
touches a display work-area edge.

> **Note:** This is a preview feature currently under active development.

## Goals

The implementation must produce these player-visible outcomes:

- An enemy can cross from an overlapping child window into the main window.
  When the child is below the main window, the enemy crosses from the child's
  top edge to the main window's bottom edge without a visible position jump.
- A borderless background window covers the current display work area during a
  run, excludes the taskbar, receives empty-area clicks, and remains below all
  combat windows.
- A window that reaches a work-area edge because projectile impact expanded it
  rebounds inward with a short DVD-logo-style movement instead of leaving the
  visible desktop.
- GLFW, SDL3, and Mock expose identical work-area behavior through App APIs.

## Non-goals

The background window does not use a permanent system-level topmost flag. It
does not prevent Alt+Tab, operating-system shortcuts, or deliberate activation
of another application. The main window does not drift continuously, and no
window bounces until it contacts a work-area boundary.

## App work-area query

Append `get_work_area` to `FS_AppWindowControlCapability` and expose
`fs_app_window_get_work_area(app, window, out_geometry, error)`. The function
returns an `FS_AppWindowGeometry` whose desktop position and logical size
describe the usable logical-pixel rectangle for the display containing the
largest portion of `window`. The rectangle excludes reserved system UI such as
the Windows taskbar.

- GLFW resolves the containing monitor and calls its monitor work-area API.
- SDL3 resolves the window display and reads the usable display bounds.
- Mock defaults to `0,0,1920,1080` and exposes a test-only setter for a
  deterministic work area.
- An unsupported backend returns `FS_RESULT_UNSUPPORTED` without inventing a
  desktop size.

Appending the capability preserves older backend compatibility through the
existing capability-size checks. This feature is an explicit exception to the
earlier survivor specification's assumption that no App public API change was
required. It changes only `app/`; root `src/` and root `include/` remain
unchanged.

## Enemy seam transfer

The world evaluates window overlap in desktop space after enemy movement. A
base-rule transfer is limited to child-to-main movement: an enemy can enter the
main window when its center reaches a shared overlap seam and its velocity
points from the child into the main window. The Window Weaver's Reverse Link
spell remains distinct because it enables main-to-child hostile movement and
hostile-projectile lane use. The transfer converts position through desktop
coordinates and changes only `owner`, `x`, `y`, and the transfer-cooldown
field. Every other byte of enemy state remains unchanged, including velocity,
health, phase, attack cooldown, modifier state, and object ID.

The resolver selects the lowest destination window ID when multiple windows
contain the crossing point. A 250-millisecond transfer cooldown prevents an
enemy from oscillating between two windows. The resolver limits one transfer
per enemy per fixed step. Closing and telegraph-only windows aren't eligible.

## Background window

The coordinator creates one background view after it obtains the main window's
work area. The background is borderless, non-resizable, opaque, and sized to
the usable work area. It has a quiet radar-grid presentation and does not join
the combat world, window capacity, overlap graph, fragment ownership, or boss
spell system.

The background isn't permanently topmost or bottommost because GLFW and SDL3
don't expose a portable bottom-level contract. Instead, the coordinator runs a
z-order reconciliation after background creation, after every child-window
creation, and whenever the background receives focus or pointer input. It
raises each active combat window in stable window-ID order and raises the main
window last. This repeatable rule prevents the activated background from
remaining above combat windows. Clicking the background keeps input inside the
Window Arena process but does not synthesize combat fire. The background is
visible only in running, boss, level-up, and pause phases. It hides on title,
victory, game over, and shutdown.

Display or work-area changes resize the background and update the containment
rectangle. If work-area discovery fails, the game continues without the
background and logs a recoverable warning.

## Boundary rebound

The coordinator owns a backend-neutral containment controller per combat
window. Each `WA_ACTION_SET_GEOMETRY` includes a provenance value. Only
`WA_GEOMETRY_PROJECTILE_EXPANSION` can start a rebound. Pressure shrink, user
drag, director placement, boss commit, and boss recovery clamp to their own
safety rules but never start this animation. Before a projectile-expansion App
geometry write, the controller compares the request with the observed work
area.

When an edge is contacted or crossed, the controller performs these actions:

1. Clamp the requested rectangle inside the work area.
2. If the rectangle is larger than the safe area, shrink it while preserving
   the existing minimum size and visible drag strip.
3. Set an inward velocity of 520 logical pixels per second on each contacted
   axis.
4. Integrate translation for 320 milliseconds with exponential damping of
   `velocity *= exp(-9 * dt)` and a maximum inward displacement of 64 logical
   pixels per axis.

Corner contact activates both axes. A controller stores its last observed
geometry, last valid work area, current velocity, remaining animation time,
and provenance serial. Geometry writes remain at or below 30 Hz.
The controller treats manually observed geometry as authoritative and cancels
stale recovery targets after user drag or display change. Reduced-motion mode
uses a maximum 24-pixel inward displacement and a 160-millisecond ease-out
instead of a spring-like rebound.

If a requested window is larger than the work area minus the 24-pixel margins,
the controller shrinks it to that safe rectangle before applying the inward
impulse. It preserves the 360 by 260 main minimum and the 48 by 48 child-visible
minimum when the work area can contain them. If the work area itself is smaller
than a required minimum, the controller uses the largest available rectangle,
disables the animated displacement, and reports a constrained result.

The background window never rebounds. Main and child combat windows rebound
only when a projectile-expansion request contacts an edge. Manual placement
and boss spell commit or recovery remain authoritative and cannot inherit a
stale rebound because a different provenance serial cancels the controller.

## Failure handling

The implementation follows these recovery rules:

- If an enemy's destination disappears, keep the enemy in its current owner
  and clamp it inside the source window.
- If background creation fails, continue the run without click capture.
- If a work-area query fails during rebound, retain the last valid work area.
  If no valid rectangle exists, skip the geometry write.
- If App rejects a rebound write, stop that recovery and reconcile from the
  next observed geometry event.

## Verification

Headless world tests cover transfers across all four overlap directions,
especially upward transfer from a lower child window, cooldown behavior, and
object-state preservation. App Mock tests cover work-area queries, background
lifecycle, all four edge reflections, corner reflection, oversize geometry,
reduced motion, and failed writes. GLFW and SDL3 smoke tests verify that the
background excludes the taskbar and that projectile-driven expansion remains
visible after edge contact.

## Implementation boundary

App work-area support belongs under `app/`. This approved exception supersedes
the earlier no-App-change assumption for this query only. Enemy transfer remains in
`examples/support/window_arena_world.[ch]`. Background creation, rendering,
and containment application remain in `examples/multi_window_arena_demo.c`
and focused support modules. Root `src/` and root `include/` remain unchanged.
