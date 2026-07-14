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

Append a work-area query to the App window-control capability. The public API
returns the usable logical-pixel rectangle for the display containing the
largest portion of a window. The rectangle uses desktop coordinates and
excludes reserved system UI such as the Windows taskbar.

- GLFW resolves the containing monitor and calls its monitor work-area API.
- SDL3 resolves the window display and reads the usable display bounds.
- Mock stores a configurable deterministic work area for tests.
- An unsupported backend returns `FS_RESULT_UNSUPPORTED` without inventing a
  desktop size.

Appending the capability preserves older backend compatibility through the
existing capability-size checks.

## Enemy seam transfer

The world evaluates window overlap in desktop space after enemy movement. An
enemy can transfer when its center reaches a shared overlap seam and its
velocity points into the destination window. The transfer converts the enemy's
position and velocity through desktop coordinates, changes its owner window,
and preserves health, phase, attack cooldown, elite state, and object ID.

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

The coordinator raises the main window once after background creation. Child
windows created later naturally remain above the background. Clicking the
background keeps input inside the Window Arena process but does not synthesize
combat fire. The background is visible only in running, boss, level-up, and
pause phases. It hides on title, victory, game over, and shutdown.

Display or work-area changes resize the background and update the containment
rectangle. If work-area discovery fails, the game continues without the
background and logs a recoverable warning.

## Boundary rebound

The coordinator owns a backend-neutral containment controller per combat
window. The world continues to emit requested geometry. Before an App geometry
write, the controller compares the request with the observed work area.

When an edge is contacted or crossed, the controller performs these actions:

1. Clamp the requested rectangle inside the work area.
2. If the rectangle is larger than the safe area, shrink it while preserving
   the existing minimum size and visible drag strip.
3. Reflect the velocity component for each contacted axis.
4. Add a bounded inward impulse and run a short, damped recovery animation.

Corner contact reflects both axes. Geometry writes remain at or below 30 Hz.
The controller treats manually observed geometry as authoritative and cancels
stale recovery targets after user drag or display change. Reduced-motion mode
uses a maximum 24-pixel inward displacement and a 160-millisecond ease-out
instead of a spring-like rebound.

The background window never rebounds. The main window rebounds only when an
expansion or game-authored geometry request contacts an edge; ordinary manual
placement remains authoritative.

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

App work-area support belongs under `app/`. Enemy transfer remains in
`examples/support/window_arena_world.[ch]`. Background creation, rendering,
and containment application remain in `examples/multi_window_arena_demo.c`
and focused support modules. Root `src/` and root `include/` remain unchanged.

