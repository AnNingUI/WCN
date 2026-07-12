# Make the MORROW clock responsive and backend-neutral

The MORROW clock demo will replace its fixed 1600 by 1000 proportional canvas
with a responsive layout that reflows for desktop, tablet, and phone-sized
windows. The change will also remove direct GLFW time and keyboard dependencies
from the demo so a future SDL3 backend can reuse the same UI behavior.

This specification supersedes the fixed proportional-scaling requirements in
the [original MORROW redesign
specification](2026-07-11-morrow-clock-monet-redesign.md).
The visual identity, clock emphasis, Monet colors, motion, and existing
interactions remain in effect unless this document explicitly changes them.

> **Note:** This specification targets narrow desktop GLFW windows. A future
> SDL3 backend will provide actual Android and iOS platform support.

## Goals

The responsive redesign must keep the complete clock experience readable across
a wide range of window sizes.

- Preserve the digital clock as about 70 percent of the main visual emphasis.
- Preserve the analog clock as about 30 percent of the main visual emphasis.
- Reflow content instead of scaling the complete page as one unit.
- Keep search, Monet selection, themes, time formats, and city management.
- Support vertical scrolling when content exceeds the viewport height.
- Establish backend-neutral time, viewport, keyboard, and scroll contracts.
- Remove direct GLFW symbols from `morrow_clock_demo.cpp`.
- Keep the UI contract reusable by a future SDL3 backend.

## Non-goals

This change does not implement the SDL3 backend, Android packaging, iOS
packaging, touch platform integration, or a polymorphic rendering backend. It
does not rewrite the MORROW UI as a tree of Taffy components.

The work does not change the approved Material Design 3 Monet color system, the
standard image seed extraction pipeline, or asynchronous theme extraction.

## Approved visual direction

The approved layout uses a responsive vertical flow on narrow windows. It does
not use a paged dashboard or hide saved cities behind a separate screen.

The mobile header uses two compact control rows. **Monet Image**, theme, and
time-format controls remain directly visible instead of moving into a hamburger
menu.

Tablet and phone-sized layouts use this content order:

1. Show the MORROW identity and appearance controls.
2. Show the city search field.
3. Show the primary digital clock.
4. Show the compact analog clock card.
5. Show the saved-city heading and add action.
6. Show the saved-city grid.

The page uses natural vertical scrolling when content is taller than the
viewport.

## Backend-neutral application contract

The UI event and viewport contract will live in `examples/ui/element.hpp` and
`examples/ui/app.hpp`. Platform backends map native input to this contract.

### Keyboard events

`element.hpp` will define a backend-neutral `Key` enumeration. The initial
enumeration must include `Unknown`, `Backspace`, `Escape`, and `Enter`.

`KeyEvent` will expose the backend-neutral key value. GLFW constants remain
inside `app.hpp`, where callbacks map them to the shared enumeration. The
MORROW demo must not reference `GLFW_KEY_*`.

The event can retain native scan codes and modifiers for diagnostics or future
expansion, but UI behavior must use `Key`.

### Time

`App` will expose monotonic time in seconds through `time_seconds()`. It will
use `std::chrono::steady_clock` instead of a window-backend clock.

The MORROW demo will use this value for theme transitions, state motion, caret
blinking, asynchronous Monet progress, and continuous clock animation. The demo
must not call `glfwGetTime()`.

### Viewport

`App` will expose `ViewportMetrics` with the current drawable width and height.
Existing `width()` and `height()` accessors can remain as compatibility
wrappers.

MORROW will only read viewport data from `App`. A future SDL3 backend must
update the same metrics when its drawable size changes.

### Scrolling

`App` will consume the active backend's scroll deltas and dispatch
`EventType::Scroll`. The GLFW backend already accumulates scroll deltas, but
`App` does not currently convert them into UI events.

MORROW will respond only to the shared scroll event. A future SDL3 backend can
map wheels, trackpads, or touch scrolling to the same event.

## Responsive layout model

The demo will introduce `ResponsiveLayout`. It contains every calculated
component rectangle, text scale, grid setting, and the total content height for
the current viewport.

Drawing and hit testing must consume the same layout value. The implementation
must not calculate visible and interactive rectangles separately.

### Width classes

The layout uses four width classes:

- **Desktop, 1100 pixels and wider:** Keep a 70/30 two-column clock composition.
  Show saved cities in four columns.
- **Tablet, 700 through 1099 pixels:** Stack the clocks. Render the analog clock
  as a compact horizontal card. Show saved cities in two columns.
- **Mobile, 420 through 699 pixels:** Use the approved two-row control bar,
  followed by search, the clocks, and a two-column saved-city grid.
- **Compact mobile, narrower than 420 pixels:** Reduce margins and gaps. Change
  saved cities to one column when two columns cannot preserve readable content.

The centered content container has a maximum width of about 1440 pixels.
Wider windows add outer breathing room instead of stretching cards without
limit.

### Fluid values

Spacing and typography can interpolate within each width class, but they must
stay within explicit readable bounds.

- Outer margins must not become smaller than 16 pixels.
- Interactive targets must remain at least 44 by 44 pixels.
- The primary digital time must remain the largest type at every width.
- Seconds can reduce more aggressively than hours and minutes.
- Supporting labels can wrap or move instead of shrinking below readability.
- Card radii and Monet surface hierarchy remain consistent.

The implementation must not restore complete-page proportional scaling inside
a breakpoint.

## Component behavior

The desktop header remains one row when space permits. Tablet and mobile
layouts wrap it into two compact rows while keeping all appearance controls
visible.

The digital clock retains city, date, UTC offset, daylight state, and the
24-hour track. Narrow layouts can move metadata onto more rows but must not
remove it.

The analog clock remains present at every width. Tablet and mobile use a compact
horizontal card with the face on one side and city metadata on the other.

Saved-city cards use four, two, or one column. Remove targets must remain
separate from UTC and daylight labels.

Search results use the existing high z-index layer. Their width and position
derive from the responsive search field, and they move with page scrolling.

The asynchronous Monet button keeps its **Extracting** label and linear
indicator. Theme application remains on the UI thread.

## Scrolling and pointer interaction

The dashboard will store current and target scroll positions. Scroll events
change the target, and short interpolation moves the current position without
blocking rendering.

Both values must remain between zero and
`content_height - viewport_height`. Resizing recalculates and reclamps the
range.

A pointer drag that starts on noninteractive page space can update the same
scroll target. This mirrors a future SDL3 touch gesture. Dragging a button,
card, search result, or text input must not start page scrolling.

All page content receives the same negative vertical offset. Hit testing must
account for that offset before comparing pointer coordinates with layout
rectangles.

The first and last positions must not overscroll into blank content. Canceling
a drag must leave controls in valid hover and press states.

## Error handling

Layout calculation must remain valid for minimized, zero-sized, and extremely
narrow windows. Rendering can skip zero-sized frames.

If content is shorter than the viewport, the scroll range is zero. If resizing
invalidates the current position, the implementation clamps it in the same
frame.

Unknown native keys map to `Key::Unknown` and do not activate search commands.
A missing scroll source leaves the current position unchanged.

Existing font fallback, Monet extraction, seed persistence, and city fallback
behavior remain unchanged.

## Implementation boundaries

The main implementation will update these files:

- `examples/ui/element.hpp` for the backend-neutral key contract.
- `examples/ui/app.hpp` for key mapping, time, viewport metrics, and scroll
  dispatch.
- `examples/morrow_clock_demo.cpp` for responsive layout, scrolling, and removal
  of direct GLFW dependencies.

The work can add focused helper structures near the demo. It must not perform
unrelated Taffy cleanup or modify unrelated dirty files.

The implementation preserves custom Canvas rendering. A conversion to
Taffy-managed child elements is deferred.

## Verification

Verification covers layout, interaction, backend boundaries, and regressions.

1. Build `morrow_clock_demo` without warnings introduced by changed files.
2. Confirm the demo contains no `glfwGetTime`, `GLFW_KEY_*`, GLFW window
   queries, or native backend pointer access.
3. Test 1600 by 1000, 1024 by 768, 768 by 1024, 430 by 932, and 360 by 800.
4. Drag the window across breakpoints and check reflow, clipping, and hit areas.
5. Test wheel and trackpad scrolling at both boundaries.
6. Test pointer-drag scrolling from empty page space.
7. Confirm dragging interactive controls does not scroll the page.
8. Verify search focus, caret, Enter, Escape, and overlay ordering.
9. Verify city selection, add, remove, UTC labels, and remove targets.
10. Verify themes, time formats, and asynchronous Monet image extraction.
11. Confirm the digital clock remains dominant and the analog clock visible.

## Future SDL3 backend

A later SDL3 implementation will map SDL keyboard, pointer, scroll, resize, and
touch events to the same UI event types. It will update the same viewport
metrics and use the same monotonic application clock.

Android and iOS support also requires SDL3 platform setup, lifecycle handling,
drawable-density handling, packaging, and platform-specific file selection.
Those requirements belong in a separate specification.

## Next steps

After final review, create an implementation plan that sequences application
contract changes before responsive MORROW layout changes.
