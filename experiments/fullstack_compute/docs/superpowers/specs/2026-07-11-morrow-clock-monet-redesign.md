# Redesign the MORROW world clock

The clock demo will become an original MORROW world-clock experience. It will
keep the large digital clock as the main product element, retain the analog
clock as a secondary instrument, and use a Material Design 3 Monet palette
without adopting Material component styling.

## Goals

The redesign must preserve the demo's identity as a clock while improving its
visual hierarchy and originality.

- Give the digital clock about 70 percent of the main visual emphasis.
- Give the analog clock about 30 percent of the main visual emphasis.
- Keep search, city selection, city add and remove actions, theme switching,
  and 12-hour or 24-hour formatting.
- Replace the EUI-NEO and TimeSpot names with the original MORROW identity.
- Use Material Design 3 Monet color roles for both light and dark themes.
- Combine the strongest editorial, precision-instrument, and product-interface
  ideas explored during visual design.
- Preserve the current 1600 by 1000 design coordinate system and proportional
  window scaling.

## Non-goals

This redesign does not convert the interface into a standard Material Design
dashboard. It does not add chat, meeting scheduling, weather data, geographic
maps, external images, or decorative three-dimensional artwork. It also does
not change the core renderer or introduce new low-level drawing APIs.

## Visual direction

The interface uses MD3 Monet only as its semantic color system. Layout,
typography, component geometry, and information pacing come from the approved
hybrid direction.

- Use precision-instrument restraint for the main digital time and analog
  clock.
- Use warm editorial spacing and typography around the clock rather than a
  dense dashboard layout.
- Use real city, UTC, date, daylight, and local-time information as visual
  content instead of decorative graphics.
- Use clear action hierarchy and compact controls without applying standard
  Material component shapes throughout the page.
- Use continuous corners from the existing renderer for containers and cards.

The primary clock area occupies most of the first screen. The large digital
time sits on the left, and the analog clock sits in a secondary surface on the
right. A functional 24-hour daylight track runs below them. Saved cities form
a quieter supporting area below the main clock.

## Monet color system

The palette uses the exact seed color `#B65F49` and contrast level `0.0`.
It follows `SchemeTonalSpot` from `@material/material-color-utilities`
version `0.3.0`. The implementation may store precomputed constants, but
they must match that implementation and configuration.

`SchemeTonalSpot` constructs its tonal palettes from the seed HCT hue:

- Primary uses the seed hue with chroma 36.
- Secondary uses the seed hue with chroma 16.
- Tertiary uses the seed hue plus 60 degrees with chroma 24.
- Neutral uses the seed hue with chroma 6.
- Neutral variant uses the seed hue with chroma 8.
- Error uses the Material Color Utilities standard error palette.

At contrast level `0.0`, light roles use these exact tones:

- Primary, secondary, tertiary, and error use tone 40; their `on` roles use
  tone 100; their containers use tone 90; and container content uses tone 10.
- `surface` uses tone 98, `surface_container_low` uses tone 96,
  `surface_container` uses tone 94, `surface_container_high` uses tone 92,
  and `surface_container_highest` uses tone 90.
- `on_surface` uses tone 10, `on_surface_variant` uses tone 30, `outline`
  uses tone 50, and `outline_variant` uses tone 80.
- `inverse_surface` uses tone 20, `inverse_on_surface` uses tone 95, and
  `inverse_primary` uses tone 80.

At contrast level `0.0`, dark roles use these exact tones:

- Primary, secondary, tertiary, and error use tone 80; their `on` roles use
  tone 20; their containers use tone 30; and container content uses tone 90.
- `surface` uses tone 6, `surface_container_low` uses tone 10,
  `surface_container` uses tone 12, `surface_container_high` uses tone 17,
  and `surface_container_highest` uses tone 22.
- `on_surface` uses tone 90, `on_surface_variant` uses tone 80, `outline`
  uses tone 60, and `outline_variant` uses tone 30.
- `inverse_surface` uses tone 90, `inverse_on_surface` uses tone 20, and
  `inverse_primary` uses tone 40.

Both themes use neutral tone 0 for the separate `shadow` and `scrim` roles.
Each theme exposes semantic roles instead of direct component colors.

- `surface` provides the page background.
- `surface_container_low` and `surface_container_high` separate page layers.
- `primary` and `on_primary` identify the highest-priority actions and current
  time markers.
- `primary_container` and `on_primary_container` identify the selected city.
- `secondary_container` supports the analog-clock surface.
- `tertiary` and `tertiary_container` represent daylight and horizon data.
- `outline` and `outline_variant` provide borders, ticks, and dividers.
- `on_surface` and `on_surface_variant` provide primary and secondary text.

The dark theme must derive from the same seed-generated Monet scheme. Monet
may shift the secondary and tertiary palette hues, but the implementation must
not introduce an independently selected neutral-gray or cool-gray palette.

Every rendered color must come from a Monet semantic role or a role-based
alpha state layer. This includes hover, focus, pressed, selected, disabled,
shadow, clock-hand, tick, divider, caret, and search-overlay colors. The
implementation must not retain ad hoc component colors created with direct
`c(...)` calls. Shadows derive from the `shadow` role, and scrims derive from
the `scrim` role.

## Layout

The top application bar retains the MORROW identity, search field, theme
selector, and time-format selector. Search results continue to render in the
existing high z-index queue so that results always appear above clock content.

The main clock area has these proportions:

- The digital clock takes about 70 percent of the available width.
- The analog clock takes about 30 percent of the available width.
- Hours and minutes carry the greatest type size.
- Seconds remain visible but use a smaller size and a quieter color role.
- City, country, UTC offset, date, and daylight status remain close to the
  clock without competing with it.

The 24-hour track sits directly below the main clocks. It must show the current
local-time position and distinguish night, transition, and daylight ranges
with Monet tertiary tones. The track is a time component, not a decorative
sunset stripe.

Saved cities use compact continuous-corner cards. The selected card uses the
primary-container roles. Other cards use low-emphasis surface containers.
Cards must remain visually subordinate to the main clock.

## Interaction behavior

The redesign preserves the current hit-testing and interaction model.

- Moving the pointer updates hover feedback.
- Pressing and releasing the same target activates it.
- Searching accepts city, country, and UTC text.
- Pressing Enter selects the first search result.
- Pressing Escape clears and closes search.
- Selecting a city updates both clocks and the daylight track.
- Theme and 12-hour or 24-hour selectors update immediately.
- Adding and removing saved cities follows the existing limits and fallback
  behavior.

Controls must provide clear hover, pressed, selected, and disabled states.
Interactive hit areas must be at least 48 design pixels where layout permits,
especially for remove and icon actions.

## Implementation boundaries

Implementation stays in the existing example and its CMake target.

- Replace the current `Palette` fields with semantic Monet roles.
- Rewrite `draw_dashboard()` to implement the approved hierarchy.
- Refine `analog_clock()` for better tick spacing, hand proportions, and
  visual integration with the main digital clock.
- Add a focused daylight-track drawing helper.
- Rewrite `city_card()` as a lower-emphasis supporting component.
- Preserve the search overlay's separate z-layer.
- Preserve the existing city data, event handling, and proportional scaling.
- Remove `EUI-NEO` and `TimeSpot` from the source filename, executable target,
  related CMake variables and compile definitions, window title, visible UI
  copy, and runtime messages. Rename each product-facing identifier to a
  MORROW-specific name.
- Retain any legally required Apache License 2.0 or original-source attribution
  in a non-product source comment or notice. Attribution must not appear as the
  demo's product identity.

The implementation must not modify unrelated dirty files or discard existing
user changes.

## Error handling

The demo keeps its current safe fallback behavior.

- If the embedded display font fails to load, log the failure and use the
  system fallback.
- If all saved cities are removed, restore Beijing as the fallback city.
- If search has no result, keep the existing selection and show no result
  menu.
- If the selected city is removed, select the first remaining saved city.

## Verification

Verification covers build success, runtime initialization, interaction, and
visual hierarchy.

1. Build the renamed demo target with the existing CMake build directory.
2. Run the executable and confirm that WebGPU and the embedded font initialize.
3. Verify city search, Enter selection, Escape clearing, and search overlay
   ordering.
4. Verify city selection, add, remove, fallback selection, and the saved-city
   limit.
5. Verify light and dark Monet themes.
6. Verify 12-hour and 24-hour formats.
7. Confirm that both the digital and analog clocks update from the selected
   city's local time.
8. Confirm that the daylight track moves with the selected local time.
9. Inspect the 1600 by 1000 window and a smaller proportional window for
   clipping, text collisions, and incorrect hit regions.
10. Confirm that the digital clock carries about 70 percent of the visual
    emphasis and the analog clock carries about 30 percent.

## Next steps

After this specification passes review and receives user approval, create an
implementation plan, rename the demo, implement the visual redesign, and run
the verification steps.
