# Window Arena survivor progression implementation plan

This plan implements the approved survivor-progression specification through
three continuously playable milestones. Each task preserves the backend-neutral
App boundary and keeps game code under `examples/` and `test/`.

> **Note:** This is an implementation plan for an experimental game demo.

## Baseline

Before editing gameplay, build the current Window Arena target and run its
headless world tests, Mock integration tests, GLFW smoke test, and SDL3 smoke
test. Record failing tests as pre-existing and do not modify unrelated dirty
worktree files.

## Milestone 1: system foundation

### Add stable content descriptors

Create `examples/support/window_arena_content.h` and
`examples/support/window_arena_content.c`.

- Define stable IDs for modes, operators, weapons, systems, evolutions,
  enemies, windows, biomes, modifiers, bosses, and upgrades.
- Add descriptor tables for the complete approved content set.
- Add validation for duplicate IDs, invalid references, invalid levels, and
  evolution pairs.
- Add content validation tests under `test/`.

### Add run progression

Create `examples/support/window_arena_progression.h` and
`examples/support/window_arena_progression.c`.

- Implement level, integer experience, 4+4 inventory, weighted offers,
  fallback cards, reroll, skip, evolution cores, and evolved-slot replacement.
- Keep offer generation deterministic for a supplied random stream.
- Add tests for first-run pool completeness, no duplicate cards, maxed items,
  fewer-than-three choices, core storage, core overflow, and every evolution.

### Add the run director

Create `examples/support/window_arena_director.h` and
`examples/support/window_arena_director.c`.

- Implement five acts, standard final-boss transition, endless cycle
  transition, threat budget, role history, and deterministic spawn quadrants.
- Represent placement requests as backend-neutral intents.
- Add accelerated timeline and deterministic-seed tests.

### Add the versioned profile

Create `examples/support/window_arena_profile.h` and
`examples/support/window_arena_profile.c`.

- Implement the exact initial unlock set and 24 reward bundles.
- Resolve the Windows user-data directory and a portable fallback.
- Write a temporary profile and replace only after successful completion.
- Recover corrupt or unknown profiles without preventing a run.
- Add round-trip, corruption, migration, and reward-bundle tests.

### Extend the world

Update `examples/support/window_arena_world.h` and
`examples/support/window_arena_world.c`.

- Add title, running, level-up, pause, boss-intro, victory, and game-over states.
- Replace one-shot rift population with director-owned encounters.
- Add fragment entities, deterministic overflow merging, pending banks,
  overlap siphoning, and clear settlement.
- Add weapon and system runtime state while retaining fixed-step authority.
- Preserve existing overlap projectile transfer and window-pressure mechanics.

### Integrate the main demo

Update `examples/multi_window_arena_demo.c` and `examples/CMakeLists.txt`.

- Render the title screen and mode selection before any child window exists.
- Remove the **Create rift** button and its hit handling.
- Create warned child windows from director intents at varied safe positions.
- Render responsive HUD, fragment links, level-up cards, and results.
- Extend automated mode to select standard, create varied windows, choose an
  upgrade, and reach an accelerated victory.

## Milestone 2: boss window stage system

### Add window spells

Create `examples/support/window_arena_window_spells.h` and
`examples/support/window_arena_window_spells.c`.

- Implement telegraph, commit, hold, recover, cancel, and reconcile states.
- Quantify work-area margins, coverage, minimum size, visibility, movement
  speed, write frequency, and reduced-motion behavior.
- Keep world intents separate from observed App geometry.
- Add deterministic safety and lifecycle tests.

### Add boss encounters

Extend content and world modules with Prism Battery, Window Weaver, and Desktop
Devourer state machines.

- Implement every approved base spell and phase threshold.
- Add temporary summon-window budgeting and cleanup.
- Freeze spells during pause, level-up, main minimization, and system drag.
- Restore a safe layout before victory or cycle advancement.
- Add accelerated boss phase tests and Mock geometry disagreement tests.

### Apply native window spells

Update `examples/multi_window_arena_demo.c`.

- Resolve work-area-safe intents and apply them with App window controls.
- Return observed geometry and failures to the spell runtime.
- Never focus, activate, or raise a window as an attack.
- Add GLFW and SDL3 automated boss smoke paths.

## Milestone 3: complete content and endless progression

### Complete combat content

Implement all eight weapons, eight systems, eight evolutions, eight enemies,
five role windows, three biomes, and five elite modifiers using descriptor IDs.
Add focused headless tests for each behavior and pairing.

### Complete profile content

Add four operators, four starting loadouts, 24 challenge evaluators, archive
records, three alternate boss spell sets, and eight endless modifiers. Verify
that each locked gameplay ID appears in exactly one reward bundle.

### Add procedural audio

Vendor miniaudio under `third_party/miniaudio/` with license metadata. Create
`examples/support/window_arena_audio.h` and
`examples/support/window_arena_audio.c`.

- Generate procedural effects for weapons, fragments, upgrades, warnings,
  windows, bosses, and results.
- Queue audio events without changing deterministic simulation.
- Degrade to silence when initialization or the device fails.
- Add no-device unit tests.

### Complete endless mode

Implement cycle scaling, all eight modifiers, boss spell combinations, cycle
recovery, separate cycle and total timers, and two-cycle accelerated tests.

## Final verification

Run these checks after all milestones:

1. Configure and build the active debug and release directories.
2. Run all Window Arena unit, world, profile, spell, and Mock tests.
3. Run the complete repository test suite.
4. Run GLFW and SDL3 standard and endless automated smoke tests.
5. Run the architecture layer audit and `git diff --check`.
6. Build `package_window_arena_win64` and run isolated package verification.
7. Confirm title mode selection, varied spawn positions, 4+4 progression,
   evolution, all bosses, layout restoration, and silent audio fallback.

## Commit sequence

Commit descriptor and headless systems before UI integration. Commit the
window-spell runtime before native boss integration. Commit miniaudio and its
license separately from game behavior. Do not include unrelated worktree files
in any commit.
