# Window Arena survivor progression design

This specification turns the existing Window Arena technology demo into a
20-to-30-minute survivor-style game with an endless mode, automatic enemy
windows, run-based builds, restrained meta progression, varied enemy roles,
and bosses that use native window movement and resizing as readable attacks.
It preserves the existing backend-neutral App boundary and deterministic
world simulation.

> **Note:** This is an experimental game demo and App integration under active
> development.

## Goals

The finished demo must provide these player-facing outcomes:

- A title screen lets the player choose a 25-minute standard expedition or an
  endless protocol before any child windows are created.
- The current manual **Create rift** button is removed.
- A run director automatically creates varied enemy windows at safe,
  non-repetitive desktop positions after a readable warning.
- The player retains mouse-aimed primary fire and gains automatic weapons,
  area effects, orbitals, deployables, and cross-window attacks.
- Kills produce experience fragments inside enemy windows. Overlapping a
  window with the main window creates a siphon link, and clearing a window
  settles its remaining fragments.
- Level-up pauses the complete world, dims and freezes child windows, and
  presents three upgrade cards in the main window.
- Each run supports four weapon modules, four system modules, and paired
  weapon evolutions.
- Standard expedition follows a five-act, 25-minute escalation and ends with
  a final boss.
- Endless mode repeats 25-minute protocol cycles with stronger enemies,
  combined boss attacks, and global modifiers.
- Normal encounters use role-based windows, biome windows appear as uncommon
  rule changes, and bosses use native windows as active attacks.
- Meta progression primarily unlocks content and play styles. Permanent raw
  power remains small enough that run decisions determine success.
- GLFW and SDL3 execute identical game logic through App APIs.

## Non-goals

This feature does not add multiplayer, network synchronization, external
mod loading, a general scripting language, or game-specific code to root
`src/`, root `include/`, or `app/`. It does not copy Vampire Survivors,
20 Minutes Till Dawn, Windowkill, or Halls of Torment names, art, sounds,
numerical balance, level layouts, or source code.

The first completed version does not require sprite sheets. It keeps the
existing procedural radar aesthetic and existing Oxanium font. It does not
make arbitrary third-party windows part of the game and never controls windows
outside the Window Arena process.

## Research basis

The design adapts general patterns observed in several games and design
references:

- The official Vampire Survivors wiki documents experience-based level-up,
  three or four weighted upgrade choices, limited weapon and passive slots,
  timed stage waves, boss chests, and paired evolution requirements.
- 20 Minutes Till Dawn combines manually aimed shooting with branching
  upgrade trees and prerequisite-based synergies. This supports retaining
  Window Arena's aiming instead of converting the game to fully automatic
  combat.
- Windowkill demonstrates that separate boss windows and window boundaries can
  be combat mechanics. Window Arena extends this idea through a backend-neutral
  intent and safety system rather than reproducing Windowkill content.
- Halls of Torment uses challenge completion to unlock characters, items,
  abilities, and traits. Window Arena uses the same broad content-unlock
  principle while avoiding large permanent damage multipliers.
- The Level Design Book recommends enemy roles with distinct speed, health,
  range, behavior, silhouettes, and complementary combinations. Window Arena's
  roster follows this orthogonal-role approach.

Reference pages:

- <https://vampire.survivors.wiki/w/Level_up>
- <https://vampire.survivors.wiki/w/Evolution>
- <https://vampire.survivors.wiki/w/Stages>
- <https://20minutestilldawn.wiki.gg/wiki/Upgrades>
- <https://windowkill.wiki.gg/wiki/Bosses>
- <https://torcado.itch.io/windowkill>
- <https://book.leveldesignbook.com/process/combat/enemy>

## Player flow

The application opens in a single decorated main window. The title screen
contains these choices:

- **Standard expedition** starts a five-act run that ends at 25 minutes.
- **Endless protocol** starts the same opening cycle and continues after each
  25-minute boss.
- **Operator** selects an unlocked character trait package.
- **Initial weapon** selects one unlocked starting weapon.
- **Archive** shows unlocked content, discovered evolutions, challenges, and
  historical statistics.
- **Settings** includes input, volume, interface scale, and reduced native
  window motion.

Starting a run resets run-only state, preserves the selected profile, and
lets the director schedule the first warning. No child window exists on the
title screen.

The repeating combat loop is:

1. The director spends a threat budget on a window role, spawn location, and
   complementary enemy group.
2. A desktop-space warning previews the future window bounds and role.
3. The coordinator creates the window hidden, applies appearance and geometry,
   renders its initial frame, and shows it.
4. Combat generates experience fragments and rewards inside the owner window.
5. An overlap with the main window creates a visible siphon lane that transfers
   fragments over time.
6. Clearing the window settles all remaining fragments and starts its collapse.
7. Level-up freezes simulation and native window spells until the player
   selects one of three upgrade cards.
8. Elite and boss clears drop evolution cores and unlock-progress rewards.

## Run modes and pacing

Standard expedition lasts 25 minutes and uses five acts.

### Act I: Intrusion, 00:00 to 05:00

Nest windows introduce swarm and tracker enemies. The first encounters teach
overlap, siphoning, and automatic window creation. A Shard Matron elite appears
at 05:00.

### Act II: Crossfire, 05:00 to 10:00

Battery windows introduce snipers, artillery, and protectors. Enemy fire begins
crossing overlapping windows. The Prism Battery boss appears at 10:00.

### Act III: Distortion, 10:00 to 15:00

One randomly selected biome protocol changes the act's movement, projectiles,
or fragment flow. A Protocol Corruptor elite appears at 15:00.

### Act IV: Siege, 15:00 to 20:00

Fortress and summoning windows combine tanks, commanders, and temporary
reinforcement windows. The Window Weaver boss appears at 20:00.

### Act V: Collapse, 20:00 to 25:00

The director mixes every learned role, increases native window-spell pressure,
and tests the completed build. The Desktop Devourer appears at 25:00.

The player must usually establish a first weapon combination in the opening
three minutes, complete a first evolution between 10 and 15 minutes, and
stabilize a full build before the final act.

Endless mode treats the opening 25 minutes as protocol cycle one. Each later
cycle increases health, speed, density, and window-spell budget, and selects a
global modifier such as Rift Multiplication, Projectile Refraction, Persistent
Biome, or Fragment Decay. Bosses combine unlocked spells instead of replaying
an unchanged script.

## Run director

`WA_RunDirector` owns elapsed run time, act, cycle, threat budget, spawn
cooldowns, recent encounter history, and the deterministic director random
stream. It does not own entities or native windows.

The director selects encounters from validated content descriptors. A valid
encounter contains a front-line role, a back-line or support role where the
act permits one, a window role, and a desktop-space objective. It must not
spawn the same window role consecutively unless a boss script explicitly
requires it.

The placement solver receives current work areas and observed game-window
geometry. It samples positions around the main window, rejects positions that
would be fully hidden, mostly cover the main window, leave the work area, or
repeat the recent spawn region, and selects the highest-scoring valid result.
If no candidate exists, the encounter becomes an event inside an existing
window instead of failing the run.

The director may reduce cosmetic density and ordinary spawn budget when frame
time or active-window count exceeds a configured limit. It must not remove a
boss warning, critical projectile, player weapon, or required encounter goal.

## Window ecology

The normal active limit is one main window plus four child windows. The solver
may lower the child limit to two or three on a small work area. A boss can
temporarily exceed the normal limit with up to three small, short-lived summon
windows. Summon windows collapse immediately when cleared or when their spell
ends.

### Role windows

Role windows form the normal encounter ecology:

- **Nest** produces fast groups and rewards area coverage.
- **Battery** houses ranged attackers and cross-window projectiles.
- **Fortress** protects commanders, artillery, or stored fragments.
- **Distortion** changes overlap, pressure, visibility, or projectile rules.
- **Vault** is a reward encounter that releases fragments or an evolution core
  after its guards are defeated.

### Biome windows

Biome windows are uncommon act-level modifiers rather than ordinary random
encounters:

- **Ember protocol** adds burning zones and accelerating projectiles.
- **Frost protocol** slows entities and makes fragment siphoning arrive in
  pulses.
- **Void protocol** adds paired transfer regions and displaced projectiles.

The director selects at most one primary biome for an act in the first cycle.
Endless modifiers may permit combinations in later cycles.

## Experience and level-up

Enemies drop data fragments into the enemy's owner window. Fragments remain
world entities until collected, transferred, settled, or removed by an
explicit gameplay effect. They do not immediately increase global experience.

When an enemy window overlaps the main window, the overlap becomes a siphon
lane. Fragments accelerate toward the overlap, transfer to the main window,
and then move toward the player. Link Amplifier, Magnet Core, biome rules, and
enemy disruption can change transfer rate. Clearing a window awards its
remaining fragments immediately so the player is not required to preserve an
empty window.

Level-up changes the world to `LEVEL_UP_PAUSED`, freezes fixed-step simulation,
native window spells, opacity transitions, and hostile movement, and dims all
child windows. The main window shows three non-duplicate eligible choices.
Keyboard keys 1, 2, and 3 and pointer selection are supported. Reroll and skip
uses are finite run resources. After selection, the world resumes and grants
0.8 seconds of player invulnerability.

## Build system

A run contains four weapon slots and four system slots. An upgrade can add an
eligible item to an empty slot or increase an owned item's level. Maxed items
are removed from ordinary offers. Weighted selection favors owned items enough
to let a build mature without making every run deterministic.

The initial content pool contains these weapons:

- Pulse Lance: aimed, fast, straight primary fire.
- Scatter Array: aimed, short-range fan fire.
- Orbit Drones: automatic close defense.
- Arc Relay: automatic chaining across linked windows.
- Gravity Well: periodic area damage and grouping.
- Boundary Saw: follows window edges and punishes edge pressure.
- Rift Beam: charged, piercing, cross-window precision attack.
- Sentry Node: deploys inside overlap lanes.

The initial system pool contains these modules:

- Overclocker: attack rate and cooldown.
- Resonance Lens: area and duration.
- Link Amplifier: cross-window damage and chain count.
- Magnet Core: fragment attraction and siphon rate.
- Reactive Armor: health, shield, and hit recovery.
- Stabilizer: shrink resistance and geometry recovery.
- Echo Kernel: projectile count, split, and repetition.
- Salvage Protocol: drop quality, rarity, and reroll economy.

A weapon and its paired system must reach their required level before an elite
or boss evolution core can transform them. The initial evolutions are:

- Pulse Lance plus Overclocker becomes Phase Gatling.
- Scatter Array plus Echo Kernel becomes Fractal Barrage.
- Arc Relay plus Link Amplifier becomes Thunder Network.
- Gravity Well plus Resonance Lens becomes Event Horizon.
- Orbit Drones plus Reactive Armor becomes Aegis Swarm.
- Boundary Saw plus Stabilizer becomes Boundary Engine.
- Rift Beam plus Magnet Core becomes Siphon Ray.
- Sentry Node plus Salvage Protocol becomes Replicator Node.

## Enemy roster

Enemy types use distinct silhouette, color, speed, health, range, and behavior.
The initial roster contains:

- **Shards** are fragile, fast swarms that test area coverage.
- **Trackers** provide predictable close-range pressure and the balance baseline.
- **Bulwarks** are slow, durable front-line blockers that protect ranged units.
- **Needlers** telegraph long-range precision attacks and are vulnerable while
  charging.
- **Bombardiers** target another window with delayed area markers.
- **Siphon Leeches** consume untransferred experience fragments.
- **Commanders** buff nearby enemies and alter spawn cadence.
- **Corruptors** temporarily modify pressure, opacity, or overlap rules after a
  clear warning.

Elite modifiers are Mirror, Armored, Frenzied, Draining, and Splitting. An
elite receives at most one primary modifier in the initial cycle. The director
combines roles deliberately instead of independently selecting every enemy.

## Boss window spells

Every native window spell passes through these phases:

```text
telegraph -> commit -> hold -> recover
```

The telegraph lasts 0.8 to 1.5 seconds for geometry-changing attacks. Commit
interpolates native geometry with bounded speed. Hold runs the attack. Recover
returns affected windows to a safe layout. Pausing, level-up, application
minimization, display reconfiguration, or an active system drag freezes or
cancels native geometry commits.

### Prism Battery

The 10-minute boss moves its own window through sampled positions around the
main window. Refraction Beam draws a desktop-space line before firing across
windows. Mirror Summon creates destructible temporary reflector windows. At
half health, Phase Rebuild changes the boss window's aspect ratio and attack
cadence.

### Window Weaver

The 20-minute boss rearranges ordinary enemy windows into overlap webs. Lane
Compression narrows the main window from two warned sides while preserving a
safe corridor. Hatch creates one to three temporary enemy windows. Reverse
Link lets enemies or hostile projectiles use overlap lanes for a limited time.

### Desktop Devourer

The final boss can move and resize the main window within the safety contract.
Bite reduces or shifts the playable area after a clear preview. Consume
overlaps an enemy window and inherits its role attack. Quadrant Split creates
moving jaw windows around the main window. Final Collapse combines spells from
the earlier bosses. Defeat restores the pre-boss desktop layout before the
victory screen.

## Window safety contract

The world emits `WA_WindowSpellIntent` values and never calls App window APIs.
The window-spell runtime resolves each intent against observed work areas,
window geometry, the activity limit, and accessibility settings. The App
coordinator applies approved geometry and reports observed results back to the
world.

The runtime enforces these invariants:

- The main window remains at or above the safe combat size.
- Every game window retains a visible recovery or grab area.
- No game window leaves its selected work area or covers reserved taskbar
  bounds.
- Game code never activates, focuses, or raises a window as part of an attack.
- Native geometry writes have speed and frequency limits.
- Manual or window-manager-adjusted geometry is authoritative.
- A failed solve cancels or degrades the spell; it never applies a partial
  unsafe layout.
- Reduced native window motion replaces large translations with canvas motion
  and slower, smaller geometry changes.

## Interface

The HUD shows health, run timer, act, level, experience, active weapon modules,
threat, active window count, and siphon rate. A narrow window keeps only health,
timer, level, and critical warning information. Color, shape, and text jointly
identify modules and threats.

The level-up screen shows item category, current or new status, resulting
numbers, tags, and known evolution partners. It becomes a vertical list when
the main window is narrow.

The result screen shows run time, score, build, defeated bosses, completed
challenges, newly unlocked content, and discovered evolutions. Returning to
the title screen does not create child windows.

## Meta progression and profile

Challenges unlock operators, weapons, systems, biomes, elite modifiers, boss
variants, starting loadouts, rerolls, and archive entries. Small convenience
upgrades may affect initial rerolls, fragment visibility, or starting health,
but the profile does not contain an unbounded permanent damage ladder.

The versioned profile stores unlock bits, challenge progress, settings, and
statistics under the user's data directory. Windows uses:

```text
%LOCALAPPDATA%\WCN\WindowArena\profile.bin
```

The writer creates a temporary file and replaces the previous profile only
after a complete write. A corrupt or unsupported file is renamed with a
`.bad` suffix and replaced with a default profile. Profile failure never
prevents a run from starting.

## Audio and visual assets

The game keeps the original procedural desktop-rift visual language: near-black
blue fields, cyan player and links, amber conventional threats, magenta rifts,
violet biome changes, and red critical attacks. Enemies receive distinct
procedural silhouettes and telegraph animations. The existing Oxanium font and
its SIL Open Font License remain under `examples/assets/window_arena/`.

Audio uses a vendored miniaudio dependency under root `third_party/miniaudio/`
with its license metadata. `examples/support/window_arena_audio.[ch]` exposes a
small game-specific interface independent of the selected window backend.
Procedural oscillators, envelopes, noise, and filters generate weapon,
fragment, level-up, warning, window-spawn, boss, and result sounds at runtime.
Audio initialization or device loss degrades to silence without affecting
simulation.

## Code organization

Game changes remain under `examples/` and `test/`:

```text
examples/support/window_arena_content.[ch]
examples/support/window_arena_director.[ch]
examples/support/window_arena_progression.[ch]
examples/support/window_arena_window_spells.[ch]
examples/support/window_arena_profile.[ch]
examples/support/window_arena_audio.[ch]
examples/support/window_arena_world.[ch]
examples/support/window_arena_protocol.h
examples/multi_window_arena_demo.c
test/window_arena_*.c
third_party/miniaudio/
```

`window_arena_content` owns stable IDs and validated constant descriptor tables.
`window_arena_director` owns timing and threat selection. Progression owns
experience, inventory, offers, and evolution. Window spells own native-window
intent state but not App handles. The world remains the sole simulation
authority. The demo remains the App and rendering coordinator.

The feature does not require changes to root Core or the App public API. It
uses existing window creation, geometry, decoration, opacity, message, event,
and scheduling operations.

## Protocol changes

The fixed-layout, pointer-free protocol adds messages or action payloads for:

- run start, mode, phase, pause, victory, and game over;
- automatic window spawn request and observed result;
- level-up ready, upgrade selection, and upgrade applied;
- boss window-spell intent, application result, cancellation, and recovery;
- experience settlement and profile unlock notifications.

Content IDs, ticks, source epochs, and spell IDs cross the boundary. Native
handles, pointers, descriptor pointers, and file paths do not.

## Failure handling

The implementation handles expected failure without corrupting the run:

- Child-window creation failure converts the encounter into an existing-window
  event and refunds the unspent window budget.
- An unsafe or rejected boss geometry intent is re-solved once, then cancelled
  into recovery.
- Display topology changes cancel active native movement, clamp all windows,
  and resume only after observed geometry stabilizes.
- Minimizing the main window pauses the world. Minimizing or manually closing
  an enemy window grants no clear reward and migrates remaining threat after a
  grace period.
- Capacity pressure removes decorative particles first, then ordinary hostile
  projectiles according to deterministic priority. It never silently removes
  the player, boss, warning, upgrade, or critical spell state.
- App, window, GPU, content, profile, and audio errors are written to
  `window-arena.log` and `stderr`. Recoverable errors also produce a short
  in-game notice.

## Capacity and performance

The initial hard ceilings are:

```text
1 main window
4 normal child windows
3 temporary boss summon windows
256 enemies
768 projectiles
512 experience fragments
512 decorative particles
```

High-frequency objects use fixed pools. The fixed-step simulation performs no
per-tick heap allocation. Static, minimized, occluded, and unchanged windows
retain the existing on-demand App rendering behavior. The director may lower
ordinary density based on activity and frame-time pressure while preserving
gameplay-critical entities and deterministic selection order.

## Delivery milestones

The complete scope is delivered through three continuously playable milestones.
The milestones are sequencing boundaries, not reductions in final scope.

### Milestone A: complete run loop

Add title and result states, both run modes, the director, automatic window
placement, experience siphoning, level-up, the 4+4 build, content descriptors,
the base roster, responsive HUD, profile foundation, and accelerated run tests.

### Milestone C: boss window stage system

Add the reusable spell lifecycle, safety solver, native geometry reconciliation,
reduced-motion behavior, temporary summon windows, and the three complete boss
encounters.

### Milestone B: full content and progression

Complete all eight weapons, eight systems, eight evolutions, eight enemy roles,
five role windows, three biomes, five elite modifiers, operators, challenges,
archive, procedural audio, balance passes, and endless combinations.

## Verification

Unit tests must cover content ID uniqueness and references, evolution pairing,
weighted offer eligibility, slot limits, reroll and skip behavior, fragment
siphoning and settlement, director determinism, encounter-history rules,
placement scoring, spell lifecycle, and every window-safety invariant.

Profile tests must cover default creation, round trips, interrupted temporary
writes, corrupt data, unknown versions, and migration. Audio tests must cover
silent fallback and command generation without requiring a physical device.

Headless world tests must accelerate a complete standard expedition and at
least two endless cycles with fixed seeds. They must enter every act, trigger
every boss phase, complete an evolution, and finish without capacity overflow
or non-finite state.

Mock App integration tests must cover creation failure, manual close,
minimization, display changes, observed geometry disagreement, level-up during
a spell, pause, recovery, and normal plus temporary window limits.

GLFW and SDL3 runtime automation must select each mode from the title screen,
observe varied automatic spawn positions, transfer fragments, select an
upgrade, execute representative boss geometry spells, restore layout, and
shut down without leaked windows. Packaged Win64 smoke tests must run both
backends from an isolated directory.

The architecture audit must confirm that root Core and App contain no Window
Arena gameplay, profile, content, audio, or boss dependencies.

## Next steps

Review this specification, then create a milestone-based implementation plan.
Implementation starts with data descriptors and deterministic headless systems
before changing native window behavior or rendering.
