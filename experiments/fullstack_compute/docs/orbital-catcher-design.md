# 引力捕手 (Orbital Catcher) — Game Design Specification

## 1. Overview

A 2D particle-catch game built with the fullstack_compute Canvas 2D rendering library.
Players place gravity wells on a grid to attract falling square particles into them, scoring points
while managing well capacity. The game blends cel-shaded aesthetics with vaporwave gradients against a
deep-space starfield.

**Core loop**: Particles spawn from screen edges → Player places gravity wells → Particles are attracted
and captured → Score increases → Wells overload and explode → Repeat, faster and denser over time.

---

## 2. Visual Style

### Color Palette (Vaporwave)

| Name        | Hex       | Use                          |
|-------------|-----------|------------------------------|
| Neon Pink   | #FF6EC7   | Particles, accents            |
| Aqua        | #00F5FF   | Particles, gravity well rings |
| Peach       | #FFB347   | Particles, score flash        |
| Lavender    | #B983FF   | Particles, UI panels          |
| Mint        | #7FFFD4   | Grid preview, success flash    |
| Coral       | #FF7F7F   | Overload warning, explosion   |
| Deep Navy   | #0A0A1A   | Background base              |
| Deep Purple | #1A0A2E   | Background gradient           |

### Background
- Base: solid #0A0A1A
- Layered radial gradient: #0A0A1A → #1A0A2E → #12072B
- Animated starfield: ~150 small dots, random positions, slow drift, varying alpha

### Particles
- Shape: square (cel-shaded)
- Size: 10x10 px in screen space (scales with DPI)
- Rendering: fill with flat color + 2px darker stroke outline
- Colors: randomly assigned from palette (excluding background colors)
- Spawn: from all four screen edges, random position along edge
- Motion: linear velocity + gravity-well attraction force
- Trail: last 5 positions drawn with decreasing alpha (vaporwave ghosting)

### Gravity Wells
- Shape: circle (cel-shaded ring)
- Outer ring: 2px stroke, color indicates mode (aqua = attract, coral = repel)
- Inner fill: radial gradient (transparent center → mode color at edge, 30% alpha)
- Pulse animation: ring radius oscillates ±3px at 2 Hz
- Capacity bar: small square segments inside the ring showing fill level

### HUD
- Score panel: top-left, cel-shaded rounded rect, dark fill with neon stroke
- Mode indicator: bottom-right, icon shows attract/repel state
- Wave/level indicator: top-right, current difficulty tier
- Challenge banner: full-width translucent bar when a challenge activates
- Pause overlay: semi-transparent dark panel with PAUSED text

### Explosions (Well Overload)
- Triggered when well reaches capacity and more particles enter
- 12 square fragments per explosion, radiate outward with velocity decay
- Fragments: same color as the well, fade out over 0.5s
- Screen shake: ±4px offset for 0.15s

---

## 3. Grid System

- Grid cell size: 64x64 px (screen space)
- Grid rendered as faint dotted lines (10% alpha aqua)
- Mouse hover: highlight current cell with semi-transparent mint fill
- Well placement preview: ghost well at hovered cell
- Wells snap to grid center
- Maximum simultaneous wells: unlimited (player skill/resource limit is score pressure)

---

## 4. Physics

### Particle Spawning
- Spawn edge: random of 4 (top, bottom, left, right)
- Spawn position: random point along chosen edge
- Initial velocity: directed toward screen center ± 30° random spread
- Base speed: 80 px/s, increases 5% per wave

### Gravity Attraction
- Formula: F = strength / (dist^1.5) (softened gravity)
- Attract strength: 8000 (attract mode)
- Repel strength: 5000 (repel mode, pushes particles away)
- Influence radius: 300 px
- Force applied per frame (fixed timestep 1/60s)

### Collision Detection
- Particle enters well when distance to well center < well radius (40px)
- On entry: particle removed, well capacity incremented, score +10 (+ combo multiplier)
- On overload: well removed, explosion spawned

---

## 5. Game Flow

### Mixed Mode (Main + Challenges)

**Wave Progression:**

```
Wave 1:  interval=4.0s, particles_per_wave=5,  capacity=10
Wave 2:  interval=3.5s, particles_per_wave=8,  capacity=12
Wave 3:  interval=3.0s, particles_per_wave=12, capacity=14
...
Wave N:  interval=max(1.0s, 4.0 - N*0.15), particles=max(50, 5+N*3), capacity=min(30, 10+N)
```

**Challenge System:**
- Triggers at score thresholds: 500, 1500, 3000, 5000, 8000, 12000...
- Each challenge has a target: e.g. "Capture 50 particles in 20 seconds"
- On challenge start: full-screen banner, brief 3-2-1 countdown
- On challenge success: all particle spawns pause for 5 seconds (breathing room), bonus score
- On challenge fail: no penalty, game continues

**Combo System:**
- Consecutive captures within 0.5s build combo multiplier (1x → 2x → 3x → max 5x)
- Combo resets if no capture for 2 seconds
- Score per particle: 10 × combo_multiplier

**Game Over:**
- Triggered when particle count on screen exceeds 100 (performance safety) OR
- Triggered when 3 wells explode in a single wave (pressure too high)
- Show final score, high score comparison, "Press R to restart"

---

## 6. Input

| Input            | Action                               |
|------------------|--------------------------------------|
| Mouse move       | Grid hover preview                   |
| Left click       | Place gravity well at hovered cell    |
| Right click      | Toggle attract/repel mode             |
| Space            | Toggle attract/repel mode            |
| ESC              | Toggle pause                         |
| R                | Restart (only on game over screen)   |

---

## 7. Rendering Pipeline (per frame)

1. Clear / draw background gradient + starfield
2. Draw grid dots
3. For each particle: draw trail → draw particle square with stroke
4. For each gravity well: draw inner gradient → draw capacity segments → draw outer ring
5. Draw placement preview (ghost well at hover cell)
6. Draw explosions (fragments)
7. Draw HUD panels (score, wave, mode, combo)
8. Draw challenge banner (if active)
9. Draw pause overlay (if paused)
10. Draw game-over screen (if game over)

---

## 8. Technical Constraints

- Target: 60 FPS on integrated graphics
- Max particles on screen: 200 (pool-based, recycled)
- Max gravity wells: 50 (pool-based)
- Max explosion fragments: 60 active at once
- Particle physics: simple Euler integration, fixed timestep
- No audio (out of scope for initial version)

---

## 9. File Structure

```
examples/
  orbital_catcher.c       # Main game file

  (reuses existing fullstack_glfw_backend.h for window/input)
  (reuses existing fullstack_core.h for rendering)
```

---

## 10. Success Criteria

- Game runs at 60 FPS with 100+ particles on screen
- All visual elements match the cel-shaded + vaporwave + space aesthetic
- Grid snapping works correctly with visible hover preview
- Attract/repel mode toggle is instant and visually clear
- Challenge system triggers correctly at score thresholds
- Score/combo/wave HUD is readable at all times
- Explosion effects play on well overload
- Game over and restart flow works correctly
