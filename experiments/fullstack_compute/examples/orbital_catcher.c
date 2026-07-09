#include "../impl/fullstack_glfw_backend.h"
#include "../include/fullstack_core_debug.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
   ORBITAL CATCHER — A 2D particle-catch game
   Visual: cel-shaded + vaporwave + deep-space starfield
   ============================================================ */

#define ARRAY_COUNT(x) (sizeof(x) / sizeof((x)[0]))
#define PI 3.14159265358979323846f

/* ------------------------------------------------------------
   Responsive viewport — game world uses actual framebuffer dimensions.
   Element sizes scale proportionally with viewport width.
   ------------------------------------------------------------ */

/* ------------------------------------------------------------
   Color helpers — ABGR packed uint32 (matches demo.c convention)
   ------------------------------------------------------------ */

static uint32_t rgba8(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return ((uint32_t)(a) << 24u) | ((uint32_t)(b) << 16u) |
           ((uint32_t)(g) << 8u)  | ((uint32_t)(r));
}

static uint32_t color_with_alpha(uint32_t c, uint8_t a) {
    return (c & 0x00FFFFFFu) | ((uint32_t)(a) << 24u);
}

static uint32_t darken(uint32_t c, float f) {
    uint8_t r = (uint8_t)((float)(c & 0xFFu) * f);
    uint8_t g = (uint8_t)((float)((c >> 8) & 0xFFu) * f);
    uint8_t b = (uint8_t)((float)((c >> 16) & 0xFFu) * f);
    uint8_t a = (uint8_t)((c >> 24) & 0xFFu);
    return rgba8(r, g, b, a);
}

/* ------------------------------------------------------------
   Constants — Vaporwave palette
   ------------------------------------------------------------ */

enum {
    COLOR_NEON_PINK   = 0xFFC7E6FF,
    COLOR_AQUA        = 0xFFF5E600,
    COLOR_PEACH       = 0xFF47B3FF,
    COLOR_LAVENDER    = 0xFF83B9FF,
    COLOR_MINT        = 0xFFD4FFFF,
    COLOR_CORAL       = 0xFF7F7FFF,
    COLOR_DEEP_NAVY   = 0x1A0A0AFF,
    COLOR_DEEP_PURPLE = 0x2E0A1AFF,
    COLOR_BG3         = 0x2B1207FF,
    COLOR_GRID_DOT    = 0x3300F5FF,
};

static const uint32_t k_particle_colors[] = {
    COLOR_NEON_PINK, COLOR_AQUA, COLOR_PEACH, COLOR_LAVENDER, COLOR_MINT
};

enum {
    GRID_CELL             = 64,
    WELL_RADIUS           = 36,
    PARTICLE_SIZE         = 10,
    MAX_PARTICLES         = 200,
    MAX_WELLS             = 50,
    MAX_FRAGMENTS         = 60,
    MAX_STARS             = 150,
    TRAIL_LEN             = 5,
    FRAGMENTS_PER_EXPLOSION = 12,
    MAX_COMBO             = 5,
    SCORE_PER_PARTICLE    = 10,
};

#define EXPLOSION_DURATION       0.5f
#define SCREEN_SHAKE_DURATION    0.15f
#define SCREEN_SHAKE_AMP         4.0f
#define ATTRACTION_STRENGTH       8000.0f
#define REPEL_STRENGTH           5000.0f
#define INFLUENCE_RADIUS         300.0f
#define COMBO_WINDOW             0.5f
#define COMBO_DECAY              2.0f
#define MAX_WELL_VELOCITY        500.0f
#define BASE_PARTICLE_SPEED       80.0f
#define SPEED_INCREASE_PER_WAVE   0.05f
#define DT                       (1.0f / 60.0f)

/* Challenge definitions */
static const int   k_challenge_scores[]  = {500, 1500, 3000, 5000, 8000, 12000};
static const int   k_challenge_targets[]  = {20,   40,   60,   80,   100,  120};
static const float k_challenge_times[]   = {20.0f, 25.0f, 30.0f, 30.0f, 35.0f, 40.0f};
#define CHALLENGE_COUNT (int)(sizeof(k_challenge_scores) / sizeof(k_challenge_scores[0]))

/* ------------------------------------------------------------
   Data structures
   ------------------------------------------------------------ */

typedef struct Particle {
    float x, y;
    float vx, vy;
    uint32_t color;
    float trail_x[TRAIL_LEN];
    float trail_y[TRAIL_LEN];
    uint8_t trail_len;
    uint8_t alive;
} Particle;

typedef struct GravityWell {
    int gx, gy;         /* grid cell (anchor-independent, stable across resize) */
    float x, y;         /* computed pixel position each frame from gx/gy */
    int capacity;
    int max_capacity;
    float pulse;
    float explode_time;
    uint8_t alive;
} GravityWell;

typedef struct ExplosionFragment {
    float x, y;
    float vx, vy;
    uint32_t color;
    float life;
    float alpha;
    uint8_t alive;
} ExplosionFragment;

typedef struct Star {
    float x, y;
    float speed;
    float alpha;
    float size;
} Star;

/* ------------------------------------------------------------
   Game state
   ------------------------------------------------------------ */

/* Viewport dimensions — game world runs in actual framebuffer pixels.
   Updated every frame from backend.width/height. */
static float g_vw = 1280.0f;   /* viewport width  */
static float g_vh = 720.0f;    /* viewport height */

static Particle         g_particles[MAX_PARTICLES];
static GravityWell     g_wells[MAX_WELLS];
static ExplosionFragment g_fragments[MAX_FRAGMENTS];
static Star            g_stars[MAX_STARS];

static int    g_score          = 0;
static int    g_high_score     = 0;
static int    g_combo          = 1;
static float  g_combo_timer    = 0.0f;
static int    g_wave           = 1;
static float  g_wave_interval  = 4.0f;
static int    g_particles_per_wave = 5;
static int    g_max_capacity   = 10;
static int    g_particles_this_wave = 0;
static float  g_wave_spawn_timer = 0.0f;

static bool   g_attract_mode   = true;
static bool   g_paused         = false;
static bool   g_game_over      = false;

static int     g_explosion_count_this_wave = 0;
static float   g_screen_shake_timer = 0.0f;
static float   g_shake_x = 0.0f, g_shake_y = 0.0f;

static int     g_challenge_idx       = 0;
static int     g_challenge_target    = 0;
static float   g_challenge_timer    = 0.0f;
static int     g_challenge_captured  = 0;
static bool    g_challenge_active    = false;
static float   g_challenge_countdown = 0.0f;
static float   g_challenge_pause_timer = 0.0f;

static float   g_mouse_x = 0.0f, g_mouse_y = 0.0f;
static int     g_hover_gx = 0, g_hover_gy = 0;
static bool    g_prev_lmb = false;

/* Sync well pixel positions from grid cell coordinates.
   Call once per frame before physics/rendering. */
static void sync_well_positions(void) {
    float scale = g_vw / 1280.0f;
    int cell = (int)(GRID_CELL * scale);
    for (int i = 0; i < MAX_WELLS; i++) {
        if (g_wells[i].alive) {
            g_wells[i].x = ((float)g_wells[i].gx + 0.5f) * (float)cell;
            g_wells[i].y = ((float)g_wells[i].gy + 0.5f) * (float)cell;
        }
    }
}

/* ------------------------------------------------------------
   High score I/O
   ------------------------------------------------------------ */

static void load_high_score(void) {
    FILE* f = fopen("orbital_catcher.dat", "rb");
    if (f) {
        uint16_t val = 0;
        if (fread(&val, sizeof(val), 1, f) == 1) {
            g_high_score = (int)val;
        }
        fclose(f);
    }
}

static void save_high_score(void) {
    if (g_score > g_high_score) {
        g_high_score = g_score;
        FILE* f = fopen("orbital_catcher.dat", "wb");
        if (f) {
            uint16_t val = (uint16_t)g_high_score;
            fwrite(&val, sizeof(val), 1, f);
            fclose(f);
        }
    }
}

/* ------------------------------------------------------------
   Pool helpers
   ------------------------------------------------------------ */

static void init_pools(void) {
    for (int i = 0; i < MAX_PARTICLES;   i++) g_particles[i].alive = 0;
    for (int i = 0; i < MAX_WELLS;       i++) g_wells[i].alive = 0;
    for (int i = 0; i < MAX_FRAGMENTS;    i++) g_fragments[i].alive = 0;
}

static void init_stars(void) {
    for (int i = 0; i < MAX_STARS; i++) {
        g_stars[i].x     = (float)((i * 7919) % (int)g_vw);
        g_stars[i].y     = (float)((i * 1031) % (int)g_vh);
        g_stars[i].speed = 0.2f + 0.8f * ((i & 7) / 7.0f);
        g_stars[i].alpha = 0.3f + 0.7f * (((i >> 3) & 7) / 7.0f);
        g_stars[i].size  = 1.0f + 2.0f * ((i % 3) / 2.0f);
    }
}

static void reset_game(void) {
    init_pools();
    g_score          = 0;
    g_combo          = 1;
    g_combo_timer    = 0.0f;
    g_wave           = 1;
    g_wave_interval  = 4.0f;
    g_particles_per_wave = 5;
    g_max_capacity   = 10;
    g_particles_this_wave = 0;
    g_wave_spawn_timer = 0.0f;
    g_attract_mode   = true;
    g_paused         = false;
    g_game_over      = false;
    g_explosion_count_this_wave = 0;
    g_screen_shake_timer = 0.0f;
    g_challenge_idx      = 0;
    g_challenge_target   = 0;
    g_challenge_timer    = 0.0f;
    g_challenge_captured = 0;
    g_challenge_active   = false;
    g_challenge_countdown = 0.0f;
    g_challenge_pause_timer = 0.0f;
}

/* ------------------------------------------------------------
   Particle spawning
   ------------------------------------------------------------ */

static float frand(float lo, float hi) {
    float t = (float)((((unsigned)rand()) & 0xFFFF) / 65535.0f);
    return lo + t * (hi - lo);
}

static void spawn_particle(void) {
    /* Find dead slot */
    Particle* p = NULL;
    for (int i = 0; i < MAX_PARTICLES; i++) {
        if (!g_particles[i].alive) { p = &g_particles[i]; break; }
    }
    if (!p) return;

    const float cx = g_vw * 0.5f, cy = g_vh * 0.5f;
    const float speed = BASE_PARTICLE_SPEED * (float)pow(1.0f + SPEED_INCREASE_PER_WAVE, (float)(g_wave - 1));

    /* Particle size relative to viewport */
    float ps = PARTICLE_SIZE * (g_vw / 1280.0f);

    int edge = rand() % 4;
    float ex = 0, ey = 0, evx = 0, evy = 0;
    switch (edge) {
        case 0: /* top */
            ex = frand(20.0f, g_vw - 20.0f);
            ey = -ps;
            break;
        case 1: /* bottom */
            ex = frand(20.0f, g_vw - 20.0f);
            ey = g_vh + ps;
            break;
        case 2: /* left */
            ex = -ps;
            ey = frand(20.0f, g_vh - 20.0f);
            break;
        case 3: /* right */
            ex = g_vw + ps;
            ey = frand(20.0f, g_vh - 20.0f);
            break;
    }
    p->x = ex;
    p->y = ey;

    float dx = cx - ex;
    float dy = cy - ey;
    float d = sqrtf(dx * dx + dy * dy);
    if (d < 1.0f) d = 1.0f;
    float base_angle = atan2f(dy, dx);
    float spread = frand(-0.5f, 0.5f); /* ±30 degrees */
    float angle = base_angle + spread;
    evx = cosf(angle) * speed;
    evy = sinf(angle) * speed;
    p->vx = evx;
    p->vy = evy;

    p->color    = k_particle_colors[rand() % ARRAY_COUNT(k_particle_colors)];
    p->trail_len = 0;
    p->alive    = 1;
}

/* ------------------------------------------------------------
   Explosion
   ------------------------------------------------------------ */

static void trigger_explosion(GravityWell* w) {
    w->alive      = 0;
    w->explode_time = EXPLOSION_DURATION;
    g_explosion_count_this_wave++;
    g_screen_shake_timer = SCREEN_SHAKE_DURATION;

    for (int i = 0; i < FRAGMENTS_PER_EXPLOSION; i++) {
        ExplosionFragment* f = NULL;
        for (int j = 0; j < MAX_FRAGMENTS; j++) {
            if (!g_fragments[j].alive) { f = &g_fragments[j]; break; }
        }
        if (!f) break;
        float angle = (2.0f * PI / (float)FRAGMENTS_PER_EXPLOSION) * (float)i + frand(0.0f, 0.5f);
        float spd   = frand(80.0f, 200.0f);
        f->x     = w->x;
        f->y     = w->y;
        f->vx    = cosf(angle) * spd;
        f->vy    = sinf(angle) * spd;
        f->color = g_attract_mode ? COLOR_AQUA : COLOR_CORAL;
        f->life  = EXPLOSION_DURATION;
        f->alpha = 1.0f;
        f->alive = 1;
    }
}

/* ------------------------------------------------------------
   Physics update (fixed dt)
   ------------------------------------------------------------ */

static void update_physics(float dt) {
    /* Stars drift */
    for (int i = 0; i < MAX_STARS; i++) {
        g_stars[i].y += g_stars[i].speed * dt * 60.0f;
        if (g_stars[i].y > g_vh) {
            g_stars[i].y = 0.0f;
            g_stars[i].x = frand(0.0f, g_vw);
        }
    }

    if (g_paused || g_game_over) return;

    /* Spawn particles */
    if (g_challenge_countdown <= 0.0f && g_challenge_pause_timer <= 0.0f) {
        g_wave_spawn_timer -= dt;
        if (g_wave_spawn_timer <= 0.0f) {
            for (int i = 0; i < g_particles_per_wave; i++) {
                spawn_particle();
            }
            g_wave_spawn_timer = g_wave_interval;
            g_explosion_count_this_wave = 0;
        }
    } else {
        /* During countdown/pause, still tick timers */
        if (g_challenge_countdown > 0.0f) {
            g_challenge_countdown -= dt;
        }
        if (g_challenge_pause_timer > 0.0f) {
            g_challenge_pause_timer -= dt;
        }
    }

    /* Update particles */
    for (int pi = 0; pi < MAX_PARTICLES; pi++) {
        Particle* p = &g_particles[pi];
        if (!p->alive) continue;

        /* Apply gravity well forces */
        for (int wi = 0; wi < MAX_WELLS; wi++) {
            GravityWell* w = &g_wells[wi];
            if (!w->alive) continue;
            if (w->explode_time > 0.0f) continue; /* exploding wells have no pull */

            float dx = w->x - p->x;
            float dy = w->y - p->y;
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist < 1.0f) dist = 1.0f;
            float inf_r = INFLUENCE_RADIUS * (g_vw / 1280.0f);
            if (dist < inf_r) {
                float strength = g_attract_mode ? ATTRACTION_STRENGTH : -REPEL_STRENGTH;
                float fmag = strength / powf(dist, 1.5f);
                p->vx += (dx / dist) * fmag * dt;
                p->vy += (dy / dist) * fmag * dt;
            }
        }

        /* Velocity damping */
        p->vx *= 0.998f;
        p->vy *= 0.998f;

        /* Clamp velocity */
        float spd = sqrtf(p->vx * p->vx + p->vy * p->vy);
        if (spd > MAX_WELL_VELOCITY) {
            p->vx = (p->vx / spd) * MAX_WELL_VELOCITY;
            p->vy = (p->vy / spd) * MAX_WELL_VELOCITY;
        }

        /* Update position */
        p->x += p->vx * dt;
        p->y += p->vy * dt;

        /* Update trail */
        for (int ti = TRAIL_LEN - 1; ti > 0; ti--) {
            p->trail_x[ti] = p->trail_x[ti - 1];
            p->trail_y[ti] = p->trail_y[ti - 1];
        }
        p->trail_x[0] = p->x;
        p->trail_y[0] = p->y;
        if (p->trail_len < TRAIL_LEN) p->trail_len++;

        /* Check well capture */
        for (int wi = 0; wi < MAX_WELLS; wi++) {
            GravityWell* w = &g_wells[wi];
            if (!w->alive) continue;
            float dx = w->x - p->x;
            float dy = w->y - p->y;
            float dist = sqrtf(dx * dx + dy * dy);
            float capture_r = (float)WELL_RADIUS * (g_vw / 1280.0f);
            if (dist < capture_r) {
                p->alive = 0;
                if (w->explode_time <= 0.0f) {
                    w->capacity++;
                    g_score += SCORE_PER_PARTICLE * g_combo;
                    g_combo = (g_combo < MAX_COMBO) ? (g_combo + 1) : MAX_COMBO;
                    g_combo_timer = COMBO_WINDOW;
                    g_challenge_captured++;
                    if (w->capacity >= w->max_capacity) {
                        trigger_explosion(w);
                    }
                }
                break;
            }
        }

        /* Cull particles that wandered too far off screen */
        if (p->x < -200.0f || p->x > 1480.0f ||
            p->y < -200.0f || p->y > 920.0f) {
            p->alive = 0;
        }
    }

    /* Update well pulse animations and explode timers */
    for (int wi = 0; wi < MAX_WELLS; wi++) {
        GravityWell* w = &g_wells[wi];
        if (!w->alive && w->explode_time > 0.0f) {
            w->explode_time -= dt;
        }
        if (w->alive) {
            w->pulse += dt * 2.0f * PI * 2.0f; /* 2 Hz */
        }
    }

    /* Update explosion fragments */
    for (int fi = 0; fi < MAX_FRAGMENTS; fi++) {
        ExplosionFragment* f = &g_fragments[fi];
        if (!f->alive) continue;
        f->life -= dt;
        f->x += f->vx * dt;
        f->y += f->vy * dt;
        f->vx *= 0.95f;
        f->vy *= 0.95f;
        f->alpha = f->life / EXPLOSION_DURATION;
        if (f->life <= 0.0f) f->alive = 0;
    }

    /* Screen shake */
    if (g_screen_shake_timer > 0.0f) {
        g_screen_shake_timer -= dt;
        float t = g_screen_shake_timer / SCREEN_SHAKE_DURATION;
        float a = SCREEN_SHAKE_AMP * t;
        g_shake_x = frand(-a, a);
        g_shake_y = frand(-a, a);
    } else {
        g_shake_x = g_shake_y = 0.0f;
    }

    /* Combo decay */
    if (g_combo_timer > 0.0f) {
        g_combo_timer -= dt;
        if (g_combo_timer <= 0.0f) {
            if (g_combo > 1) {
                g_combo--;
                g_combo_timer = COMBO_DECAY;
            } else {
                g_combo = 1;
            }
        }
    }

    /* Wave progression */
    if (g_wave_spawn_timer <= 0.0f) {
        g_wave++;
        g_wave_interval    = (4.0f - g_wave * 0.15f < 1.0f) ? 1.0f : (4.0f - g_wave * 0.15f);
        g_particles_per_wave = (5 + g_wave * 3 > 50) ? 50 : (5 + g_wave * 3);
        g_max_capacity      = (10 + g_wave > 30) ? 30 : (10 + g_wave);
        /* Retroactively update existing well capacities */
        for (int wi = 0; wi < MAX_WELLS; wi++) {
            if (g_wells[wi].alive) {
                g_wells[wi].max_capacity = g_max_capacity;
            }
        }
    }

    /* Challenge system */
    if (!g_challenge_active && !g_paused && !g_game_over) {
        for (int ci = g_challenge_idx; ci < CHALLENGE_COUNT; ci++) {
            if (g_score >= k_challenge_scores[ci]) {
                g_challenge_idx      = ci;
                g_challenge_target   = k_challenge_targets[ci];
                g_challenge_timer    = k_challenge_times[ci];
                g_challenge_captured = 0;
                g_challenge_active   = true;
                g_challenge_countdown = 3.0f;
                break;
            }
        }
    }
    if (g_challenge_active) {
        if (g_challenge_countdown > 0.0f) {
            g_challenge_countdown -= dt;
        } else {
            g_challenge_timer -= dt;
            if (g_challenge_captured >= g_challenge_target) {
                g_score += g_challenge_target * 5;
                g_challenge_active      = false;
                g_challenge_pause_timer = 5.0f;
            } else if (g_challenge_timer <= 0.0f) {
                g_challenge_active = false;
            }
        }
    }

    /* Game over check */
    {
        int alive_count = 0;
        for (int i = 0; i < MAX_PARTICLES; i++) {
            if (g_particles[i].alive) alive_count++;
        }
        if (alive_count >= 100 || g_explosion_count_this_wave >= 3) {
            g_game_over = true;
            save_high_score();
        }
    }

    /* Save high score on improvement */
    save_high_score();
}

/* ------------------------------------------------------------
   Rendering helpers
   ------------------------------------------------------------ */

static void draw_background(FS_Core* core) {
    fs_cmd_rect(core, 0.0f, 0.0f, g_vw, g_vh, 0.0f, COLOR_DEEP_NAVY);
    /* Radial glow center */
    float cx = g_vw * 0.5f, cy = g_vh * 0.5f;
    float gw = g_vw * 0.25f, gh = g_vh * 0.56f;
    fs_cmd_rect(core, cx - gw * 0.5f, cy - gh * 0.5f, gw, gh, 0.0f,
                color_with_alpha(COLOR_DEEP_PURPLE, 90));
    fs_cmd_rect(core, cx - gw * 0.25f, cy - gh * 0.35f, gw * 0.5f, gh * 0.78f, 0.0f,
                color_with_alpha(COLOR_BG3, 60));
}

static void draw_stars(FS_Core* core) {
    for (int i = 0; i < MAX_STARS; i++) {
        Star* s = &g_stars[i];
        uint32_t c = color_with_alpha(COLOR_AQUA, (uint8_t)(s->alpha * 200.0f));
        fs_cmd_rect(core, s->x, s->y, s->size, s->size, 0.0f, c);
    }
}

static void draw_grid(FS_Core* core) {
    /* Grid cell size scales with viewport */
    int cell = (int)(GRID_CELL * (g_vw / 1280.0f));
    float dot_size = 2.0f * (g_vw / 1280.0f);

    for (int gx = 0; gx <= (int)g_vw; gx += cell) {
        for (int gy = 0; gy <= (int)g_vh; gy += cell) {
            fs_cmd_rect(core,
                        (float)gx + dot_size * 0.5f, (float)gy + dot_size * 0.5f,
                        dot_size, dot_size, 0.0f,
                        color_with_alpha(COLOR_GRID_DOT, 40));
        }
    }
    /* Hover highlight */
    if (!g_paused && !g_game_over) {
        float hx = (float)g_hover_gx * (float)cell;
        float hy = (float)g_hover_gy * (float)cell;
        fs_cmd_rect(core, hx, hy, (float)cell, (float)cell, 0.0f,
                    color_with_alpha(COLOR_MINT, 25));
    }
}

static void draw_particles(FS_Core* core) {
    float scale = g_vw / 1280.0f;
    for (int i = 0; i < MAX_PARTICLES; i++) {
        Particle* p = &g_particles[i];
        if (!p->alive) continue;

        /* Trail */
        for (int t = 0; t < p->trail_len; t++) {
            float alpha_factor = 1.0f - (float)t / (float)TRAIL_LEN;
            uint8_t a = (uint8_t)(40.0f * alpha_factor);
            uint32_t tc = color_with_alpha(p->color, a);
            float sz = PARTICLE_SIZE * (1.0f - 0.4f * (float)t / (float)TRAIL_LEN) * scale;
            float tx = p->trail_x[t] - sz * 0.5f;
            float ty = p->trail_y[t] - sz * 0.5f;
            fs_cmd_rect(core, tx, ty, sz, sz, 1.0f * scale, tc);
        }

        /* Particle — cel-shaded square with outline */
        float ps = (float)PARTICLE_SIZE * scale;
        float px = p->x - ps * 0.5f;
        float py = p->y - ps * 0.5f;
        /* Fill */
        fs_cmd_rect(core, px, py, ps, ps, 2.0f * scale, p->color);
        /* Stroke outline */
        uint32_t dark = darken(p->color, 0.55f);
        fs_cmd_rect_stroke(core, px, py, ps, ps,
                           2.0f * scale, 1.5f * scale, dark);
    }
}

static void draw_gravity_wells(FS_Core* core) {
    uint32_t mode_color = g_attract_mode ? COLOR_AQUA : COLOR_CORAL;
    float mode_alpha = g_attract_mode ? 1.0f : 0.8f;
    float scale = g_vw / 1280.0f;
    float wr = (float)WELL_RADIUS * scale;

    for (int i = 0; i < MAX_WELLS; i++) {
        GravityWell* w = &g_wells[i];
        if (!w->alive && w->explode_time <= 0.0f) continue;

        float pulse_r = wr + sinf(w->pulse) * 3.0f * scale;
        float alpha = (w->alive) ? 1.0f : (w->explode_time / EXPLOSION_DURATION);
        float cap_alpha = alpha * 0.4f;

        /* Inner gradient rings (vaporwave glow) */
        for (int ring = 5; ring >= 0; ring--) {
            float rr = pulse_r * ((float)ring + 1.0f) / 6.0f;
            uint8_t ra = (uint8_t)(cap_alpha * 50.0f * (1.0f - (float)ring / 6.0f));
            uint32_t rc = color_with_alpha(mode_color, ra);
            fs_cmd_circle(core, w->x, w->y, rr, rc);
        }

        if (w->alive) {
            /* Outer ring stroke */
            fs_path_begin(core);
            fs_path_arc(core, w->x, w->y, pulse_r, 0.0f, 2.0f * PI, false);
            fs_path_stroke(core, 2.0f * scale, color_with_alpha(mode_color, (uint8_t)(200.0f * mode_alpha)));

            /* Capacity segments inside */
            int segs_per_row = 4;
            int rows = (w->max_capacity + segs_per_row - 1) / segs_per_row;
            float seg_w = 5.0f * scale, seg_h = 4.0f * scale;
            float total_w = (float)segs_per_row * seg_w;
            float total_h = (float)rows * seg_h;
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < segs_per_row; col++) {
                    int idx = row * segs_per_row + col;
                    if (idx >= w->max_capacity) break;
                    bool filled = idx < w->capacity;
                    float sx = w->x - total_w * 0.5f + (float)col * seg_w;
                    float sy = w->y - total_h * 0.5f + (float)row * seg_h;
                    uint32_t sc = filled
                        ? color_with_alpha(COLOR_MINT, 230)
                        : color_with_alpha(mode_color, 50);
                    fs_cmd_rect(core, sx, sy, seg_w - 1.0f * scale, seg_h - 1.0f * scale, 1.0f * scale, sc);
                }
            }
        } else {
            /* Fading explosion ring */
            float ring_alpha = alpha * 0.6f;
            fs_path_begin(core);
            fs_path_arc(core, w->x, w->y, pulse_r, 0.0f, 2.0f * PI, false);
            fs_path_stroke(core, 2.5f * scale, color_with_alpha(COLOR_CORAL, (uint8_t)(ring_alpha * 255.0f)));
        }
    }
}

static void draw_ghost_preview(FS_Core* core) {
    if (g_paused || g_game_over) return;
    float scale = g_vw / 1280.0f;
    int grid_cell = (int)(GRID_CELL * scale);
    float cx = ((float)g_hover_gx + 0.5f) * (float)grid_cell;
    float cy = ((float)g_hover_gy + 0.5f) * (float)grid_cell;

    /* Check occupancy by grid cell (resize-stable) */
    bool occupied = false;
    for (int i = 0; i < MAX_WELLS; i++) {
        if (g_wells[i].alive) {
            if (g_wells[i].gx == g_hover_gx && g_wells[i].gy == g_hover_gy) {
                occupied = true; break;
            }
        }
    }
    if (occupied) return;

    float wr = (float)WELL_RADIUS * scale;
    uint32_t mode_color = g_attract_mode ? COLOR_AQUA : COLOR_CORAL;

    /* Dashed ring preview — 4 arcs with gaps */
    float angles[4] = {0.0f, PI * 0.5f, PI, PI * 1.5f};
    for (int a = 0; a < 4; a++) {
        fs_path_begin(core);
        fs_path_arc(core, cx, cy, wr,
                    angles[a], angles[a] + PI * 0.38f, false);
        fs_path_stroke(core, 2.0f * scale, color_with_alpha(mode_color, 100));
    }
    /* Ghost fill */
    fs_cmd_circle(core, cx, cy, wr,
                  color_with_alpha(mode_color, 30));
}

static void draw_explosions(FS_Core* core) {
    float scale = g_vw / 1280.0f;
    for (int i = 0; i < MAX_FRAGMENTS; i++) {
        ExplosionFragment* f = &g_fragments[i];
        if (!f->alive) continue;
        float sz = 6.0f * scale * f->alpha;
        float fx = f->x - sz * 0.5f;
        float fy = f->y - sz * 0.5f;
        uint32_t fc = color_with_alpha(f->color, (uint8_t)(f->alpha * 255.0f));
        fs_cmd_rect(core, fx, fy, sz, sz, 1.0f * scale, fc);
    }
}

static void draw_hud(FS_Core* core) {
    /* HUD uses viewport dimensions directly (same coordinate space as game world) */
    float s = g_vw / 1280.0f;
    float hw = g_vw;
    float hh = g_vh;

    uint32_t panel_bg = rgba8(14, 14, 28, 220);
    uint32_t panel_stroke_aqua = color_with_alpha(COLOR_AQUA, 140);
    uint32_t panel_stroke_lav  = color_with_alpha(COLOR_LAVENDER, 140);

    /* Score panel (top-left) */
    float sp_x = 16.0f * s, sp_y = 16.0f * s;
    float sp_w = 240.0f * s, sp_h = 84.0f * s;
    fs_cmd_rect(core, sp_x, sp_y, sp_w, sp_h, 12.0f * s, panel_bg);
    fs_cmd_rect_stroke(core, sp_x, sp_y, sp_w, sp_h, 12.0f * s, 2.0f * s, panel_stroke_aqua);

    /* Score value */
    {
        char buf[32];
        snprintf(buf, sizeof(buf), "%d", g_score);
        fs_cmd_text_utf8(core, sp_x + 16.0f * s, sp_y + 28.0f * s, 26.0f * s,
                         buf, COLOR_NEON_PINK, sp_w - 32.0f * s);
    }
    {
        char buf[48];
        snprintf(buf, sizeof(buf), "WAVE %d", g_wave);
        fs_cmd_text_utf8(core, sp_x + 16.0f * s, sp_y + 58.0f * s, 13.0f * s,
                         buf, COLOR_LAVENDER, sp_w - 32.0f * s);
    }

    /* Combo indicator (if > 1) */
    if (g_combo > 1) {
        char buf[16];
        snprintf(buf, sizeof(buf), "x%d", g_combo);
        uint32_t combo_col = (g_combo >= 4) ? COLOR_PEACH
                           : (g_combo >= 3) ? COLOR_MINT : COLOR_AQUA;
        fs_cmd_text_utf8(core, sp_x + sp_w - 60.0f * s, sp_y + 16.0f * s, 18.0f * s,
                         buf, combo_col, 50.0f * s);
    }

    /* Wave indicator (top-right) */
    float wp_x = hw - 16.0f * s - 130.0f * s;
    fs_cmd_rect(core, wp_x, sp_y, 130.0f * s, 50.0f * s, 10.0f * s, panel_bg);
    fs_cmd_rect_stroke(core, wp_x, sp_y, 130.0f * s, 50.0f * s, 10.0f * s, 2.0f * s, panel_stroke_lav);
    {
        char buf[24];
        snprintf(buf, sizeof(buf), "WAVE %d", g_wave);
        fs_cmd_text_utf8(core, wp_x + 12.0f * s, sp_y + 18.0f * s, 16.0f * s,
                         buf, COLOR_AQUA, 110.0f * s);
    }

    /* Mode indicator (bottom-right) */
    float mp_x = hw - 16.0f * s - 120.0f * s;
    float mp_y = hh - 16.0f * s - 44.0f * s;
    fs_cmd_rect(core, mp_x, mp_y, 120.0f * s, 44.0f * s, 8.0f * s, panel_bg);
    uint32_t mode_color = g_attract_mode ? COLOR_AQUA : COLOR_CORAL;
    uint32_t mode_stroke = g_attract_mode
        ? color_with_alpha(COLOR_AQUA, 160) : color_with_alpha(COLOR_CORAL, 160);
    fs_cmd_rect_stroke(core, mp_x, mp_y, 120.0f * s, 44.0f * s, 8.0f * s, 2.0f * s, mode_stroke);
    {
        const char* label = g_attract_mode ? "ATTRACT" : "REPEL";
        fs_cmd_text_utf8(core, mp_x + 10.0f * s, mp_y + 16.0f * s, 14.0f * s,
                         label, mode_color, 100.0f * s);
    }

    /* Challenge banner */
    if (g_challenge_active) {
        float bw = hw - 80.0f * s;
        float bh = 120.0f * s;
        float bx = 40.0f * s;
        float by = hh * 0.5f - bh * 0.5f;
        fs_cmd_rect(core, bx, by, bw, bh, 16.0f * s, rgba8(12, 10, 28, 240));
        fs_cmd_rect_stroke(core, bx, by, bw, bh, 16.0f * s, 2.5f * s,
                           color_with_alpha(COLOR_PEACH, 200));

        if (g_challenge_countdown > 0.0f) {
            char buf[4];
            snprintf(buf, sizeof(buf), "%d", (int)(g_challenge_countdown) + 1);
            fs_cmd_text_utf8(core, hw * 0.5f - 60.0f * s,
                             hh * 0.5f - 30.0f * s, 72.0f * s,
                             buf, COLOR_PEACH, 120.0f * s);
        } else {
            char buf[128];
            snprintf(buf, sizeof(buf),
                     "CHALLENGE: Capture %d particles in %.0fs",
                     g_challenge_target, g_challenge_timer);
            fs_cmd_text_utf8(core, hw * 0.5f - 340.0f * s,
                             hh * 0.5f - 18.0f * s, 22.0f * s,
                             buf, COLOR_PEACH, 680.0f * s);
            {
                char buf2[64];
                snprintf(buf2, sizeof(buf2), "%d / %d",
                         g_challenge_captured, g_challenge_target);
                fs_cmd_text_utf8(core, hw * 0.5f - 80.0f * s,
                                 hh * 0.5f + 18.0f * s, 16.0f * s,
                                 buf2, COLOR_MINT, 160.0f * s);
            }
        }
    }

    /* Breathing room message */
    if (g_challenge_pause_timer > 0.0f) {
        fs_cmd_text_utf8(core, hw * 0.5f - 200.0f * s,
                         hh * 0.5f, 32.0f * s,
                         "BREATHING ROOM +5s",
                         color_with_alpha(COLOR_MINT, 200), 400.0f * s);
    }

    /* Pause overlay */
    if (g_paused && !g_game_over) {
        fs_cmd_rect(core, 0.0f, 0.0f, hw, hh, 0.0f, rgba8(0, 0, 0, 160));
        fs_cmd_text_utf8(core, hw * 0.5f - 120.0f * s,
                         hh * 0.5f - 30.0f * s, 56.0f * s,
                         "PAUSED", COLOR_AQUA, 240.0f * s);
        fs_cmd_text_utf8(core, hw * 0.5f - 150.0f * s,
                         hh * 0.5f + 30.0f * s, 18.0f * s,
                         "ESC to resume", COLOR_LAVENDER, 300.0f * s);
    }

    /* Game over screen */
    if (g_game_over) {
        fs_cmd_rect(core, 0.0f, 0.0f, hw, hh, 0.0f, rgba8(0, 0, 0, 185));
        fs_cmd_text_utf8(core, hw * 0.5f - 160.0f * s,
                         hh * 0.5f - 90.0f * s, 52.0f * s,
                         "GAME OVER", COLOR_CORAL, 320.0f * s);
        {
            char buf[64];
            snprintf(buf, sizeof(buf), "Score: %d", g_score);
            fs_cmd_text_utf8(core, hw * 0.5f - 160.0f * s,
                             hh * 0.5f - 10.0f * s, 28.0f * s,
                             buf, COLOR_NEON_PINK, 320.0f * s);
        }
        {
            char buf[64];
            snprintf(buf, sizeof(buf), "Best: %d", g_high_score);
            fs_cmd_text_utf8(core, hw * 0.5f - 120.0f * s,
                             hh * 0.5f + 30.0f * s, 20.0f * s,
                             buf, COLOR_LAVENDER, 240.0f * s);
        }
        fs_cmd_text_utf8(core, hw * 0.5f - 150.0f * s,
                         hh * 0.5f + 70.0f * s, 18.0f * s,
                         "Press R to restart", COLOR_MINT, 300.0f * s);
    }
}

/* ------------------------------------------------------------
   Place well at grid cell
   ------------------------------------------------------------ */

static void try_place_well(void) {
    if (g_paused || g_game_over) return;

    /* Check if occupied by grid cell (resize-stable) */
    for (int i = 0; i < MAX_WELLS; i++) {
        if (g_wells[i].alive) {
            if (g_wells[i].gx == g_hover_gx && g_wells[i].gy == g_hover_gy) return;
        }
    }

    /* Find dead slot */
    for (int i = 0; i < MAX_WELLS; i++) {
        if (!g_wells[i].alive) {
            g_wells[i].gx = g_hover_gx;
            g_wells[i].gy = g_hover_gy;
            /* x/y computed by sync_well_positions() each frame */
            g_wells[i].capacity    = 0;
            g_wells[i].max_capacity = g_max_capacity;
            g_wells[i].pulse       = 0.0f;
            g_wells[i].explode_time = 0.0f;
            g_wells[i].alive       = 1;
            break;
        }
    }
}

/* ------------------------------------------------------------
   Main
   ------------------------------------------------------------ */

int main(void) {
    /* Seed rand with a hash of __TIME__ for pseudo-randomness without time.h */
    unsigned seed = 0;
    const char* t = __TIME__;
    while (*t) seed = seed * 31u + (unsigned char)(*t++);
    srand(seed);
    FS_GlfwBackend backend;
    if (!fs_glfw_backend_init(&backend, 1280, 720, "Orbital Catcher")) {
        fprintf(stderr, "Failed to initialize backend\n");
        return 1;
    }

    FS_Core* core = fs_glfw_backend_core(&backend);
    load_high_score();
    init_pools();
    init_stars();
    reset_game();

    printf("=== ORBITAL CATCHER ===\n");
    printf("Left-click: place gravity well (grid-snapped)\n");
    printf("Space/Right-click: toggle attract/repel mode\n");
    printf("ESC: pause\n");
    printf("R (game over): restart\n");

    while (!fs_glfw_backend_should_close(&backend)) {
        fs_glfw_backend_poll_events();

        /* True responsive: game world = actual viewport dimensions */
        g_vw = (float)backend.width;
        g_vh = (float)backend.height;

        /* Mouse cursor in viewport (game world) space */
        double cursor_x = 0.0, cursor_y = 0.0;
        glfwGetCursorPos(backend.window, &cursor_x, &cursor_y);

        /* Account for framebuffer vs window size ratio (DPI scaling) */
        int win_w = 0, win_h = 0;
        glfwGetWindowSize(backend.window, &win_w, &win_h);
        float sx = (win_w > 0) ? ((float)backend.width  / (float)win_w) : 1.0f;
        float sy = (win_h > 0) ? ((float)backend.height / (float)win_h) : 1.0f;
        g_mouse_x = (float)cursor_x * sx;
        g_mouse_y = (float)cursor_y * sy;

        /* Grid snapping: hover cell = mouse / cell_size */
        int grid_cell = (int)(GRID_CELL * (g_vw / 1280.0f));
        g_hover_gx = (int)floorf(g_mouse_x / (float)grid_cell);
        g_hover_gy = (int)floorf(g_mouse_y / (float)grid_cell);

        int lmb = glfwGetMouseButton(backend.window, GLFW_MOUSE_BUTTON_LEFT);
        if (lmb == GLFW_PRESS && !g_prev_lmb) {
            try_place_well();
        }
        g_prev_lmb = (lmb == GLFW_PRESS);

        int rmb = glfwGetMouseButton(backend.window, GLFW_MOUSE_BUTTON_RIGHT);
        static int prev_rmb = GLFW_RELEASE;
        if (rmb == GLFW_PRESS && prev_rmb != GLFW_PRESS) {
            g_attract_mode = !g_attract_mode;
        }
        prev_rmb = rmb;

        static int prev_space = GLFW_RELEASE;
        int space = glfwGetKey(backend.window, GLFW_KEY_SPACE);
        if (space == GLFW_PRESS && prev_space != GLFW_PRESS) {
            g_attract_mode = !g_attract_mode;
        }
        prev_space = space;

        static int prev_esc = GLFW_RELEASE;
        int esc = glfwGetKey(backend.window, GLFW_KEY_ESCAPE);
        if (esc == GLFW_PRESS && prev_esc != GLFW_PRESS) {
            if (!g_game_over) g_paused = !g_paused;
        }
        prev_esc = esc;

        static int prev_r = GLFW_RELEASE;
        int rkey = glfwGetKey(backend.window, GLFW_KEY_R);
        if (rkey == GLFW_PRESS && prev_r != GLFW_PRESS && g_game_over) {
            reset_game();
        }
        prev_r = rkey;

        /* Sync well pixel positions from grid cells (resize-stable) */
        sync_well_positions();

        /* --- Physics --- */
        update_physics(DT);

        /* --- Render --- */
        fs_core_begin_commands(core);
        fs_transform_reset(core);
        fs_style_reset(core);
        fs_path_begin(core);

        /* True responsive: draw game world directly in viewport space.
           No scale/translate transform needed — coordinates are already in
           framebuffer pixels. Clip to viewport bounds. */
        fs_clip_rect(core, 0.0f, 0.0f, g_vw, g_vh);

        /* Screen shake (applied in design space) */
        if (g_shake_x != 0.0f || g_shake_y != 0.0f) {
            fs_translate(core, g_shake_x, g_shake_y);
        }

        draw_background(core);
        draw_stars(core);
        draw_grid(core);
        draw_particles(core);
        draw_gravity_wells(core);
        draw_ghost_preview(core);
        draw_explosions(core);

        if (g_shake_x != 0.0f || g_shake_y != 0.0f) {
            fs_translate(core, -g_shake_x, -g_shake_y);
        }

        /* HUD draws in same coordinate space as game world (viewport pixels) */
        draw_hud(core);

        if (!fs_glfw_backend_present(&backend, 0.05f, 0.05f, 0.1f, 1.0f)) {
            fprintf(stderr, "Frame present failed\n");
            break;
        }
    }

    save_high_score();
    fs_glfw_backend_shutdown(&backend);
    return 0;
}
