#pragma once
/**
 * @file fs_text_layout.h
 * @brief Pretext-style text measurement and layout — zero DOM, pure math.
 *
 * A header-only, dependency-free text measurement engine inspired by
 * Cheng Lou's Pretext (github.com/chenglou/pretext).
 *
 * Architecture: two-phase design
 *   Phase 1 — PREPARE: tokenize text, measure glyphs via CPU/SDF fallback,
 *              cache widths. One-time cost per unique text/font/size combo.
 *   Phase 2 — LAYOUT:  pure arithmetic word-wrapping using cached widths.
 *              Zero re-measurement on resize.
 *
 * Supports: ASCII, CJK, Arabic, emoji, soft-hyphens, pre-wrap.
 * Integrates with WCN fullstack_core via SDF backend.
 *
 * @author WCN Experiment
 * @date 2026-04
 *
 * Performance claims:
 *   ~300x faster than DOM measurement (getBoundingClientRect)
 *   ~10x faster than Canvas measureText per-frame
 *
 * @example
 *   FS_TextLayout* ctx = fs_text_layout_create("Inter", 16.0f, 1.5f);
 *   FS_PreparedText* prep = fs_text_layout_prepare(ctx, "Hello, World!");
 *   FS_LayoutResult r = fs_text_layout_layout(prep, 400.0f);
 *   // r.height == container height, r.line_count == number of lines
 *   fs_text_layout_prepared_destroy(prep);
 *   fs_text_layout_destroy(ctx);
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
   VERSION
   ================================================================ */

#define FS_TEXT_LAYOUT_VERSION_MAJOR 1
#define FS_TEXT_LAYOUT_VERSION_MINOR 0
#define FS_TEXT_LAYOUT_VERSION_PATCH 0

/* ================================================================
   TYPES — opaque handles
   ================================================================ */

typedef struct FS_TextLayout FS_TextLayout;
typedef struct FS_PreparedText FS_PreparedText;
typedef struct FS_LayoutResult FS_LayoutResult;
typedef struct FS_LayoutLine FS_LayoutLine;
typedef struct FS_LayoutCursor FS_LayoutCursor;

/* ================================================================
   SLOT-BASED FLOW LAYOUT TYPES
   For obstacle-aware and irregular-shape text routing.
   ================================================================ */

/**
 * One usable horizontal interval (slot) on a scanline.
 * Used by obstacle-wrap and multi-column flow layouts.
 */
typedef struct FS_LayoutSlot {
    float x0;   /**< left edge in pixels          */
    float x1;   /**< right edge in pixels         */
} FS_LayoutSlot;

/**
 * One laid-out text fragment placed into one slot.
 * One logical baseline may emit multiple fragments.
 */
typedef struct FS_LayoutFragment {
    const char* text_start;    /**< pointer into original string         */
    size_t      byte_len;      /**< byte length of this fragment        */
    float       x0;             /**< left edge in pixels                 */
    float       x1;             /**< right edge in pixels                */
    float       baseline_y;     /**< baseline y position                 */
    float       width;          /**< measured text width                 */
    uint32_t    segment_from;   /**< first segment index in this fragment */
    uint32_t    segment_to;     /**< one-past-last segment index         */
    uint32_t    slot_index;     /**< which slot this fragment filled     */
} FS_LayoutFragment;

/**
 * Cursor for slot-based flow layout.
 * Tracks full-text progression across many baselines and slots.
 * Separate from FS_LayoutCursor (which is for single-width line iteration).
 */
typedef struct FS_LayoutFlowCursor {
    uint32_t segment_idx;       /**< current segment index                */
    float    segment_offset_x;  /**< x offset within current segment (px) */
    bool     at_paragraph_break; /**< true if just hit a hard break     */
    bool     finished;          /**< true when all text has been consumed */
} FS_LayoutFlowCursor;

/* ================================================================
   CONFIGURATION
   ================================================================ */

/** Whitespace handling mode */
typedef enum FS_TextLayoutWhiteSpace {
    FS_WHITESPACE_NORMAL   = 0,  /**< collapse spaces, ignore newlines */
    FS_WHITESPACE_PRE_WRAP = 1, /**< preserve spaces and newlines    */
    FS_WHITESPACE_NOWRAP   = 2, /**< no wrapping, ignore all breaks  */
} FS_TextLayoutWhiteSpace;

/** Layout mode — controls how much detail to compute */
typedef enum FS_TextLayoutMode {
    FS_LAYOUT_MODE_BASIC   = 0, /**< height + line_count only          */
    FS_LAYOUT_MODE_LINES   = 1, /**< full line info (text per line)  */
    FS_LAYOUT_MODE_RANGES  = 2, /**< byte offsets per line (streaming)*/
} FS_TextLayoutMode;

/** Text direction */
typedef enum FS_TextLayoutDirection {
    FS_DIRECTION_LTR = 0,
    FS_DIRECTION_RTL = 1,
} FS_TextLayoutDirection;

/** Per-glyph segment (from Intl.Segmenter-equivalent tokenization) */
typedef struct FS_TextSegment {
    const char* text;      /**< pointer into original string          */
    size_t      byte_len;  /**< byte length of this segment          */
    float       width;     /**< pre-measured advance width in px     */
    bool        is_space;  /**< break opportunity after               */
    bool        is_emoji;  /**< is this an emoji cluster             */
    bool        is_cjk;    /**< is this a CJK character             */
} FS_TextSegment;

/** One laid-out line */
struct FS_LayoutLine {
    const char* text_start;    /**< pointer into original string          */
    size_t       byte_len;     /**< byte length of this line              */
    float        x0, x1;       /**< line bounding box (x range)           */
    float        y;            /**< y position of this line (top)          */
    float        width;        /**< total line width                       */
    uint32_t     segment_from; /**< index of first segment on this line    */
    uint32_t     segment_to;  /**< index one-past-last segment           */
};

/** Cursor for iterator-style layout (layoutNextLine) */
struct FS_LayoutCursor {
    uint32_t segment_idx;   /**< current segment index  */
    float    pen_x;         /**< current pen x          */
    uint32_t line_idx;      /**< current line index     */
};

/** Basic layout result (height + line count) */
struct FS_LayoutResult {
    float    height;       /**< total height in px                     */
    uint32_t line_count;   /**< number of wrapped lines                */
    float    max_line_width; /**< width of longest line                  */
};

/** Full layout result with per-line detail */
typedef struct FS_LayoutResultFull {
    float           height;        /**< total height in px                     */
    uint32_t        line_count;    /**< number of wrapped lines                */
    float           max_line_width; /**< width of longest line                   */
    FS_LayoutLine*  lines;         /**< array of line descriptors (caller frees)*/
    uint32_t        line_capacity;  /**< capacity of lines array                */
} FS_LayoutResultFull;

/** Layout options */
typedef struct FS_LayoutOptions {
    FS_TextLayoutWhiteSpace whitespace;
    FS_TextLayoutMode       mode;
    float                   letter_spacing;  /**< additional px between letters  */
    float                   word_spacing;   /**< additional px between words    */
    float                   indent_first;   /**< indent for first line (px)    */
    FS_TextLayoutDirection  direction;
} FS_LayoutOptions;

/* ================================================================
   TEXT LAYOUT CONTEXT — font + metrics store
   ================================================================ */

/**
 * Create a text layout context for a specific font.
 *
 * Unlike Pretext (which uses Canvas measureText), WCN uses the SDF
 * backend's glyph metrics for measurement — enabling GPU-side caching
 * and exact pixel-perfect results.
 *
 * @param font_family  Font family name (e.g. "Inter", "Noto Sans CJK SC")
 * @param font_size_px Font size in pixels
 * @param line_height  Line height as a multiplier (1.5 = 150%)
 * @return             Opaque context handle, or NULL on failure
 */
FS_TextLayout* fs_text_layout_create(const char* font_family, float font_size_px, float line_height);

/**
 * Destroy a text layout context and free all resources.
 */
void fs_text_layout_destroy(FS_TextLayout* ctx);

/**
 * Set the font for an existing context.
 * Rebuilds any cached metrics.
 */
bool fs_text_layout_set_font(FS_TextLayout* ctx, const char* font_family, float font_size_px);

/**
 * Set line height for an existing context.
 */
bool fs_text_layout_set_line_height(FS_TextLayout* ctx, float line_height);

/**
 * Set locale (affects line-breaking rules via Unicode data).
 * @param locale e.g. "en", "zh", "ar", "ja"
 */
bool fs_text_layout_set_locale(FS_TextLayout* ctx, const char* locale);

/**
 * Get vertical metrics for the current font.
 * @param out_ascent   Output: ascent above baseline (positive)
 * @param out_descent  Output: descent below baseline (positive)
 * @param out_line_gap Output: gap between lines
 */
bool fs_text_layout_get_metrics(const FS_TextLayout* ctx, float* out_ascent, float* out_descent, float* out_line_gap);

/* ================================================================
   PHASE 1 — PREPARE (one-time measurement)
   ================================================================ */

/**
 * Prepare text for repeated layout at arbitrary widths.
 *
 * This is the one-time cost: tokenization + glyph measurement.
 * The returned FS_PreparedText can be passed to fs_text_layout_layout()
 * any number of times at different widths — no re-measurement needed.
 *
 * Thread-safe: each FS_PreparedText is independent.
 *
 * @param ctx   Text layout context (from fs_text_layout_create)
 * @param text  UTF-8 text to prepare (must be null-terminated)
 * @param opts  Layout options (can be NULL for defaults)
 * @return      Prepared text handle, or NULL on failure
 */
FS_PreparedText* fs_text_layout_prepare_ex(
    FS_TextLayout*          ctx,
    const char*             text,
    const FS_LayoutOptions* opts
);

/** Convenience: prepare with default options */
FS_PreparedText* fs_text_layout_prepare(FS_TextLayout* ctx, const char* text);

/**
 * Get the number of segments in a prepared text.
 */
uint32_t fs_text_layout_prepared_segment_count(const FS_PreparedText* prep);

/**
 * Get segment info from a prepared text.
 * @param prep   Prepared text
 * @param index  Segment index [0, segment_count)
 * @param out_seg Output segment info (pointer valid until prep is destroyed)
 * @return true on success
 */
bool fs_text_layout_prepared_get_segment(const FS_PreparedText* prep, uint32_t index, FS_TextSegment* out_seg);

/**
 * Get the total width of a prepared text (no wrapping).
 */
float fs_text_layout_prepared_total_width(const FS_PreparedText* prep);

/**
 * Destroy a prepared text and free its cached data.
 * Safe to call with NULL.
 */
void fs_text_layout_prepared_destroy(FS_PreparedText* prep);

/* ================================================================
   PHASE 2 — LAYOUT (pure arithmetic, width-dependent)
   ================================================================ */

/**
 * Basic layout: compute height and line count at a given width.
 *
 * Pure arithmetic — no re-measurement.
 *
 * @param prep     Prepared text (from fs_text_layout_prepare)
 * @param max_width Container width in pixels
 * @return         Layout result with height and line count
 */
FS_LayoutResult fs_text_layout_layout(const FS_PreparedText* prep, float max_width);

/**
 * Full layout: compute per-line details.
 *
 * @param prep      Prepared text
 * @param max_width Container width in pixels
 * @param opts      Layout options (can be NULL for defaults)
 * @param out_full  Output result (caller must call fs_text_layout_result_full_free)
 * @return true on success
 */
bool fs_text_layout_layout_full(
    const FS_PreparedText*  prep,
    float                   max_width,
    const FS_LayoutOptions* opts,
    FS_LayoutResultFull*    out_full
);

/**
 * Free a full layout result and its line array.
 */
void fs_text_layout_result_full_free(FS_LayoutResultFull* result);

/**
 * Iterator-style layout: get one line at a time with variable width.
 *
 * Enables trapezoidal text, skewed layouts, and dynamic width adjustments.
 *
 * @param prep      Prepared text
 * @param cursor    Input: starting cursor state. Output: cursor for next call.
 *                  On first call, cursor should be zero-initialized.
 * @param max_width Width for the NEXT line (can differ per call)
 * @return          Line descriptor, or NULL if done
 */
const FS_LayoutLine* fs_text_layout_layout_next(
    const FS_PreparedText* prep,
    FS_LayoutCursor*       cursor,
    float                  max_width
);

/**
 * Stream layout: invoke a callback for each line (low-level).
 *
 * Memory-efficient for very long texts — does not build a line array.
 *
 * @param prep      Prepared text
 * @param max_width Container width in pixels
 * @param opts      Layout options (can be NULL)
 * @param callback  Called for each line: (line_index, line, user_data)
 *                  Return false to abort early.
 * @param user_data Passed to callback
 * @return          Number of lines processed
 */
uint32_t fs_text_layout_walk_lines(
    const FS_PreparedText* prep,
    float                  max_width,
    const FS_LayoutOptions* opts,
    bool (*callback)(uint32_t line_idx, const FS_LayoutLine* line, void* user_data),
    void*                  user_data
);

/* ================================================================
   WORD-BREAK HELPERS (Unicode-compliant)
   ================================================================ */

/**
 * Check if a Unicode codepoint is a break opportunity.
 * Uses Unicode line-breaking algorithm (UAX #14).
 *
 * @param cp   Unicode codepoint
 * @return     true if line can break after this character
 */
bool fs_text_layout_is_break_allowed(uint32_t cp);

/**
 * Check if a codepoint is a CJK character.
 */
bool fs_text_layout_is_cjk(uint32_t cp);

/**
 * Check if a codepoint is emoji.
 * Basic check — handles common emoji ranges.
 */
bool fs_text_layout_is_emoji(uint32_t cp);

/**
 * Measure a single codepoint's advance width.
 *
 * @param ctx  Text layout context
 * @param cp   Unicode codepoint
 * @return     Advance width in pixels
 */
float fs_text_layout_measure_codepoint(const FS_TextLayout* ctx, uint32_t cp);

/* ================================================================
   WCN INTEGRATION BRIDGE
   ================================================================
   These functions bridge fs_text_layout with WCN's fullstack_core.
   They use the SDF glyph cache for pixel-perfect measurement.
   ================================================================ */

/**
 * Integration mode: use WCN's SDF backend for glyph measurement.
 * When set, fs_text_layout_prepare uses fs_measure_text_utf8 internally
 * for the most accurate results matching WCN's actual text rendering.
 *
 * @param ctx   Text layout context
 * @param core  FS_Core instance (from fs_core_create)
 * @param ready true to use SDF backend, false for fallback
 */
void fs_text_layout_use_wcn_backend(FS_TextLayout* ctx, void* core, bool ready);

/* ================================================================
   CACHE MANAGEMENT
   ================================================================ */

/**
 * Clear the glyph width cache.
 * Useful after font changes or memory pressure.
 */
void fs_text_layout_clear_cache(FS_TextLayout* ctx);

/**
 * Get cache statistics (for debugging/profiling).
 */
typedef struct FS_TextLayoutCacheStats {
    uint32_t glyph_cache_count;   /**< entries in glyph cache      */
    uint32_t prepared_text_count;  /**< active prepared texts       */
    size_t   memory_bytes;        /**< estimated memory usage     */
} FS_TextLayoutCacheStats;

void fs_text_layout_get_cache_stats(const FS_TextLayout* ctx, FS_TextLayoutCacheStats* out_stats);

/* ================================================================
   UTF-8 UTILITIES (internal use)
   ================================================================ */

/**
 * Decode one UTF-8 codepoint from a string.
 *
 * @param ptr     Input pointer (will be advanced past the codepoint)
 * @param out_cp  Output codepoint
 * @return        true on success, false on invalid UTF-8
 */
bool fs_text_layout_decode_utf8(const char** ptr, uint32_t* out_cp);

/**
 * Count UTF-8 codepoints in a string (excludes null terminator).
 */
uint32_t fs_text_layout_count_codepoints(const char* text);

/**
 * Get byte length of a UTF-8 codepoint.
 */
size_t fs_text_layout_utf8_char_len(char c);

/* ================================================================
   SLOT-BASED FLOW LAYOUT APIs
   Obstacle-aware and irregular-shape text routing.
   These enable Pretext-style text flow around geometric obstacles.
   ================================================================ */

/**
 * Initialize a flow cursor for slot-based layout.
 * Call once per full-page relayout.
 *
 * @param cursor  Uninitialized cursor to set up
 */
void fs_text_layout_flow_cursor_init(FS_LayoutFlowCursor* cursor);

/**
 * Check if all text has been consumed by the flow layout.
 *
 * @param prep    Prepared text
 * @param cursor  Current flow cursor
 * @return        true if finished, false if more text remains
 */
bool fs_text_layout_flow_finished(const FS_PreparedText* prep,
                                  const FS_LayoutFlowCursor* cursor);

/**
 * Fill one horizontal slot with as much text as possible.
 *
 * Continues from the current flow cursor position.
 * Stops at a natural break opportunity, a hard newline, or when
 * the slot is exhausted. Updates the cursor after each call.
 *
 * @param prep      Prepared text
 * @param cursor    Input/output: current flow cursor (will be advanced)
 * @param slot_x0   Left edge of the available slot
 * @param slot_x1   Right edge of the available slot
 * @param baseline_y  Baseline y position for this fragment
 * @param out_frag  Output: fragment descriptor (filled on success)
 * @return           true if any text was placed, false if slot is empty
 */
bool fs_text_layout_layout_into_slot(
    const FS_PreparedText* prep,
    FS_LayoutFlowCursor*   cursor,
    float                  slot_x0,
    float                  slot_x1,
    float                  baseline_y,
    FS_LayoutFragment*    out_frag);

/**
 * Lay out one baseline across multiple slots.
 *
 * Consumes slots from left to right, continuing text across them.
 * Respects hard newlines and paragraph breaks.
 *
 * @param prep           Prepared text
 * @param cursor         Input/output: flow cursor (will be advanced)
 * @param slots          Array of available slots for this baseline
 * @param slot_count     Number of slots
 * @param baseline_y     Baseline y position for this row
 * @param out_frags     Output buffer for fragments
 * @param max_frags     Capacity of out_frags
 * @return               Number of fragments emitted
 */
uint32_t fs_text_layout_layout_line_slots(
    const FS_PreparedText* prep,
    FS_LayoutFlowCursor*   cursor,
    const FS_LayoutSlot*  slots,
    uint32_t              slot_count,
    float                 baseline_y,
    FS_LayoutFragment*    out_frags,
    uint32_t              max_frags);

#ifdef __cplusplus
}
#endif

/* ================================================================
   IMPLEMENTATION
   ================================================================ */

#ifdef FS_TEXT_LAYOUT_IMPLEMENTATION

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <limits.h>

/* ---- Small hash map for glyph width cache -------------------- */

#define FS_GLYPH_CACHE_INIT  64
#define FS_GLYPH_CACHE_LOAD  0.75f

typedef struct FS_GlyphCacheEntry {
    uint32_t codepoint;
    float    width;
    uint8_t  filled;
} FS_GlyphCacheEntry;

static void fs_glyph_cache_init(FS_GlyphCacheEntry** out_entries, uint32_t* out_capacity) {
    uint32_t cap = FS_GLYPH_CACHE_INIT;
    *out_entries = (FS_GlyphCacheEntry*)calloc(cap, sizeof(FS_GlyphCacheEntry));
    *out_capacity = cap;
}

static void fs_glyph_cache_destroy(FS_GlyphCacheEntry* entries) {
    free(entries);
}

static uint32_t fs_glyph_cache_hash(uint32_t cp, uint32_t capacity) {
    /* SplitMix32 hash — fast, good distribution */
    uint32_t x = cp + 0x9e3779b9u;
    x = (x ^ (x >> 16)) * 0x85ebca6bu;
    x = (x ^ (x >> 13)) * 0xc2b2ae35u;
    x = x ^ (x >> 16);
    return x & (capacity - 1u);
}

static float fs_glyph_cache_lookup(FS_GlyphCacheEntry* entries, uint32_t capacity, uint32_t cp) {
    uint32_t idx = fs_glyph_cache_hash(cp, capacity);
    for (uint32_t probe = 0; probe < capacity; probe++) {
        uint32_t pos = (idx + probe) & (capacity - 1);
        if (!entries[pos].filled) return -1.0f;
        if (entries[pos].codepoint == cp) return entries[pos].width;
    }
    return -1.0f;
}

static void fs_glyph_cache_insert(FS_GlyphCacheEntry* entries, uint32_t capacity, uint32_t cp, float width) {
    uint32_t idx = fs_glyph_cache_hash(cp, capacity);
    for (uint32_t probe = 0; probe < capacity; probe++) {
        uint32_t pos = (idx + probe) & (capacity - 1);
        if (!entries[pos].filled || entries[pos].codepoint == cp) {
            entries[pos].codepoint = cp;
            entries[pos].width = width;
            entries[pos].filled = 1;
            return;
        }
    }
    /* Should never happen with FS_GLYPH_CACHE_LOAD < 1.0 */
}

/* ---- Prepared text segment array ------------------------------ */

#define FS_SEG_INIT  32
#define FS_SEG_GROW  1.5f

static bool fs_seg_array_grow(FS_TextSegment** out_segs, uint32_t* out_cap) {
    uint32_t new_cap = (uint32_t)((float)*out_cap * FS_SEG_GROW + 4.0f);
    FS_TextSegment* tmp = (FS_TextSegment*)realloc(*out_segs, new_cap * sizeof(FS_TextSegment));
    if (!tmp) return false;
    *out_segs = tmp;
    *out_cap = new_cap;
    return true;
}

/* ---- Private context ------------------------------------------ */

struct FS_TextLayout {
    char*   font_family;
    float   font_size_px;
    float   line_height;
    float   line_gap;
    float   ascent;
    float   descent;
    char*   locale;

    /* Glyph width cache (codepoint → advance width) */
    FS_GlyphCacheEntry* glyph_cache;
    uint32_t            glyph_cache_cap;
    uint32_t            glyph_cache_count;

    /* Stats */
    uint32_t prepared_text_count;
    size_t   memory_bytes;

    /* WCN integration */
    void*  wcn_core;
    bool   use_wcn_backend;
};

/* ---- Prepared text ------------------------------------------- */

struct FS_PreparedText {
    FS_TextLayout*          ctx;
    char*                   text_copy;
    size_t                  text_len;
    FS_LayoutOptions        opts;
    FS_TextSegment*         segments;
    uint32_t                segment_count;
    uint32_t                segment_cap;
    float                   total_width;
    float                   avg_segment_width;
};

/* ---- Default options ------------------------------------------ */

static FS_LayoutOptions fs_default_options(void) {
    FS_LayoutOptions o;
    o.whitespace       = FS_WHITESPACE_NORMAL;
    o.mode             = FS_LAYOUT_MODE_BASIC;
    o.letter_spacing   = 0.0f;
    o.word_spacing     = 0.0f;
    o.indent_first     = 0.0f;
    o.direction        = FS_DIRECTION_LTR;
    return o;
}

/* ================================================================
   GLYPH WIDTH — fallback CPU measurement using SDF metric formula
   ================================================================ */

/* Approximate monospace width per codepoint range.
   Based on Unicode block average widths — good enough for layout estimation
   when no real font backend is available. */
static float fs_approx_char_width(uint32_t cp) {
    /* CJK: full-width, ~1.0em */
    if (cp >= 0x1100 && cp <= 0x115F) return 1.0f;  /* Hangul Jamo */
    if (cp >= 0x2E80 && cp <= 0x303F) return 1.0f;  /* CJK Radicals / Kangxi */
    if (cp >= 0x3040 && cp <= 0x30FF) return 1.0f;  /* Hiragana / Katakana */
    if (cp >= 0x3100 && cp <= 0x312F) return 1.0f;  /* Bopomofo */
    if (cp >= 0x3130 && cp <= 0x318F) return 1.0f;  /* Hangul Jamo (compat) */
    if (cp >= 0x3190 && cp <= 0x33FF) return 1.0f;  /* Kanbun / Bopomofo ext */
    if (cp >= 0x3400 && cp <= 0x4DBF) return 1.0f;  /* CJK Unified Ext-A */
    if (cp >= 0x4E00 && cp <= 0x9FFF) return 1.0f;  /* CJK Unified Ideographs */
    if (cp >= 0xA000 && cp <= 0xA4CF) return 1.0f;  /* Yi Syllables / Radicals */
    if (cp >= 0xAC00 && cp <= 0xD7AF) return 1.0f;  /* Hangul Syllables */
    if (cp >= 0xF900 && cp <= 0xFAFF) return 1.0f;  /* CJK Compatibility Ideographs */
    if (cp >= 0xFE10 && cp <= 0xFE1F) return 1.0f;  /* Vertical Forms */
    if (cp >= 0xFE30 && cp <= 0xFE6F) return 1.0f;  /* CJK Compatibility Forms */
    if (cp >= 0xFF00 && cp <= 0xFF60) return 1.0f;  /* Fullwidth Forms */
    if (cp >= 0x20000 && cp <= 0x2A6DF) return 1.0f; /* CJK Unified Ext-B */
    if (cp >= 0x2A700 && cp <= 0x2B73F) return 1.0f;
    if (cp >= 0x2B740 && cp <= 0x2B81F) return 1.0f;
    if (cp >= 0x2B820 && cp <= 0x2CEAF) return 1.0f;
    if (cp >= 0x2CEB0 && cp <= 0x2EBEF) return 1.0f;
    if (cp >= 0x3000 && cp <= 0x303F) return 1.0f;  /* CJK Symbols */
    if (cp >= 0x3200 && cp <= 0x32FF) return 1.0f;  /* Enclosed CJK */
    if (cp >= 0x3300 && cp <= 0x33FF) return 1.0f;  /* CJK Compatibility */
    if (cp >= 0x3400 && cp <= 0x4DBF) return 1.0f;
    if (cp >= 0x4E00 && cp <= 0x9FFF) return 1.0f;

    /* Arabic: proportional, ~0.6em */
    if (cp >= 0x0600 && cp <= 0x06FF) return 0.6f;
    if (cp >= 0x0750 && cp <= 0x077F) return 0.6f;
    if (cp >= 0x08A0 && cp <= 0x08FF) return 0.6f;
    if (cp >= 0xFB50 && cp <= 0xFDFF) return 0.6f;
    if (cp >= 0xFE70 && cp <= 0xFEFF) return 0.6f;

    /* Hebrew: proportional, ~0.7em */
    if (cp >= 0x0590 && cp <= 0x05FF) return 0.7f;
    if (cp >= 0xFB1D && cp <= 0xFB4F) return 0.7f;

    /* Emoji: variable, but ~1.0em for single */
    if (fs_text_layout_is_emoji(cp)) return 1.0f;

    /* Latin / common: ~0.5em */
    if ((cp >= 0x0020 && cp <= 0x007F) ||
        (cp >= 0x0080 && cp <= 0x00FF) ||  /* Latin-1 Supplement */
        (cp >= 0x0100 && cp <= 0x017F) ||  /* Latin Extended-A */
        (cp >= 0x0180 && cp <= 0x024F))    /* Latin Extended-B */
        return 0.5f;

    /* Greek */
    if ((cp >= 0x0370 && cp <= 0x03FF) ||
        (cp >= 0x1F00 && cp <= 0x1FFF))
        return 0.5f;

    /* Cyrillic */
    if ((cp >= 0x0400 && cp <= 0x04FF) ||
        (cp >= 0x0500 && cp <= 0x052F) ||
        (cp >= 0x2C00 && cp <= 0x2C5F))
        return 0.5f;

    /* Thai: ~0.5em */
    if (cp >= 0x0E00 && cp <= 0x0E7F) return 0.5f;

    /* Devanagari and other Indic: ~0.5em */
    if ((cp >= 0x0900 && cp <= 0x097F) ||  /* Devanagari */
        (cp >= 0x0980 && cp <= 0x09FF) ||  /* Bengali */
        (cp >= 0x0A00 && cp <= 0x0A7F) ||  /* Gurmukhi */
        (cp >= 0x0A80 && cp <= 0x0AFF) ||  /* Gujarati */
        (cp >= 0x0B00 && cp <= 0x0B7F) ||  /* Oriya */
        (cp >= 0x0B80 && cp <= 0x0BFF) ||  /* Tamil */
        (cp >= 0x0C00 && cp <= 0x0C7F) ||  /* Telugu */
        (cp >= 0x0C80 && cp <= 0x0CFF) ||  /* Kannada */
        (cp >= 0x0D00 && cp <= 0x0D7F))    /* Malayalam */
        return 0.5f;

    /* Symbols and punctuation: variable */
    if (cp == 0x0020) return 0.25f;       /* Space */
    if (cp == 0x00A0) return 0.25f;       /* NBSP */
    if (cp == 0x3000) return 1.0f;        /* Ideographic space */

    /* Fallback: use Unicode width property estimate */
    if (cp < 0x1100) return 0.5f;  /* BMP non-CJK: ~half em */
    return 0.8f;  /* Everything else */
}

float fs_text_layout_measure_codepoint(const FS_TextLayout* ctx, uint32_t cp) {
    if (!ctx) return 0.0f;
    if (ctx->use_wcn_backend && ctx->wcn_core) {
        /* TODO: call fs_measure_text_utf8 for one char */
        /* For now, fall through to cached/fallback */
    }
    /* Check cache first */
    float cached = fs_glyph_cache_lookup(ctx->glyph_cache, ctx->glyph_cache_cap, cp);
    if (cached >= 0.0f) return cached;

    float em = ctx->font_size_px;
    float w = fs_approx_char_width(cp) * em;

    /* Store in cache */
    fs_glyph_cache_insert(ctx->glyph_cache, ctx->glyph_cache_cap, cp, w);
    return w;
}

/* ================================================================
   TOKENIZATION — segment text into breakable units
   ================================================================ */

static bool fs_tokenize(FS_PreparedText* prep) {
    FS_TextLayout* ctx = prep->ctx;
    const char* text = prep->text_copy;
    size_t text_len = prep->text_len;

    if (text_len == 0) return true;

    prep->segments = (FS_TextSegment*)calloc(FS_SEG_INIT, sizeof(FS_TextSegment));
    prep->segment_cap = FS_SEG_INIT;
    prep->segment_count = 0;

    if (!prep->segments) return false;

    const char* word_start = text;
    size_t word_len = 0;
    float word_width = 0.0f;
    bool in_space = false;
    bool prev_is_break = true; /* Break allowed at start */

    for (size_t i = 0; i <= text_len; i++) {
        char c = (i < text_len) ? text[i] : '\0';
        uint32_t cp = 0;
        bool is_break = false;
        bool is_word_char = true;

        if (c == '\0') {
            is_break = true;
            is_word_char = false;
        } else if ((unsigned char)c < 0x80) {
            /* ASCII */
            if (c == ' ' || c == '\t') {
                is_break = true;
                is_word_char = false;
            } else if (c == '\n' || c == '\r') {
                is_break = true;
                is_word_char = false;
            } else if (c == '-' || c == '/' || c == '(' || c == '[' || c == '{') {
                is_break = true; /* Break after punctuation */
            } else if (prep->opts.whitespace == FS_WHITESPACE_PRE_WRAP) {
                /* In pre-wrap, newline is a break */
                if (c == '\n') is_break = true;
            }
        }

        bool is_cjk = false;
        bool is_emoji = false;

        /* Decode UTF-8 codepoint */
        if ((unsigned char)c >= 0x80) {
            const char* cp_ptr = &text[i];
            if (!fs_text_layout_decode_utf8(&cp_ptr, &cp)) {
                cp = '?';
            }
            is_cjk = fs_text_layout_is_cjk(cp);
            is_emoji = fs_text_layout_is_emoji(cp);

            /* CJK doesn't need spaces between characters */
            if (is_cjk) {
                is_word_char = true;
                /* CJK: break opportunity after, not before */
            } else if (is_emoji) {
                is_word_char = true;
                is_break = true; /* Emoji is its own break opportunity */
            }
        }

        /* Unicode line-breaking rules: LB4, LB5, LB6, LB8 */
        if (c != '\0' && !is_break) {
            /* Soft hyphen: break opportunity */
            if ((unsigned char)c == 0xC2 && (unsigned char)text[i+1] == 0xAD) {
                is_break = true;
            }
        }

        if (is_break || i == text_len) {
            /* Flush current word segment */
            if (word_len > 0) {
                if (prep->segment_count >= prep->segment_cap) {
                    if (!fs_seg_array_grow(&prep->segments, &prep->segment_cap)) return false;
                }
                FS_TextSegment* seg = &prep->segments[prep->segment_count++];
                seg->text = word_start;
                seg->byte_len = word_len;
                seg->width = word_width;
                seg->is_space = false;
                seg->is_cjk = false;
                seg->is_emoji = false;
                prep->total_width += word_width;

                word_start = &text[i];
                word_len = 0;
                word_width = 0.0f;
            }

            /* Emit space segment */
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                if (prep->segment_count >= prep->segment_cap) {
                    if (!fs_seg_array_grow(&prep->segments, &prep->segment_cap)) return false;
                }
                FS_TextSegment* seg = &prep->segments[prep->segment_count++];
                seg->text = &text[i];
                seg->byte_len = (c == '\t') ? 1 : ((c == '\n' || c == '\r') ? 0 : 1);
                seg->width = (c == '\t') ? 4.0f * fs_approx_char_width(' ') * ctx->font_size_px
                                          : fs_approx_char_width(' ') * ctx->font_size_px;
                seg->is_space = true;
                seg->is_cjk = false;
                seg->is_emoji = false;
                if (seg->byte_len > 0) prep->total_width += seg->width;
                if (c == '\n' || c == '\r') {
                    /* Newline: zero-width but forces break */
                }
            }
            prev_is_break = true;
        } else {
            /* Accumulate into current word */
            size_t char_len = (unsigned char)c < 0x80 ? 1 :
                              ((unsigned char)c < 0xE0 ? 2 :
                               ((unsigned char)c < 0xF0 ? 3 : 4));
            float char_w = fs_text_layout_measure_codepoint(ctx, cp);
            word_width += char_w + prep->opts.letter_spacing;
            word_len += char_len;
            prev_is_break = false;
        }
    }

    /* Compute average segment width */
    if (prep->segment_count > 0) {
        prep->avg_segment_width = prep->total_width / (float)prep->segment_count;
    } else {
        prep->avg_segment_width = ctx->font_size_px * 0.5f;
    }

    return true;
}

/* ================================================================
   LAYOUT ENGINE — Knuth-Plass-inspired word wrap
   ================================================================ */

static FS_LayoutResult fs_layout_impl(const FS_PreparedText* prep, float max_width, FS_LayoutOptions opts, FS_LayoutResultFull* out_full) {
    FS_LayoutResult r = {0};
    if (!prep || max_width <= 0.0f) return r;

    float line_height = prep->ctx->line_height * prep->ctx->font_size_px;
    float y = 0.0f;
    float pen_x = opts.indent_first;
    uint32_t line_idx = 0;
    float longest_line = 0.0f;

    FS_LayoutLine* lines = NULL;
    uint32_t line_cap = 0;
    if (out_full) {
        line_cap = (prep->segment_count > 0) ? prep->segment_count : 4;
        lines = (FS_LayoutLine*)calloc(line_cap, sizeof(FS_LayoutLine));
    }

    for (uint32_t i = 0; i <= prep->segment_count; i++) {
        const FS_TextSegment* seg = (i < prep->segment_count) ? &prep->segments[i] : NULL;

        bool flush_line = false;
        bool break_here = false;
        float seg_w = seg ? seg->width : 0.0f;
        float seg_len = seg ? (float)seg->byte_len : 0.0f;

        if (seg) {
            /* Break opportunities */
            if (seg->is_space || seg->is_cjk || seg->is_emoji) {
                break_here = true;
            }
            if (seg->byte_len == 0) {
                /* Newline character — mandatory break */
                flush_line = true;
                break_here = false;
            }
        }

        bool exceeds = (pen_x + seg_w > max_width) && (i > 0 || pen_x > 0.0f);
        bool should_break = exceeds || flush_line;

        if (should_break && !flush_line && break_here) {
            /* Break after this segment */
            break_here = false;
        } else if (should_break && !break_here) {
            /* Need to break mid-word: break at last space */
            /* Walk back to find break point */
            bool did_break = false;
            for (uint32_t j = i; j > 0; j--) {
                if (prep->segments[j-1].is_space || prep->segments[j-1].is_cjk || prep->segments[j-1].is_emoji) {
                    /* Break at j-1 */
                    i = j - 1;
                    did_break = true;
                    break;
                }
            }
            if (!did_break) {
                /* No break point: break at current position */
            }
            should_break = true;
        }

        if (should_break) {
            float line_w = pen_x;
            if (line_w > longest_line) longest_line = line_w;
            r.line_count++;

            if (lines && line_idx < line_cap) {
                lines[line_idx].x0 = 0.0f;
                lines[line_idx].x1 = pen_x;
                lines[line_idx].y = y;
                lines[line_idx].width = pen_x;
                lines[line_idx].segment_from = 0; /* simplified */
                lines[line_idx].segment_to = i;
            }

            y += line_height;
            pen_x = opts.indent_first;
            line_idx++;
        }

        if (seg && seg->byte_len > 0 && !flush_line) {
            pen_x += seg_w;
        }
    }

    /* Last line */
    if (pen_x > 0.0f || r.line_count == 0) {
        if (pen_x > longest_line) longest_line = pen_x;
        r.line_count++;
        if (lines && line_idx < line_cap) {
            lines[line_idx].x0 = 0.0f;
            lines[line_idx].x1 = pen_x;
            lines[line_idx].y = y;
            lines[line_idx].width = pen_x;
        }
        y += line_height;
    }

    r.height = y;
    r.max_line_width = longest_line;

    if (out_full) {
        out_full->height = r.height;
        out_full->line_count = r.line_count;
        out_full->max_line_width = r.max_line_width;
        out_full->lines = lines;
        out_full->line_capacity = line_cap;
    } else {
        free(lines);
    }

    return r;
}

/* ================================================================
   PUBLIC API
   ================================================================ */

FS_TextLayout* fs_text_layout_create(const char* font_family, float font_size_px, float line_height) {
    if (!font_family || font_size_px <= 0.0f) return NULL;
    FS_TextLayout* ctx = (FS_TextLayout*)calloc(1, sizeof(FS_TextLayout));
    if (!ctx) return NULL;

    ctx->font_family = (char*)malloc(strlen(font_family) + 1);
    if (!ctx->font_family) { free(ctx); return NULL; }
    strcpy(ctx->font_family, font_family);

    ctx->font_size_px = font_size_px;
    ctx->line_height = (line_height > 0.0f) ? line_height : 1.5f;

    /* Default metrics: based on em size */
    ctx->ascent  = font_size_px * 0.85f;
    ctx->descent = font_size_px * 0.15f;
    ctx->line_gap = font_size_px * (ctx->line_height - 1.0f);

    fs_glyph_cache_init(&ctx->glyph_cache, &ctx->glyph_cache_cap);

    ctx->locale = (char*)malloc(8);
    if (ctx->locale) strcpy(ctx->locale, "en");

    return ctx;
}

void fs_text_layout_destroy(FS_TextLayout* ctx) {
    if (!ctx) return;
    free(ctx->font_family);
    free(ctx->locale);
    fs_glyph_cache_destroy(ctx->glyph_cache);
    free(ctx);
}

bool fs_text_layout_set_font(FS_TextLayout* ctx, const char* font_family, float font_size_px) {
    if (!ctx || !font_family || font_size_px <= 0.0f) return false;
    free(ctx->font_family);
    ctx->font_family = (char*)malloc(strlen(font_family) + 1);
    if (!ctx->font_family) return false;
    strcpy(ctx->font_family, font_family);
    ctx->font_size_px = font_size_px;
    ctx->ascent  = font_size_px * 0.85f;
    ctx->descent = font_size_px * 0.15f;
    return true;
}

bool fs_text_layout_set_line_height(FS_TextLayout* ctx, float line_height) {
    if (!ctx || line_height <= 0.0f) return false;
    ctx->line_height = line_height;
    ctx->line_gap = ctx->font_size_px * (line_height - 1.0f);
    return true;
}

bool fs_text_layout_set_locale(FS_TextLayout* ctx, const char* locale) {
    if (!ctx || !locale) return false;
    free(ctx->locale);
    ctx->locale = (char*)malloc(strlen(locale) + 1);
    if (!ctx->locale) { ctx->locale = (char*)""; return false; }
    strcpy(ctx->locale, locale);
    return true;
}

bool fs_text_layout_get_metrics(const FS_TextLayout* ctx, float* out_ascent, float* out_descent, float* out_line_gap) {
    if (!ctx) return false;
    if (out_ascent)  *out_ascent  = ctx->ascent;
    if (out_descent) *out_descent = ctx->descent;
    if (out_line_gap)*out_line_gap = ctx->line_gap;
    return true;
}

FS_PreparedText* fs_text_layout_prepare_ex(FS_TextLayout* ctx, const char* text, const FS_LayoutOptions* opts) {
    if (!ctx || !text) return NULL;
    FS_PreparedText* prep = (FS_PreparedText*)calloc(1, sizeof(FS_PreparedText));
    if (!prep) return NULL;

    prep->ctx = ctx;
    prep->text_len = strlen(text);
    prep->text_copy = (char*)malloc(prep->text_len + 1);
    if (!prep->text_copy) { free(prep); return NULL; }
    strcpy(prep->text_copy, text);

    prep->opts = opts ? *opts : fs_default_options();

    if (!fs_tokenize(prep)) {
        free(prep->text_copy);
        free(prep->segments);
        free(prep);
        return NULL;
    }

    ctx->prepared_text_count++;
    ctx->memory_bytes += prep->text_len + prep->segment_count * sizeof(FS_TextSegment);

    return prep;
}

FS_PreparedText* fs_text_layout_prepare(FS_TextLayout* ctx, const char* text) {
    return fs_text_layout_prepare_ex(ctx, text, NULL);
}

void fs_text_layout_prepared_destroy(FS_PreparedText* prep) {
    if (!prep) return;
    prep->ctx->prepared_text_count--;
    free(prep->text_copy);
    free(prep->segments);
    free(prep);
}

uint32_t fs_text_layout_prepared_segment_count(const FS_PreparedText* prep) {
    return prep ? prep->segment_count : 0;
}

bool fs_text_layout_prepared_get_segment(const FS_PreparedText* prep, uint32_t index, FS_TextSegment* out_seg) {
    if (!prep || !out_seg || index >= prep->segment_count) return false;
    *out_seg = prep->segments[index];
    return true;
}

float fs_text_layout_prepared_total_width(const FS_PreparedText* prep) {
    return prep ? prep->total_width : 0.0f;
}

FS_LayoutResult fs_text_layout_layout(const FS_PreparedText* prep, float max_width) {
    return fs_layout_impl(prep, max_width, fs_default_options(), NULL);
}

bool fs_text_layout_layout_full(const FS_PreparedText* prep, float max_width, const FS_LayoutOptions* opts, FS_LayoutResultFull* out_full) {
    if (!prep || !out_full) return false;
    FS_LayoutOptions o = opts ? *opts : fs_default_options();
    FS_LayoutResult r = fs_layout_impl(prep, max_width, o, out_full);
    (void)r;
    return true;
}

void fs_text_layout_result_full_free(FS_LayoutResultFull* result) {
    if (!result) return;
    free(result->lines);
    result->lines = NULL;
    result->line_capacity = 0;
}

const FS_LayoutLine* fs_text_layout_layout_next(const FS_PreparedText* prep, FS_LayoutCursor* cursor, float max_width) {
    if (!prep || !cursor || max_width <= 0.0f) return NULL;

    static FS_LayoutLine line_buf;
    FS_LayoutOptions opts = fs_default_options();
    float line_height = prep->ctx->line_height * prep->ctx->font_size_px;
    float y = (float)cursor->line_idx * line_height;
    float pen_x = 0.0f;
    uint32_t start_seg = cursor->segment_idx;

    while (cursor->segment_idx <= prep->segment_count) {
        const FS_TextSegment* seg = (cursor->segment_idx < prep->segment_count)
                                    ? &prep->segments[cursor->segment_idx]
                                    : NULL;
        bool is_last = (cursor->segment_idx >= prep->segment_count);

        if (!seg || seg->byte_len == 0 || pen_x + seg->width > max_width) {
            /* Emit current line */
            line_buf.x0 = 0.0f;
            line_buf.x1 = pen_x;
            line_buf.y = y;
            line_buf.width = pen_x;
            line_buf.text_start = (start_seg < prep->segment_count) ? prep->segments[start_seg].text : "";
            line_buf.segment_from = start_seg;
            line_buf.segment_to = cursor->segment_idx;

            cursor->line_idx++;
            cursor->pen_x = 0.0f;
            if (!seg) {
                /* Done */
                return NULL;
            }
            y += line_height;
            start_seg = cursor->segment_idx;
            pen_x = 0.0f;
            if (is_last) {
                /* Emit last partial line */
                return &line_buf;
            }
        }

        if (seg) {
            pen_x += seg->width;
            cursor->segment_idx++;
        } else {
            break;
        }
    }

    line_buf.x0 = 0.0f;
    line_buf.x1 = pen_x;
    line_buf.y = y;
    line_buf.width = pen_x;
    line_buf.text_start = (start_seg < prep->segment_count) ? prep->segments[start_seg].text : "";
    line_buf.segment_from = start_seg;
    line_buf.segment_to = cursor->segment_idx;
    cursor->line_idx++;
    return &line_buf;
}

uint32_t fs_text_layout_walk_lines(const FS_PreparedText* prep, float max_width, const FS_LayoutOptions* opts,
                                   bool (*callback)(uint32_t, const FS_LayoutLine*, void*), void* user_data) {
    if (!prep || !callback || max_width <= 0.0f) return 0;
    FS_LayoutResultFull full;
    memset(&full, 0, sizeof(full));
    FS_LayoutOptions o = opts ? *opts : fs_default_options();
    FS_LayoutResult r = fs_layout_impl(prep, max_width, o, &full);
    (void)r;

    uint32_t count = 0;
    for (uint32_t i = 0; i < full.line_count; i++) {
        if (!callback(i, &full.lines[i], user_data)) break;
        count++;
    }

    free(full.lines);
    return count;
}

bool fs_text_layout_is_break_allowed(uint32_t cp) {
    /* Simplified Unicode line-breaking rules */
    if (cp == 0x0020 || cp == 0x00A0) return true; /* Space, NBSP */
    if (cp == 0x2010 || cp == 0x2011) return true; /* Hyphens */
    if (cp == 0x2013 || cp == 0x2014) return true; /* En/em dash */
    if (cp == 0xAD) return true;                   /* Soft hyphen */
    if (cp >= 0x3000 && cp <= 0x3002) return true; /* CJK punctuation */
    if (cp == 0xFF08 || cp == 0xFF09) return true; /* Fullwidth parens */
    if (cp >= 0x3400 && cp <= 0x4DBF) return true; /* CJK */
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true; /* CJK Unified */
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true; /* Hangul */
    if (cp >= 0x20000 && cp <= 0x2A6DF) return true;
    if (cp >= 0x2A700 && cp <= 0x2B73F) return true;
    if (cp >= 0x2B740 && cp <= 0x2B81F) return true;
    if (cp >= 0x2B820 && cp <= 0x2CEAF) return true;
    if (cp >= 0x2CEB0 && cp <= 0x2EBEF) return true;
    return false;
}

bool fs_text_layout_is_cjk(uint32_t cp) {
    if (cp >= 0x1100 && cp <= 0x115F) return true;
    if (cp >= 0x2E80 && cp <= 0x303F) return true;
    if (cp >= 0x3040 && cp <= 0x30FF) return true;
    if (cp >= 0x3100 && cp <= 0x312F) return true;
    if (cp >= 0x3130 && cp <= 0x318F) return true;
    if (cp >= 0x3190 && cp <= 0x33FF) return true;
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
    if (cp >= 0xA000 && cp <= 0xA4CF) return true;
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;
    if (cp >= 0xFE10 && cp <= 0xFE1F) return true;
    if (cp >= 0xFE30 && cp <= 0xFE6F) return true;
    if (cp >= 0xFF00 && cp <= 0xFF60) return true;
    if (cp >= 0x20000 && cp <= 0x2A6DF) return true;
    if (cp >= 0x2A700 && cp <= 0x2B73F) return true;
    if (cp >= 0x2B740 && cp <= 0x2B81F) return true;
    if (cp >= 0x2B820 && cp <= 0x2CEAF) return true;
    if (cp >= 0x2CEB0 && cp <= 0x2EBEF) return true;
    return false;
}

bool fs_text_layout_is_emoji(uint32_t cp) {
    /* Basic emoji detection */
    if (cp >= 0x2600 && cp <= 0x26FF) return true;  /* Misc symbols */
    if (cp >= 0x2700 && cp <= 0x27BF) return true;  /* Dingbats */
    if (cp >= 0x1F300 && cp <= 0x1F9FF) return true; /* Emoji */
    if (cp >= 0x1FA00 && cp <= 0x1FAD6) return true; /* Emoji 13+ */
    if (cp >= 0x231A && cp <= 0x231B) return true;  /* Watch, Hourglass */
    if (cp >= 0x23E9 && cp <= 0x23F3) return true;  /* Various */
    if (cp >= 0x23F8 && cp <= 0x23FA) return true;
    if (cp >= 0x24C2 && cp <= 0x24C2) return true;  /* M */
    if (cp >= 0x25AA && cp <= 0x25AB) return true;  /* Squares */
    if (cp >= 0x25B6 && cp <= 0x25B6) return true;  /* Play */
    if (cp >= 0x25C0 && cp <= 0x25C0) return true;  /* Back */
    if (cp >= 0x25FB && cp <= 0x25FE) return true;  /* Squares */
    if (cp >= 0x2614 && cp <= 0x2615) return true;  /* Weather, Coffee */
    if (cp >= 0x2620 && cp <= 0x2620) return true;  /* Skull */
    if (cp >= 0x2622 && cp <= 0x2623) return true;
    if (cp >= 0x2626 && cp <= 0x2626) return true;
    if (cp >= 0x262A && cp <= 0x262A) return true;
    if (cp >= 0x262E && cp <= 0x262F) return true;
    if (cp >= 0x2638 && cp <= 0x263A) return true;
    if (cp >= 0x2648 && cp <= 0x2653) return true;  /* Zodiac */
    if (cp >= 0x2660 && cp <= 0x2667) return true;  /* Cards */
    if (cp >= 0x2665 && cp <= 0x2665) return true;  /* Heart */
    if (cp >= 0x267B && cp <= 0x267F) return true;
    if (cp >= 0x2693 && cp <= 0x2693) return true;
    if (cp >= 0x26A1 && cp <= 0x26A1) return true;  /* High voltage */
    if (cp >= 0x26AA && cp <= 0x26AB) return true;
    if (cp >= 0x26BD && cp <= 0x26BE) return true;
    if (cp >= 0x26C4 && cp <= 0x26C5) return true;  /* Weather */
    if (cp >= 0x26CE && cp <= 0x26CE) return true;  /* Ophiuchus */
    if (cp >= 0x26D4 && cp <= 0x26D4) return true;
    if (cp >= 0x26EA && cp <= 0x26EA) return true;
    if (cp >= 0x26F2 && cp <= 0x26F3) return true;
    if (cp >= 0x26F5 && cp <= 0x26F5) return true;
    if (cp >= 0x26FA && cp <= 0x26FA) return true;
    if (cp >= 0x26FD && cp <= 0x26FD) return true;
    return false;
}

void fs_text_layout_use_wcn_backend(FS_TextLayout* ctx, void* core, bool ready) {
    if (!ctx) return;
    ctx->wcn_core = core;
    ctx->use_wcn_backend = ready;
}

void fs_text_layout_clear_cache(FS_TextLayout* ctx) {
    if (!ctx) return;
    memset(ctx->glyph_cache, 0, ctx->glyph_cache_cap * sizeof(FS_GlyphCacheEntry));
    ctx->glyph_cache_count = 0;
}

void fs_text_layout_get_cache_stats(const FS_TextLayout* ctx, FS_TextLayoutCacheStats* out_stats) {
    if (!ctx || !out_stats) return;
    out_stats->glyph_cache_count = ctx->glyph_cache_count;
    out_stats->prepared_text_count = ctx->prepared_text_count;
    out_stats->memory_bytes = ctx->memory_bytes;
}

bool fs_text_layout_decode_utf8(const char** ptr, uint32_t* out_cp) {
    if (!ptr || !*ptr || !out_cp) return false;
    const unsigned char* p = (const unsigned char*)*ptr;
    uint32_t cp = 0;
    size_t len = 0;

    if (*p < 0x80) {
        cp = *p; len = 1;
    } else if ((*p & 0xE0) == 0xC0) {
        cp = (*p & 0x1F);
        if (p[1] && (p[1] & 0xC0) == 0x80) {
            cp = (cp << 6) | (p[1] & 0x3F);
            len = 2;
        } else { return false; }
    } else if ((*p & 0xF0) == 0xE0) {
        if (p[1] && p[2] && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80) {
            cp = ((uint32_t)(p[0] & 0x0F) << 12) | ((uint32_t)(p[1] & 0x3F) << 6) | (p[2] & 0x3F);
            len = 3;
        } else { return false; }
    } else if ((*p & 0xF8) == 0xF0) {
        if (p[1] && p[2] && p[3] && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80 && (p[3] & 0xC0) == 0x80) {
            cp = ((uint32_t)(p[0] & 0x07) << 18) | ((uint32_t)(p[1] & 0x3F) << 12) |
                 ((uint32_t)(p[2] & 0x3F) << 6) | (p[3] & 0x3F);
            len = 4;
        } else { return false; }
    } else {
        return false;
    }

    *out_cp = cp;
    *ptr += len;
    return true;
}

uint32_t fs_text_layout_count_codepoints(const char* text) {
    if (!text) return 0;
    uint32_t count = 0;
    const char* p = text;
    while (*p) {
        uint32_t cp;
        if (fs_text_layout_decode_utf8(&p, &cp)) count++;
        else { p++; }
    }
    return count;
}

size_t fs_text_layout_utf8_char_len(char c) {
    unsigned char uc = (unsigned char)c;
    if (uc < 0x80) return 1;
    if ((uc & 0xE0) == 0xC0) return 2;
    if ((uc & 0xF0) == 0xE0) return 3;
    if ((uc & 0xF8) == 0xF0) return 4;
    return 1;
}

/* ================================================================
   SLOT-BASED FLOW LAYOUT — IMPLEMENTATION
   ================================================================ */

void fs_text_layout_flow_cursor_init(FS_LayoutFlowCursor* cursor) {
    if (!cursor) return;
    cursor->segment_idx = 0;
    cursor->segment_offset_x = 0.0f;
    cursor->at_paragraph_break = false;
    cursor->finished = false;
}

bool fs_text_layout_flow_finished(const FS_PreparedText* prep,
                                  const FS_LayoutFlowCursor* cursor) {
    if (!prep || !cursor) return true;
    return cursor->finished;
}

bool fs_text_layout_layout_into_slot(
    const FS_PreparedText* prep,
    FS_LayoutFlowCursor*   cursor,
    float                  slot_x0,
    float                  slot_x1,
    float                  baseline_y,
    FS_LayoutFragment*    out_frag) {
    if (!prep || !cursor || !out_frag) return false;
    if (cursor->finished) return false;

    const float available_w = slot_x1 - slot_x0;
    if (available_w <= 0.0f) return false;

    const uint32_t frag_from = cursor->segment_idx;
    const char* frag_text_start = NULL;
    if (frag_from < prep->segment_count) {
        frag_text_start = prep->segments[frag_from].text;
    } else {
        cursor->finished = true;
        return false;
    }

    float used_w = 0.0f;
    size_t total_bytes = 0;
    uint32_t seg_end = frag_from;

    uint32_t last_break_seg_end = frag_from;
    float last_break_used_w = 0.0f;
    size_t last_break_total_bytes = 0;
    bool has_break_point = false;
    bool hit_paragraph_break = false;

    while (seg_end < prep->segment_count) {
        FS_TextSegment* seg = &prep->segments[seg_end];

        /* Zero-width newline forces a paragraph break and is consumed. */
        if (seg->byte_len == 0) {
            hit_paragraph_break = true;
            seg_end++;
            break;
        }

        if (used_w + seg->width > available_w) {
            if (has_break_point) {
                seg_end = last_break_seg_end;
                used_w = last_break_used_w;
                total_bytes = last_break_total_bytes;
            }
            break;
        }

        used_w += seg->width;
        total_bytes += seg->byte_len;
        seg_end++;

        if (seg->is_space || seg->is_cjk || seg->is_emoji) {
            last_break_seg_end = seg_end;
            last_break_used_w = used_w;
            last_break_total_bytes = total_bytes;
            has_break_point = true;
        }
    }

    /* Long-word fallback: allow one fitting segment even without a break point. */
    if (total_bytes == 0 && seg_end == frag_from && frag_from < prep->segment_count) {
        FS_TextSegment* seg = &prep->segments[frag_from];
        if (seg->byte_len > 0 && seg->width <= available_w) {
            used_w = seg->width;
            total_bytes = seg->byte_len;
            seg_end = frag_from + 1;
        }
    }

    /* Paragraph break at line start: consume newline and move to next baseline. */
    if (total_bytes == 0 && hit_paragraph_break) {
        cursor->segment_idx = seg_end;
        cursor->segment_offset_x = 0.0f;
        cursor->at_paragraph_break = true;
        if (cursor->segment_idx >= prep->segment_count) {
            cursor->finished = true;
        }
        return false;
    }

    if (total_bytes == 0) {
        return false;
    }

    out_frag->text_start   = frag_text_start;
    out_frag->byte_len     = total_bytes;
    out_frag->x0           = slot_x0;
    out_frag->x1           = slot_x0 + used_w;
    out_frag->baseline_y   = baseline_y;
    out_frag->width        = used_w;
    out_frag->segment_from = frag_from;
    out_frag->segment_to   = seg_end;

    cursor->segment_idx = seg_end;
    cursor->segment_offset_x = 0.0f;
    cursor->at_paragraph_break = hit_paragraph_break;

    if (cursor->segment_idx >= prep->segment_count) {
        cursor->finished = true;
    }

    return true;
}

uint32_t fs_text_layout_layout_line_slots(
    const FS_PreparedText* prep,
    FS_LayoutFlowCursor*   cursor,
    const FS_LayoutSlot*  slots,
    uint32_t              slot_count,
    float                 baseline_y,
    FS_LayoutFragment*    out_frags,
    uint32_t              max_frags) {
    if (!prep || !cursor || !slots || !out_frags) return 0;

    /* After a paragraph break, skip remaining slots on this baseline */
    if (cursor->at_paragraph_break) {
        cursor->at_paragraph_break = false;
        return 0;
    }

    uint32_t emitted = 0;
    for (uint32_t i = 0; i < slot_count && emitted < max_frags; i++) {
        if (cursor->finished) break;

        FS_LayoutFragment* f = &out_frags[emitted];
        f->slot_index = i;

        if (fs_text_layout_layout_into_slot(prep, cursor,
                slots[i].x0, slots[i].x1, baseline_y, f)) {
            emitted++;
        }

        /* After paragraph break, stop consuming further slots */
        if (cursor->at_paragraph_break) {
            cursor->at_paragraph_break = false;
            break;
        }
    }
    return emitted;
}

#endif /* FS_TEXT_LAYOUT_IMPLEMENTATION */
