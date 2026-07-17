// taffy.hpp - C++17 header-only CSS layout library (Flexbox, Grid, Block)
// Ported from the Rust "taffy" library by Dioxus Labs
// https://github.com/DioxusLabs/taffy
//
// Usage: #include "taffy.hpp"
// Namespace: taffy
//
// Feature flags (define before include):
//   TAFFY_FLEXBOX        - Flexbox layout (enabled by default)
//   TAFFY_GRID           - CSS Grid layout (enabled by default)
//   TAFFY_BLOCK_LAYOUT   - Block layout with float support (enabled by default)
//   TAFFY_FLOAT_LAYOUT   - Float support for block layout (enabled with block)
//   TAFFY_CONTENT_SIZE   - Track content size in Layout/LayoutOutput
//   TAFFY_CSS_PARSE      - CSS string parsing for Style
//   TAFFY_JSON           - JSON serialization/deserialization
//   TAFFY_DEBUG          - Debug tree printing
//
// calc() values are NOT supported in this port.

#pragma once
#ifndef TAFFY_HPP
#define TAFFY_HPP

// --- Default feature flags ---
#if !defined(TAFFY_FLEXBOX) && !defined(TAFFY_NO_FLEXBOX)
#define TAFFY_FLEXBOX
#endif
#if !defined(TAFFY_GRID) && !defined(TAFFY_NO_GRID)
#define TAFFY_GRID
#endif
#if !defined(TAFFY_BLOCK_LAYOUT) && !defined(TAFFY_NO_BLOCK_LAYOUT)
#define TAFFY_BLOCK_LAYOUT
#endif
#if defined(TAFFY_BLOCK_LAYOUT) && !defined(TAFFY_FLOAT_LAYOUT) && !defined(TAFFY_NO_FLOAT_LAYOUT)
#define TAFFY_FLOAT_LAYOUT
#endif
#if !defined(TAFFY_CONTENT_SIZE) && !defined(TAFFY_NO_CONTENT_SIZE)
#define TAFFY_CONTENT_SIZE
#endif

// --- Includes ---
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#ifdef TAFFY_CSS_PARSE
#include <sstream>
#include <stdexcept>
#endif
#ifdef TAFFY_JSON
#include <sstream>
#endif

namespace taffy {

// ===========================================================================
// Section 1: Axis enums
// ===========================================================================

enum class AbsoluteAxis : uint8_t { Horizontal, Vertical };
enum class AbstractAxis : uint8_t { Inline, Block };

inline constexpr AbsoluteAxis other_axis(AbsoluteAxis a) {
    return a == AbsoluteAxis::Horizontal ? AbsoluteAxis::Vertical : AbsoluteAxis::Horizontal;
}
inline constexpr AbstractAxis other(AbstractAxis a) {
    return a == AbstractAxis::Inline ? AbstractAxis::Block : AbstractAxis::Inline;
}
inline constexpr AbsoluteAxis as_abs_naive(AbstractAxis a) {
    return a == AbstractAxis::Inline ? AbsoluteAxis::Horizontal : AbsoluteAxis::Vertical;
}

// Forward-declare FlexDirection for geometry helpers
#ifdef TAFFY_FLEXBOX
enum class FlexDirection : uint8_t { Row, Column, RowReverse, ColumnReverse };
bool is_row(FlexDirection d);
bool is_reverse(FlexDirection d);
#endif

// ===========================================================================
// Section 2: Geometry Primitives (proper order: Size, Point, Line, Rect, MinMax)
// Ref: src/geometry.rs
// ===========================================================================

// --- Size<T> ---
template <typename T> struct Size;
template <typename T, typename U>
constexpr auto operator+(const Size<T>& a, const Size<U>& b) -> Size<decltype(a.width + b.width)>;
template <typename T, typename U>
constexpr auto operator-(const Size<T>& a, const Size<U>& b) -> Size<decltype(a.width - b.width)>;

template <typename T>
struct Size {
    T width{};
    T height{};
    constexpr Size() = default;
    constexpr Size(T w, T h) : width(w), height(h) {}
    constexpr bool operator==(const Size& o) const { return width == o.width && height == o.height; }
    constexpr bool operator!=(const Size& o) const { return !(*this == o); }

    template <typename F>
    constexpr auto map(F&& f) const -> Size<std::invoke_result_t<F, T>> {
        return {f(width), f(height)};
    }
    template <typename F>
    constexpr Size map_width(F&& f) const { return {f(width), height}; }
    template <typename F>
    constexpr Size map_height(F&& f) const { return {width, f(height)}; }

    template <typename Other, typename F>
    constexpr auto zip_map(const Size<Other>& o, F&& f) const -> Size<std::invoke_result_t<F, T, Other>> {
        return {f(width, o.width), f(height, o.height)};
    }

    constexpr T get_abs(AbsoluteAxis axis) const {
        return axis == AbsoluteAxis::Horizontal ? width : height;
    }
    constexpr T get(AbstractAxis axis) const {
        return axis == AbstractAxis::Inline ? width : height;
    }
    constexpr Size& set(AbstractAxis axis, T value) {
        (axis == AbstractAxis::Inline ? width : height) = value;
        return *this;
    }
    constexpr Size with(AbstractAxis axis, T value) const {
        Size r = *this; r.set(axis, value); return r;
    }

#ifdef TAFFY_FLEXBOX
    constexpr T main(FlexDirection dir) const;
    constexpr T cross(FlexDirection dir) const;
    constexpr Size& set_main(FlexDirection dir, T value);
    constexpr Size& set_cross(FlexDirection dir, T value);
    constexpr Size with_main(FlexDirection dir, T value) const;
    constexpr Size with_cross(FlexDirection dir, T value) const;
#endif
};

template <typename T, typename U>
constexpr auto operator+(const Size<T>& a, const Size<U>& b) -> Size<decltype(a.width + b.width)> {
    return {a.width + b.width, a.height + b.height};
}
template <typename T, typename U>
constexpr auto operator-(const Size<T>& a, const Size<U>& b) -> Size<decltype(a.width - b.width)> {
    return {a.width - b.width, a.height - b.height};
}

// --- Size<float> specialization ---
template <>
struct Size<float> {
    float width{};
    float height{};
    constexpr Size() = default;
    constexpr Size(float w, float h) : width(w), height(h) {}
    constexpr bool operator==(const Size& o) const { return width == o.width && height == o.height; }
    constexpr bool operator!=(const Size& o) const { return !(*this == o); }
    constexpr Size f32_max(const Size& rhs) const { return {std::max(width, rhs.width), std::max(height, rhs.height)}; }
    constexpr Size f32_min(const Size& rhs) const { return {std::min(width, rhs.width), std::min(height, rhs.height)}; }
    constexpr bool has_non_zero_area() const { return width > 0.0f && height > 0.0f; }
    constexpr float get_abs(AbsoluteAxis axis) const { return axis == AbsoluteAxis::Horizontal ? width : height; }
    constexpr float get(AbstractAxis axis) const { return axis == AbstractAxis::Inline ? width : height; }
    constexpr float main(FlexDirection dir) const { return is_row(dir) ? width : height; }
    constexpr float cross(FlexDirection dir) const { return is_row(dir) ? height : width; }
    constexpr Size& set_main(FlexDirection dir, float v) { (is_row(dir) ? width : height) = v; return *this; }
    constexpr Size& set_cross(FlexDirection dir, float v) { (is_row(dir) ? height : width) = v; return *this; }
    constexpr Size with_main(FlexDirection dir, float v) const { Size r = *this; r.set_main(dir, v); return r; }
    constexpr Size with_cross(FlexDirection dir, float v) const { Size r = *this; r.set_cross(dir, v); return r; }
};
inline constexpr Size<float> Size_f32_ZERO{0.0f, 0.0f};

// --- Size<optional<float>> specialization ---
template <>
struct Size<std::optional<float>> {
    std::optional<float> width{};
    std::optional<float> height{};
    constexpr Size() = default;
    constexpr Size(std::optional<float> w, std::optional<float> h) : width(w), height(h) {}
    constexpr Size(float w, float h) : width(w), height(h) {}
    constexpr bool operator==(const Size& o) const { return width == o.width && height == o.height; }
    constexpr bool operator!=(const Size& o) const { return !(*this == o); }

    Size<float> unwrap_or(const Size<float>& alt) const {
        return {width.value_or(alt.width), height.value_or(alt.height)};
    }
    Size or_(const Size& alt) const {
        return {width ? width : alt.width, height ? height : alt.height};
    }
    bool both_axis_defined() const { return width.has_value() && height.has_value(); }
    Size maybe_apply_aspect_ratio(std::optional<float> ratio) const {
        if (!ratio) return *this;
        if (width && !height) return {width, *width / *ratio};
        if (!width && height) return {*height * *ratio, height};
        return *this;
    }

    std::optional<float> main(FlexDirection dir) const { return is_row(dir) ? width : height; }
    std::optional<float> cross(FlexDirection dir) const { return is_row(dir) ? height : width; }
    Size& set_main(FlexDirection dir, std::optional<float> v) { (is_row(dir) ? width : height) = v; return *this; }
    Size& set_cross(FlexDirection dir, std::optional<float> v) { (is_row(dir) ? height : width) = v; return *this; }
    Size with_main(FlexDirection dir, std::optional<float> v) const { Size r = *this; r.set_main(dir, v); return r; }
    Size with_cross(FlexDirection dir, std::optional<float> v) const { Size r = *this; r.set_cross(dir, v); return r; }
};
inline constexpr Size<std::optional<float>> Size_optf32_NONE{std::nullopt, std::nullopt};

// Flex-direction helpers for Size<optional<float>>
#ifdef TAFFY_FLEXBOX
inline std::optional<float> optf32_main(const Size<std::optional<float>>& s, FlexDirection dir) {
    return is_row(dir) ? s.width : s.height;
}
inline std::optional<float> optf32_cross(const Size<std::optional<float>>& s, FlexDirection dir) {
    return is_row(dir) ? s.height : s.width;
}
inline void optf32_set_main(Size<std::optional<float>>& s, FlexDirection dir, std::optional<float> v) {
    (is_row(dir) ? s.width : s.height) = v;
}
inline void optf32_set_cross(Size<std::optional<float>>& s, FlexDirection dir, std::optional<float> v) {
    (is_row(dir) ? s.height : s.width) = v;
}
inline Size<std::optional<float>> optf32_with_main(const Size<std::optional<float>>& s, FlexDirection dir, std::optional<float> v) {
    Size<std::optional<float>> r = s; optf32_set_main(r, dir, v); return r;
}
inline Size<std::optional<float>> optf32_with_cross(const Size<std::optional<float>>& s, FlexDirection dir, std::optional<float> v) {
    Size<std::optional<float>> r = s; optf32_set_cross(r, dir, v); return r;
}
#endif // TAFFY_FLEXBOX

// --- Point<T> ---
template <typename T>
struct Point {
    T x{};
    T y{};
    constexpr Point() = default;
    constexpr Point(T x_, T y_) : x(x_), y(y_) {}
    constexpr bool operator==(const Point& o) const { return x == o.x && y == o.y; }
    constexpr bool operator!=(const Point& o) const { return !(*this == o); }
    template <typename F>
    constexpr auto map(F&& f) const -> Point<std::invoke_result_t<F, T>> { return {f(x), f(y)}; }
    constexpr Point transpose() const { return {y, x}; }
    constexpr T get(AbstractAxis axis) const { return axis == AbstractAxis::Inline ? x : y; }
    constexpr Point& set(AbstractAxis axis, T value) {
        (axis == AbstractAxis::Inline ? x : y) = value; return *this;
    }
#ifdef TAFFY_FLEXBOX
    constexpr T main(FlexDirection dir) const;
    constexpr T cross(FlexDirection dir) const;
#endif
};

template <typename T, typename U>
constexpr auto operator+(const Point<T>& a, const Point<U>& b) -> Point<decltype(a.x + b.x)> {
    return {a.x + b.x, a.y + b.y};
}

template <>
struct Point<float> {
    float x{};
    float y{};
    constexpr Point() = default;
    constexpr Point(float x_, float y_) : x(x_), y(y_) {}
    constexpr bool operator==(const Point& o) const { return x == o.x && y == o.y; }
    constexpr bool operator!=(const Point& o) const { return !(*this == o); }
    constexpr Point transpose() const { return {y, x}; }
#ifdef TAFFY_FLEXBOX
    constexpr float main(FlexDirection dir) const;
    constexpr float cross(FlexDirection dir) const;
#endif
};
inline constexpr Point<float> Point_f32_ZERO{0.0f, 0.0f};

template <>
struct Point<std::optional<float>> {
    std::optional<float> x{};
    std::optional<float> y{};
    constexpr Point() = default;
    constexpr Point(std::optional<float> x_, std::optional<float> y_) : x(x_), y(y_) {}
    constexpr bool operator==(const Point& o) const { return x == o.x && y == o.y; }
    constexpr bool operator!=(const Point& o) const { return !(*this == o); }
};
inline constexpr Point<std::optional<float>> Point_optf32_NONE{std::nullopt, std::nullopt};

// --- Line<T> ---
template <typename T>
struct Line {
    T start{};
    T end{};
    constexpr Line() = default;
    constexpr Line(T s, T e) : start(s), end(e) {}
    constexpr bool operator==(const Line& o) const { return start == o.start && end == o.end; }
    constexpr bool operator!=(const Line& o) const { return !(*this == o); }
    template <typename F>
    constexpr auto map(F&& f) const -> Line<std::invoke_result_t<F, T>> { return {f(start), f(end)}; }
    template <typename U = T>
    constexpr auto sum() const -> decltype(std::declval<U>() + std::declval<U>()) { return start + end; }
};

template <>
struct Line<bool> {
    bool start{};
    bool end{};
    constexpr Line() = default;
    constexpr Line(bool s, bool e) : start(s), end(e) {}
    constexpr bool operator==(const Line& o) const { return start == o.start && end == o.end; }
    constexpr bool operator!=(const Line& o) const { return !(*this == o); }
};
inline constexpr Line<bool> Line_bool_TRUE{true, true};
inline constexpr Line<bool> Line_bool_FALSE{false, false};

// --- Rect<T> ---
template <typename T>
struct Rect {
    T left{};
    T right{};
    T top{};
    T bottom{};
    constexpr Rect() = default;
    constexpr Rect(T l, T r, T t, T b) : left(l), right(r), top(t), bottom(b) {}
    constexpr bool operator==(const Rect& o) const { return left == o.left && right == o.right && top == o.top && bottom == o.bottom; }
    constexpr bool operator!=(const Rect& o) const { return !(*this == o); }

    template <typename U, typename F>
    constexpr Rect<std::invoke_result_t<F, T, U>> zip_size(const Size<U>& size, F&& f) const {
        return {f(left, size.width), f(right, size.width), f(top, size.height), f(bottom, size.height)};
    }
    template <typename F>
    constexpr auto map(F&& f) const -> Rect<std::invoke_result_t<F, T>> {
        return {f(left), f(right), f(top), f(bottom)};
    }
    constexpr Line<T> horizontal_components() const { return {left, right}; }
    constexpr Line<T> vertical_components() const { return {top, bottom}; }

    constexpr T horizontal_axis_sum() const { return left + right; }
    constexpr T vertical_axis_sum() const { return top + bottom; }
    constexpr Size<T> sum_axes() const { return {left + right, top + bottom}; }

#ifdef TAFFY_FLEXBOX
    constexpr T main_axis_sum(FlexDirection dir) const;
    constexpr T cross_axis_sum(FlexDirection dir) const;
    constexpr T main_start(FlexDirection dir) const;
    constexpr T main_end(FlexDirection dir) const;
    constexpr T cross_start(FlexDirection dir) const;
    constexpr T cross_end(FlexDirection dir) const;
#endif
};

template <typename T, typename U>
constexpr auto operator+(const Rect<T>& a, const Rect<U>& b) -> Rect<decltype(a.left + b.left)> {
    return {a.left + b.left, a.right + b.right, a.top + b.top, a.bottom + b.bottom};
}

// --- MinMax<Min, Max> ---
template <typename Min, typename Max>
struct MinMax {
    Min min{};
    Max max{};
    constexpr MinMax() = default;
    constexpr MinMax(Min mn, Max mx) : min(mn), max(mx) {}
    constexpr bool operator==(const MinMax& o) const { return min == o.min && max == o.max; }
    constexpr bool operator!=(const MinMax& o) const { return !(*this == o); }
};

// --- FlexDirection and flex-direction-aware helpers ---
#ifdef TAFFY_FLEXBOX
inline bool is_row(FlexDirection d) {
    return d == FlexDirection::Row || d == FlexDirection::RowReverse;
}
inline bool is_reverse(FlexDirection d) {
    return d == FlexDirection::RowReverse || d == FlexDirection::ColumnReverse;
}

// Size<T> flex helpers
template <typename T>
constexpr T Size<T>::main(FlexDirection dir) const { return is_row(dir) ? width : height; }
template <typename T>
constexpr T Size<T>::cross(FlexDirection dir) const { return is_row(dir) ? height : width; }
template <typename T>
constexpr Size<T>& Size<T>::set_main(FlexDirection dir, T value) {
    (is_row(dir) ? width : height) = value; return *this;
}
template <typename T>
constexpr Size<T>& Size<T>::set_cross(FlexDirection dir, T value) {
    (is_row(dir) ? height : width) = value; return *this;
}
template <typename T>
constexpr Size<T> Size<T>::with_main(FlexDirection dir, T value) const {
    Size r = *this; r.set_main(dir, value); return r;
}
template <typename T>
constexpr Size<T> Size<T>::with_cross(FlexDirection dir, T value) const {
    Size r = *this; r.set_cross(dir, value); return r;
}

// Point<T> flex helpers
template <typename T>
constexpr T Point<T>::main(FlexDirection dir) const { return is_row(dir) ? x : y; }
template <typename T>
constexpr T Point<T>::cross(FlexDirection dir) const { return is_row(dir) ? y : x; }
inline constexpr float Point<float>::main(FlexDirection dir) const { return is_row(dir) ? x : y; }
inline constexpr float Point<float>::cross(FlexDirection dir) const { return is_row(dir) ? y : x; }

// Rect<T> flex helpers
template <typename T>
constexpr T Rect<T>::main_axis_sum(FlexDirection dir) const {
    return is_row(dir) ? horizontal_axis_sum() : vertical_axis_sum();
}
template <typename T>
constexpr T Rect<T>::cross_axis_sum(FlexDirection dir) const {
    return is_row(dir) ? vertical_axis_sum() : horizontal_axis_sum();
}
template <typename T>
constexpr T Rect<T>::main_start(FlexDirection dir) const { return is_row(dir) ? left : top; }
template <typename T>
constexpr T Rect<T>::main_end(FlexDirection dir) const { return is_row(dir) ? right : bottom; }
template <typename T>
constexpr T Rect<T>::cross_start(FlexDirection dir) const { return is_row(dir) ? top : left; }
template <typename T>
constexpr T Rect<T>::cross_end(FlexDirection dir) const { return is_row(dir) ? bottom : right; }
#endif // TAFFY_FLEXBOX

// ===========================================================================
// Section 3: CompactLength
// Ref: src/style/compact_length.rs
// ===========================================================================

namespace detail {
    inline uint32_t f32_to_bits(float v) { uint32_t b; std::memcpy(&b, &v, 4); return b; }
    inline float f32_from_bits(uint32_t b) { float v; std::memcpy(&v, &b, 4); return v; }
}

struct CompactLength {
    static constexpr size_t LENGTH_TAG              = 0b00000001;
    static constexpr size_t PERCENT_TAG             = 0b00000010;
    static constexpr size_t AUTO_TAG                = 0b00000011;
    static constexpr size_t FR_TAG                  = 0b00000100;
    static constexpr size_t MIN_CONTENT_TAG         = 0b00000111;
    static constexpr size_t MAX_CONTENT_TAG         = 0b00001111;
    static constexpr size_t FIT_CONTENT_PX_TAG      = 0b00010111;
    static constexpr size_t FIT_CONTENT_PERCENT_TAG = 0b00011111;

    uint64_t data{};
    constexpr CompactLength() = default;

    static constexpr CompactLength length(float val) {
        CompactLength cl; cl.data = (uint64_t(detail::f32_to_bits(val)) << 32) | LENGTH_TAG; return cl;
    }
    // ZERO is length(0.0f) — must be defined after length() exists
    static constexpr CompactLength zero_() { return length(0.0f); }
    static constexpr CompactLength percent(float val) {
        CompactLength cl; cl.data = (uint64_t(detail::f32_to_bits(val)) << 32) | PERCENT_TAG; return cl;
    }
    static constexpr CompactLength auto_() {
        CompactLength cl; cl.data = AUTO_TAG; return cl;
    }
    static constexpr CompactLength fr(float val) {
        CompactLength cl; cl.data = (uint64_t(detail::f32_to_bits(val)) << 32) | FR_TAG; return cl;
    }
    static constexpr CompactLength min_content() {
        CompactLength cl; cl.data = MIN_CONTENT_TAG; return cl;
    }
    static constexpr CompactLength max_content() {
        CompactLength cl; cl.data = MAX_CONTENT_TAG; return cl;
    }
    static constexpr CompactLength fit_content_px(float limit) {
        CompactLength cl; cl.data = (uint64_t(detail::f32_to_bits(limit)) << 32) | FIT_CONTENT_PX_TAG; return cl;
    }
    static constexpr CompactLength fit_content_percent(float limit) {
        CompactLength cl; cl.data = (uint64_t(detail::f32_to_bits(limit)) << 32) | FIT_CONTENT_PERCENT_TAG; return cl;
    }

    constexpr size_t tag() const { return data & 0xFF; }
    constexpr float value() const { return detail::f32_from_bits(uint32_t(data >> 32)); }
    constexpr bool operator==(const CompactLength& o) const { return data == o.data; }
    constexpr bool operator!=(const CompactLength& o) const { return data != o.data; }
    constexpr bool is_zero() const { return *this == zero_(); }
    constexpr bool is_auto() const { return tag() == AUTO_TAG; }
    constexpr bool is_fr() const { return tag() == FR_TAG; }
    constexpr bool is_length_or_percentage() const { return tag() == LENGTH_TAG || tag() == PERCENT_TAG; }
    constexpr bool is_min_content() const { return tag() == MIN_CONTENT_TAG; }
    constexpr bool is_max_content() const { return tag() == MAX_CONTENT_TAG; }
    constexpr bool is_fit_content() const { return tag() == FIT_CONTENT_PX_TAG || tag() == FIT_CONTENT_PERCENT_TAG; }
    constexpr bool is_max_or_fit_content() const {
        return tag() == MAX_CONTENT_TAG || tag() == FIT_CONTENT_PX_TAG || tag() == FIT_CONTENT_PERCENT_TAG;
    }
    constexpr bool is_max_content_alike() const {
        return tag() == AUTO_TAG || tag() == MAX_CONTENT_TAG || tag() == FIT_CONTENT_PX_TAG || tag() == FIT_CONTENT_PERCENT_TAG;
    }
    constexpr bool is_min_or_max_content() const { return tag() == MIN_CONTENT_TAG || tag() == MAX_CONTENT_TAG; }
    constexpr bool is_intrinsic() const {
        return tag() == AUTO_TAG || tag() == MIN_CONTENT_TAG || tag() == MAX_CONTENT_TAG
            || tag() == FIT_CONTENT_PX_TAG || tag() == FIT_CONTENT_PERCENT_TAG;
    }
    constexpr bool uses_percentage() const { return tag() == PERCENT_TAG || tag() == FIT_CONTENT_PERCENT_TAG; }

    std::optional<float> resolved_percentage_size(float parent_size) const {
        if (tag() == PERCENT_TAG) return (value() * parent_size) / 100.0f;
        return std::nullopt;
    }
};

// ===========================================================================
// Section 4: Dimension Types
// Ref: src/style/dimension.rs
// ===========================================================================

struct LengthPercentage {
    CompactLength inner;
    constexpr LengthPercentage() : inner(CompactLength::length(0.0f)) {}
    constexpr explicit LengthPercentage(CompactLength cl) : inner(cl) {}
    static constexpr LengthPercentage length(float v) { return LengthPercentage(CompactLength::length(v)); }
    static constexpr LengthPercentage percent(float v) { return LengthPercentage(CompactLength::percent(v)); }
    constexpr size_t tag() const { return inner.tag(); }
    constexpr float value() const { return inner.value(); }
    constexpr bool is_zero() const { return inner.is_zero(); }
    constexpr bool uses_percentage() const { return inner.uses_percentage(); }
    std::optional<float> resolved_percentage_size(float ps) const { return inner.resolved_percentage_size(ps); }
    constexpr bool operator==(const LengthPercentage& o) const { return inner == o.inner; }
    constexpr bool operator!=(const LengthPercentage& o) const { return inner != o.inner; }
};

struct LengthPercentageAuto {
    CompactLength inner;
    constexpr LengthPercentageAuto() : inner(CompactLength::length(0.0f)) {}
    constexpr explicit LengthPercentageAuto(CompactLength cl) : inner(cl) {}
    static constexpr LengthPercentageAuto length(float v) { return LengthPercentageAuto(CompactLength::length(v)); }
    static constexpr LengthPercentageAuto percent(float v) { return LengthPercentageAuto(CompactLength::percent(v)); }
    static constexpr LengthPercentageAuto auto_() { return LengthPercentageAuto(CompactLength::auto_()); }
    constexpr size_t tag() const { return inner.tag(); }
    constexpr float value() const { return inner.value(); }
    constexpr bool is_auto() const { return inner.is_auto(); }
    constexpr bool is_zero() const { return inner.is_zero(); }
    constexpr bool uses_percentage() const { return inner.uses_percentage(); }
    std::optional<float> resolve_to_option(float context) const {
        switch (inner.tag()) {
            case CompactLength::LENGTH_TAG: return inner.value();
            case CompactLength::PERCENT_TAG: return (context * inner.value()) / 100.0f;
            case CompactLength::AUTO_TAG: return std::nullopt;
            default: return std::nullopt;
        }
    }
    constexpr bool operator==(const LengthPercentageAuto& o) const { return inner == o.inner; }
    constexpr bool operator!=(const LengthPercentageAuto& o) const { return inner != o.inner; }
};

struct Dimension {
    CompactLength inner;
    constexpr Dimension() : inner(CompactLength::length(0.0f)) {}
    constexpr explicit Dimension(CompactLength cl) : inner(cl) {}
    constexpr Dimension(LengthPercentage lp) : inner(lp.inner) {}
    constexpr Dimension(LengthPercentageAuto lpa) : inner(lpa.inner) {}
    static constexpr Dimension length(float v) { return Dimension(CompactLength::length(v)); }
    static constexpr Dimension percent(float v) { return Dimension(CompactLength::percent(v)); }
    static constexpr Dimension auto_() { return Dimension(CompactLength::auto_()); }
    static constexpr Dimension fr(float v) { return Dimension(CompactLength::fr(v)); }
    static constexpr Dimension min_content() { return Dimension(CompactLength::min_content()); }
    static constexpr Dimension max_content() { return Dimension(CompactLength::max_content()); }
    static constexpr Dimension fit_content_px(float l) { return Dimension(CompactLength::fit_content_px(l)); }
    static constexpr Dimension fit_content_percent(float l) { return Dimension(CompactLength::fit_content_percent(l)); }
    constexpr size_t tag() const { return inner.tag(); }
    constexpr float value() const { return inner.value(); }
    constexpr bool is_auto() const { return inner.is_auto(); }
    constexpr bool is_zero() const { return inner.is_zero(); }
    constexpr bool is_fr() const { return inner.is_fr(); }
    constexpr bool is_intrinsic() const { return inner.is_intrinsic(); }
    constexpr bool is_max_content_alike() const { return inner.is_max_content_alike(); }
    constexpr bool uses_percentage() const { return inner.uses_percentage(); }
    std::optional<float> into_option() const {
        if (inner.tag() == CompactLength::LENGTH_TAG) return inner.value();
        return std::nullopt;
    }
    std::optional<float> resolve_to_option(float context) const {
        switch (inner.tag()) {
            case CompactLength::LENGTH_TAG: return inner.value();
            case CompactLength::PERCENT_TAG: return (context * inner.value()) / 100.0f;
            case CompactLength::AUTO_TAG: return std::nullopt;
            default: return std::nullopt;
        }
    }
    constexpr bool operator==(const Dimension& o) const { return inner == o.inner; }
    constexpr bool operator!=(const Dimension& o) const { return inner != o.inner; }
};

// ===========================================================================
// Section 5: Alignment Types
// Ref: src/style/alignment.rs
// ===========================================================================

enum class AlignItemsKeyword : uint8_t { Start, End, FlexStart, FlexEnd, Center, Baseline, Stretch };
enum class AlignContentKeyword : uint8_t { Start, End, FlexStart, FlexEnd, Center, Stretch, SpaceBetween, SpaceEvenly, SpaceAround };
enum class AlignmentSafety : uint8_t { Unsafe, Safe };

struct AlignItems {
    AlignItemsKeyword keyword{AlignItemsKeyword::Start};
    AlignmentSafety safety{AlignmentSafety::Unsafe};
    constexpr AlignItems() = default;
    constexpr AlignItems(AlignItemsKeyword kw, AlignmentSafety s) : keyword(kw), safety(s) {}
    constexpr bool is_safe() const { return safety == AlignmentSafety::Safe; }
    constexpr AlignItemsKeyword kw() const { return keyword; }
    constexpr bool operator==(const AlignItems& o) const { return keyword == o.keyword && safety == o.safety; }
    constexpr bool operator!=(const AlignItems& o) const { return !(*this == o); }
};
inline constexpr AlignItems AlignItems_START{AlignItemsKeyword::Start, AlignmentSafety::Unsafe};
inline constexpr AlignItems AlignItems_END{AlignItemsKeyword::End, AlignmentSafety::Unsafe};
inline constexpr AlignItems AlignItems_FLEX_START{AlignItemsKeyword::FlexStart, AlignmentSafety::Unsafe};
inline constexpr AlignItems AlignItems_FLEX_END{AlignItemsKeyword::FlexEnd, AlignmentSafety::Unsafe};
inline constexpr AlignItems AlignItems_CENTER{AlignItemsKeyword::Center, AlignmentSafety::Unsafe};
inline constexpr AlignItems AlignItems_BASELINE{AlignItemsKeyword::Baseline, AlignmentSafety::Unsafe};
inline constexpr AlignItems AlignItems_STRETCH{AlignItemsKeyword::Stretch, AlignmentSafety::Unsafe};
inline constexpr AlignItems AlignItems_SAFE_START{AlignItemsKeyword::Start, AlignmentSafety::Safe};
inline constexpr AlignItems AlignItems_SAFE_END{AlignItemsKeyword::End, AlignmentSafety::Safe};
inline constexpr AlignItems AlignItems_SAFE_FLEX_START{AlignItemsKeyword::FlexStart, AlignmentSafety::Safe};
inline constexpr AlignItems AlignItems_SAFE_FLEX_END{AlignItemsKeyword::FlexEnd, AlignmentSafety::Safe};
inline constexpr AlignItems AlignItems_SAFE_CENTER{AlignItemsKeyword::Center, AlignmentSafety::Safe};

struct AlignContent {
    AlignContentKeyword keyword{AlignContentKeyword::Start};
    AlignmentSafety safety{AlignmentSafety::Unsafe};
    constexpr AlignContent() = default;
    constexpr AlignContent(AlignContentKeyword kw, AlignmentSafety s) : keyword(kw), safety(s) {}
    constexpr bool is_safe() const { return safety == AlignmentSafety::Safe; }
    constexpr AlignContentKeyword kw() const { return keyword; }
    AlignContentKeyword reversed() const {
        switch (keyword) {
            case AlignContentKeyword::Start: return AlignContentKeyword::End;
            case AlignContentKeyword::End: return AlignContentKeyword::Start;
            case AlignContentKeyword::FlexStart: return AlignContentKeyword::FlexEnd;
            case AlignContentKeyword::FlexEnd: return AlignContentKeyword::FlexStart;
            case AlignContentKeyword::Stretch: return AlignContentKeyword::End;
            default: return keyword;
        }
    }
    constexpr bool operator==(const AlignContent& o) const { return keyword == o.keyword && safety == o.safety; }
    constexpr bool operator!=(const AlignContent& o) const { return !(*this == o); }
};
inline constexpr AlignContent AlignContent_START{AlignContentKeyword::Start, AlignmentSafety::Unsafe};
inline constexpr AlignContent AlignContent_END{AlignContentKeyword::End, AlignmentSafety::Unsafe};
inline constexpr AlignContent AlignContent_FLEX_START{AlignContentKeyword::FlexStart, AlignmentSafety::Unsafe};
inline constexpr AlignContent AlignContent_FLEX_END{AlignContentKeyword::FlexEnd, AlignmentSafety::Unsafe};
inline constexpr AlignContent AlignContent_CENTER{AlignContentKeyword::Center, AlignmentSafety::Unsafe};
inline constexpr AlignContent AlignContent_STRETCH{AlignContentKeyword::Stretch, AlignmentSafety::Unsafe};
inline constexpr AlignContent AlignContent_SPACE_BETWEEN{AlignContentKeyword::SpaceBetween, AlignmentSafety::Unsafe};
inline constexpr AlignContent AlignContent_SPACE_EVENLY{AlignContentKeyword::SpaceEvenly, AlignmentSafety::Unsafe};
inline constexpr AlignContent AlignContent_SPACE_AROUND{AlignContentKeyword::SpaceAround, AlignmentSafety::Unsafe};
inline constexpr AlignContent AlignContent_SAFE_START{AlignContentKeyword::Start, AlignmentSafety::Safe};
inline constexpr AlignContent AlignContent_SAFE_END{AlignContentKeyword::End, AlignmentSafety::Safe};
inline constexpr AlignContent AlignContent_SAFE_FLEX_START{AlignContentKeyword::FlexStart, AlignmentSafety::Safe};
inline constexpr AlignContent AlignContent_SAFE_FLEX_END{AlignContentKeyword::FlexEnd, AlignmentSafety::Safe};
inline constexpr AlignContent AlignContent_SAFE_CENTER{AlignContentKeyword::Center, AlignmentSafety::Safe};

using AlignSelf = AlignItems;
using JustifyItems = AlignItems;
using JustifySelf = AlignItems;
using JustifyContent = AlignContent;

// ===========================================================================
// Section 6: Style Enums
// Ref: src/style/mod.rs, flex.rs, grid.rs, block.rs, float.rs
// ===========================================================================

enum class Display : uint8_t {
#ifdef TAFFY_BLOCK_LAYOUT
    Block,
#endif
#ifdef TAFFY_FLEXBOX
    Flex,
#endif
#ifdef TAFFY_GRID
    Grid,
#endif
    None
};

enum class BoxGenerationMode : uint8_t { Normal, None };
enum class Position : uint8_t { Relative, Absolute };
enum class BoxSizing : uint8_t { BorderBox, ContentBox };
enum class Overflow : uint8_t { Visible, Clip, Hidden, Scroll };
enum class Direction : uint8_t { Ltr, Rtl };

inline bool is_scroll_container(Overflow o) { return o == Overflow::Hidden || o == Overflow::Scroll; }
inline bool is_rtl(Direction d) { return d == Direction::Rtl; }

#ifdef TAFFY_FLEXBOX
enum class FlexWrap : uint8_t { NoWrap, Wrap, WrapReverse };
#endif

#ifdef TAFFY_BLOCK_LAYOUT
enum class TextAlign : uint8_t { Auto, LegacyLeft, LegacyRight, LegacyCenter };
#endif

#ifdef TAFFY_FLOAT_LAYOUT
enum class Float : uint8_t { None, Left, Right };
enum class Clear : uint8_t { None, Left, Right, Both };
enum class FloatDirection : uint8_t { Left, Right };
#endif

#ifdef TAFFY_GRID
enum class GridAutoFlow : uint8_t { Row, Column, RowDense, ColumnDense };
inline bool is_dense(GridAutoFlow f) { return f == GridAutoFlow::RowDense || f == GridAutoFlow::ColumnDense; }
inline AbsoluteAxis primary_axis(GridAutoFlow f) {
    return (f == GridAutoFlow::Row || f == GridAutoFlow::RowDense) ? AbsoluteAxis::Horizontal : AbsoluteAxis::Vertical;
}
#endif

// ===========================================================================
// Section 7: AvailableSpace
// Ref: src/style/available_space.rs
// ===========================================================================

struct AvailableSpace {
    enum Type : uint8_t { Definite, MinContent, MaxContent };
    Type type{MaxContent};
    float value{0.0f};

    constexpr AvailableSpace() = default;
    constexpr AvailableSpace(Type t, float v = 0.0f) : type(t), value(v) {}
    static constexpr AvailableSpace definite(float v) { return {Definite, v}; }
    static constexpr AvailableSpace min_content() { return {MinContent, 0.0f}; }
    static constexpr AvailableSpace max_content() { return {MaxContent, 0.0f}; }

    constexpr bool is_definite() const { return type == Definite; }
    std::optional<float> into_option() const { return type == Definite ? std::optional<float>(value) : std::nullopt; }
    float unwrap_or(float default_val) const { return type == Definite ? value : default_val; }
    float unwrap() const { assert(type == Definite); return value; }

    AvailableSpace or_(AvailableSpace default_val) const { return type == Definite ? *this : default_val; }
    AvailableSpace maybe_set(std::optional<float> v) const { return v ? definite(*v) : *this; }

    AvailableSpace map_definite_value(std::function<float(float)> f) const {
        return type == Definite ? definite(f(value)) : *this;
    }

    float compute_free_space(float used_space) const {
        switch (type) {
            case MaxContent: return std::numeric_limits<float>::infinity();
            case MinContent: return 0.0f;
            case Definite: return value - used_space;
        }
        return 0.0f;
    }

    bool is_roughly_equal(AvailableSpace other) const {
        if (type != other.type) return false;
        if (type == Definite) return std::abs(value - other.value) < std::numeric_limits<float>::epsilon();
        return true;
    }

    constexpr bool operator==(const AvailableSpace& o) const { return type == o.type && (type != Definite || value == o.value); }
    constexpr bool operator!=(const AvailableSpace& o) const { return !(*this == o); }
};

inline Size<std::optional<float>> into_options(const Size<AvailableSpace>& s) {
    return {s.width.into_option(), s.height.into_option()};
}

// AvailableSpace maybe-math helpers
inline AvailableSpace maybe_sub(AvailableSpace as, float v) {
    if (as.is_definite()) return AvailableSpace::definite(as.value - v);
    return as;
}
inline AvailableSpace maybe_set(AvailableSpace as, std::optional<float> v) {
    return v ? AvailableSpace::definite(*v) : as;
}
inline AvailableSpace maybe_clamp_as(AvailableSpace as, std::optional<float> mn, std::optional<float> mx) {
    if (!as.is_definite()) return as;
    float val = as.value;
    if (mn) val = std::max(val, *mn);
    if (mx) val = std::min(val, *mx);
    return AvailableSpace::definite(val);
}

// ===========================================================================
// Section 8: Grid Types
// Ref: src/style/grid.rs, src/compute/grid/types/coordinates.rs
// ===========================================================================

#ifdef TAFFY_GRID

/// Grid line coordinate (1-based CSS grid lines, 0 = invalid)
struct GridLine {
    int16_t value{0};
    constexpr GridLine() = default;
    constexpr explicit GridLine(int16_t v) : value(v) {}
    constexpr bool operator==(const GridLine& o) const { return value == o.value; }
    constexpr bool operator!=(const GridLine& o) const { return value != o.value; }
};

/// Origin-zero line coordinate (0-based normalized)
struct OriginZeroLine {
    int16_t value{0};
    constexpr OriginZeroLine() = default;
    constexpr explicit OriginZeroLine(int16_t v) : value(v) {}
    constexpr bool operator==(const OriginZeroLine& o) const { return value == o.value; }
    constexpr bool operator!=(const OriginZeroLine& o) const { return value != o.value; }
};

enum class RepetitionCount : uint8_t { AutoFill, AutoFit, Count };
struct RepetitionCountValue {
    RepetitionCount kind{RepetitionCount::AutoFill};
    uint16_t count_val{0};
    constexpr RepetitionCountValue() = default;
    constexpr RepetitionCountValue(RepetitionCount k, uint16_t c = 0) : kind(k), count_val(c) {}
    static constexpr RepetitionCountValue auto_fill() { return {RepetitionCount::AutoFill, 0}; }
    static constexpr RepetitionCountValue auto_fit() { return {RepetitionCount::AutoFit, 0}; }
    static constexpr RepetitionCountValue make_count(uint16_t n) { return {RepetitionCount::Count, n}; }
    constexpr bool operator==(const RepetitionCountValue& o) const { return kind == o.kind && count_val == o.count_val; }
    constexpr bool operator!=(const RepetitionCountValue& o) const { return !(*this == o); }
};

/// GridPlacement: Auto, Line, NamedLine, Span, NamedSpan
struct GridPlacement {
    enum Kind : uint8_t { Auto, LineKind, NamedLineKind, SpanKind, NamedSpanKind };
    Kind kind{Auto};
    GridLine line_val{};
    int16_t named_line_index{0};
    uint16_t span_val{1};
    std::string name{};

    constexpr GridPlacement() = default;
    static GridPlacement auto_() { return {}; }
    static GridPlacement line(GridLine l) { GridPlacement gp; gp.kind = LineKind; gp.line_val = l; return gp; }
    static GridPlacement named_line(std::string n, int16_t idx) {
        GridPlacement gp; gp.kind = NamedLineKind; gp.name = std::move(n); gp.named_line_index = idx; return gp;
    }
    static GridPlacement span(uint16_t s) { GridPlacement gp; gp.kind = SpanKind; gp.span_val = s; return gp; }
    static GridPlacement named_span(std::string n, uint16_t s) {
        GridPlacement gp; gp.kind = NamedSpanKind; gp.name = std::move(n); gp.span_val = s; return gp;
    }

    bool is_definite() const { return kind == LineKind || kind == NamedLineKind; }
    bool is_auto() const { return kind == Auto; }
    bool is_span() const { return kind == SpanKind || kind == NamedSpanKind; }

    bool operator==(const GridPlacement& o) const {
        if (kind != o.kind) return false;
        switch (kind) {
            case Auto: return true;
            case LineKind: return line_val == o.line_val;
            case NamedLineKind: return name == o.name && named_line_index == o.named_line_index;
            case SpanKind: return span_val == o.span_val;
            case NamedSpanKind: return name == o.name && span_val == o.span_val;
        }
        return false;
    }
    bool operator!=(const GridPlacement& o) const { return !(*this == o); }
};

/// MinTrackSizingFunction (wraps CompactLength, supports: length, percent, auto, min-content, max-content)
using MinTrackSizingFunction = CompactLength;

/// MaxTrackSizingFunction (wraps CompactLength, supports all + fr, fit-content)
using MaxTrackSizingFunction = CompactLength;

/// TrackSizingFunction = MinMax<Min, Max>
using TrackSizingFunction = MinMax<MinTrackSizingFunction, MaxTrackSizingFunction>;

/// GridTemplateRepetition
struct GridTemplateRepetition {
    RepetitionCountValue count{};
    std::vector<TrackSizingFunction> tracks{};
    std::vector<std::vector<std::string>> line_names{};
    bool operator==(const GridTemplateRepetition& o) const {
        return count == o.count && tracks == o.tracks && line_names == o.line_names;
    }
    bool operator!=(const GridTemplateRepetition& o) const { return !(*this == o); }
};

/// GridTemplateComponent: Single or Repeat
struct GridTemplateComponent {
    enum Kind : uint8_t { Single, Repeat };
    Kind kind{Single};
    TrackSizingFunction single_track{};
    GridTemplateRepetition repeat_val{};

    GridTemplateComponent() = default;
    static GridTemplateComponent single(TrackSizingFunction tsf) {
        GridTemplateComponent gtc; gtc.kind = Single; gtc.single_track = tsf; return gtc;
    }
    static GridTemplateComponent repeat(GridTemplateRepetition rep) {
        GridTemplateComponent gtc; gtc.kind = Repeat; gtc.repeat_val = std::move(rep); return gtc;
    }
    bool is_auto_repetition() const {
        if (kind != Repeat) return false;
        return repeat_val.count.kind == RepetitionCount::AutoFill || repeat_val.count.kind == RepetitionCount::AutoFit;
    }
    bool operator==(const GridTemplateComponent& o) const {
        if (kind != o.kind) return false;
        return kind == Single ? single_track == o.single_track : repeat_val == o.repeat_val;
    }
    bool operator!=(const GridTemplateComponent& o) const { return !(*this == o); }
};

/// Grid template area definition
struct GridTemplateArea {
    std::string name{};
    uint16_t row_start{0};
    uint16_t row_end{0};
    uint16_t column_start{0};
    uint16_t column_end{0};
    bool operator==(const GridTemplateArea& o) const {
        return name == o.name && row_start == o.row_start && row_end == o.row_end
            && column_start == o.column_start && column_end == o.column_end;
    }
    bool operator!=(const GridTemplateArea& o) const { return !(*this == o); }
};

#endif // TAFFY_GRID

// --- Grid internal types (used by the grid algorithm) ---
#ifdef TAFFY_GRID

// NodeId: defined here since grid types need it as a complete type
struct NodeId {
    uint32_t index{UINT32_MAX};
    uint32_t generation{0};
    constexpr NodeId() = default;
    constexpr NodeId(uint32_t idx, uint32_t gen) : index(idx), generation(gen) {}
    constexpr bool operator==(const NodeId& o) const { return index == o.index && generation == o.generation; }
    constexpr bool operator!=(const NodeId& o) const { return !(*this == o); }
    constexpr bool is_valid() const { return index != UINT32_MAX; }
};
inline constexpr NodeId NODE_ID_NONE{UINT32_MAX, 0};

namespace grid_detail {

enum class GridTrackKind : uint8_t { Track, Gutter };

struct GridTrack {
    GridTrackKind kind{GridTrackKind::Track};
    bool is_collapsed{false};
    CompactLength min_track_sizing_function{CompactLength::length(0)};
    CompactLength max_track_sizing_function{CompactLength::length(0)};
    float offset{0.0f};
    float base_size{0.0f};
    float growth_limit{0.0f};
    float content_alignment_adjustment{0.0f};
    float item_incurred_increase{0.0f};
    float base_size_planned_increase{0.0f};
    float growth_limit_planned_increase{0.0f};
    bool infinitely_growable{false};

    static GridTrack new_track(CompactLength min_fn, CompactLength max_fn) {
        GridTrack t;
        t.kind = GridTrackKind::Track;
        t.min_track_sizing_function = min_fn;
        t.max_track_sizing_function = max_fn;
        return t;
    }
    static GridTrack gutter(LengthPercentage size) {
        GridTrack t;
        t.kind = GridTrackKind::Gutter;
        t.min_track_sizing_function = size.inner;
        t.max_track_sizing_function = size.inner;
        return t;
    }
    void collapse() {
        is_collapsed = true;
        min_track_sizing_function = CompactLength::length(0);
        max_track_sizing_function = CompactLength::length(0);
    }
    bool is_flexible() const { return max_track_sizing_function.is_fr(); }
    bool uses_percentage() const { return min_track_sizing_function.uses_percentage() || max_track_sizing_function.uses_percentage(); }
    bool has_intrinsic_sizing_function() const { return min_track_sizing_function.is_intrinsic() || max_track_sizing_function.is_intrinsic(); }
    float fit_content_limit(std::optional<float> axis_available) const {
        if (max_track_sizing_function.tag() == CompactLength::FIT_CONTENT_PX_TAG) return max_track_sizing_function.value();
        if (max_track_sizing_function.tag() == CompactLength::FIT_CONTENT_PERCENT_TAG)
            return axis_available ? *axis_available * max_track_sizing_function.value() : std::numeric_limits<float>::infinity();
        return std::numeric_limits<float>::infinity();
    }
    float fit_content_limited_growth_limit(std::optional<float> axis_available) const {
        return std::min(growth_limit, fit_content_limit(axis_available));
    }
    float flex_factor() const { return max_track_sizing_function.is_fr() ? max_track_sizing_function.value() : 0.0f; }
};

struct TrackCounts {
    uint16_t negative_implicit{0};
    uint16_t explicit_{0};
    uint16_t positive_implicit{0};

    static TrackCounts from_raw(uint16_t neg, uint16_t exp, uint16_t pos) { return {neg, exp, pos}; }
    size_t len() const { return negative_implicit + explicit_ + positive_implicit; }
    OriginZeroLine implicit_start_line() const { return OriginZeroLine{static_cast<int16_t>(-negative_implicit)}; }
    OriginZeroLine implicit_end_line() const { return OriginZeroLine{static_cast<int16_t>(explicit_ + positive_implicit)}; }

    int16_t oz_line_to_next_track(OriginZeroLine index) const { return index.value + negative_implicit; }
    std::pair<int16_t, int16_t> oz_line_range_to_track_range(Line<OriginZeroLine> input) const {
        return {oz_line_to_next_track(input.start), oz_line_to_next_track(input.end)};
    }
    OriginZeroLine track_to_prev_oz_line(uint16_t index) const { return OriginZeroLine{static_cast<int16_t>(index - negative_implicit)}; }
};

enum class CellOccupancyState : uint8_t { Unoccupied, DefinitelyPlaced, AutoPlaced };

class CellOccupancyMatrix {
public:
    std::vector<std::vector<CellOccupancyState>> inner{};
    TrackCounts columns{};
    TrackCounts rows{};

    CellOccupancyMatrix() = default;
    static CellOccupancyMatrix with_track_counts(TrackCounts cols, TrackCounts rws) {
        CellOccupancyMatrix m;
        m.columns = cols;
        m.rows = rws;
        m.inner.resize(rws.len(), std::vector<CellOccupancyState>(cols.len(), CellOccupancyState::Unoccupied));
        return m;
    }

    void expand_to_fit(int16_t row_start, int16_t row_end, int16_t col_start, int16_t col_end) {
        int16_t req_neg_rows = std::max<int16_t>(-row_start, 0);
        int16_t req_pos_rows = std::max<int16_t>(row_end - (int16_t)rows.len(), 0);
        int16_t req_neg_cols = std::max<int16_t>(-col_start, 0);
        int16_t req_pos_cols = std::max<int16_t>(col_end - (int16_t)columns.len(), 0);
        if (req_neg_rows == 0 && req_pos_rows == 0 && req_neg_cols == 0 && req_pos_cols == 0) return;

        size_t old_rows = rows.len();
        size_t old_cols = columns.len();
        size_t new_rows = old_rows + req_neg_rows + req_pos_rows;
        size_t new_cols = old_cols + req_neg_cols + req_pos_cols;

        std::vector<std::vector<CellOccupancyState>> new_inner(new_rows, std::vector<CellOccupancyState>(new_cols, CellOccupancyState::Unoccupied));
        for (size_t r = 0; r < old_rows; ++r)
            for (size_t c = 0; c < old_cols; ++c)
                new_inner[r + req_neg_rows][c + req_neg_cols] = inner[r][c];

        inner = std::move(new_inner);
        rows.negative_implicit += req_neg_rows;
        rows.positive_implicit += req_pos_rows;
        columns.negative_implicit += req_neg_cols;
        columns.positive_implicit += req_pos_cols;
    }

    void mark_area_as(AbsoluteAxis primary_axis, Line<OriginZeroLine> primary_span, Line<OriginZeroLine> secondary_span, CellOccupancyState value) {
        auto [row_span, col_span] = (primary_axis == AbsoluteAxis::Horizontal)
            ? std::make_pair(secondary_span, primary_span) : std::make_pair(primary_span, secondary_span);
        auto [col_start, col_end] = columns.oz_line_range_to_track_range(col_span);
        auto [row_start, row_end] = rows.oz_line_range_to_track_range(row_span);
        expand_to_fit(row_start, row_end, col_start, col_end);
        // Re-resolve after potential expansion
        std::tie(col_start, col_end) = columns.oz_line_range_to_track_range(col_span);
        std::tie(row_start, row_end) = rows.oz_line_range_to_track_range(row_span);
        for (int16_t r = row_start; r < row_end; ++r)
            for (int16_t c = col_start; c < col_end; ++c)
                if (r >= 0 && r < (int16_t)inner.size() && c >= 0 && c < (int16_t)inner[0].size())
                    inner[r][c] = value;
    }

    bool line_area_is_unoccupied(AbsoluteAxis primary_axis, Line<OriginZeroLine> primary_span, Line<OriginZeroLine> secondary_span) {
        auto& pc = track_counts(primary_axis);
        auto& sc = track_counts(primary_axis == AbsoluteAxis::Horizontal ? AbsoluteAxis::Vertical : AbsoluteAxis::Horizontal);
        auto [p_start, p_end] = pc.oz_line_range_to_track_range(primary_span);
        auto [s_start, s_end] = sc.oz_line_range_to_track_range(secondary_span);
        auto [row_range, col_range] = (primary_axis == AbsoluteAxis::Horizontal)
            ? std::make_pair(std::make_pair(s_start, s_end), std::make_pair(p_start, p_end))
            : std::make_pair(std::make_pair(p_start, p_end), std::make_pair(s_start, s_end));
        for (int16_t r = row_range.first; r < row_range.second; ++r) {
            for (int16_t c = col_range.first; c < col_range.second; ++c) {
                if (r < 0 || r >= (int16_t)inner.size() || c < 0 || c >= (int16_t)inner[0].size()) continue;
                if (inner[r][c] != CellOccupancyState::Unoccupied) return false;
            }
        }
        return true;
    }

    bool row_is_occupied(size_t row_idx) const {
        if (row_idx >= inner.size()) return false;
        for (auto& cell : inner[row_idx]) if (cell != CellOccupancyState::Unoccupied) return true;
        return false;
    }
    bool column_is_occupied(size_t col_idx) const {
        for (auto& row : inner) if (col_idx < row.size() && row[col_idx] != CellOccupancyState::Unoccupied) return true;
        return false;
    }

    const TrackCounts& track_counts(AbsoluteAxis axis) const {
        return axis == AbsoluteAxis::Horizontal ? columns : rows;
    }
    TrackCounts& track_counts(AbsoluteAxis axis) {
        return axis == AbsoluteAxis::Horizontal ? columns : rows;
    }

    std::optional<OriginZeroLine> last_of_type(AbsoluteAxis track_type, OriginZeroLine start_at, CellOccupancyState kind) {
        auto& tc = track_counts(track_type == AbsoluteAxis::Horizontal ? AbsoluteAxis::Vertical : AbsoluteAxis::Horizontal);
        int16_t idx = tc.oz_line_to_next_track(start_at);
        if (track_type == AbsoluteAxis::Horizontal) {
            if (idx < 0 || idx >= (int16_t)inner.size()) return std::nullopt;
            for (int16_t c = (int16_t)inner[idx].size() - 1; c >= 0; --c)
                if (inner[idx][c] == kind) return tc.track_to_prev_oz_line(c);
        } else {
            if (idx < 0 || idx >= (int16_t)inner[0].size()) return std::nullopt;
            for (int16_t r = (int16_t)inner.size() - 1; r >= 0; --r)
                if (inner[r][idx] == kind) return tc.track_to_prev_oz_line(r);
        }
        return std::nullopt;
    }

    std::optional<OriginZeroLine> first_of_type(AbsoluteAxis track_type, OriginZeroLine start_at, CellOccupancyState kind) {
        auto& tc = track_counts(track_type == AbsoluteAxis::Horizontal ? AbsoluteAxis::Vertical : AbsoluteAxis::Horizontal);
        int16_t idx = tc.oz_line_to_next_track(start_at);
        if (track_type == AbsoluteAxis::Horizontal) {
            if (idx < 0 || idx >= (int16_t)inner.size()) return std::nullopt;
            for (int16_t c = 0; c < (int16_t)inner[idx].size(); ++c)
                if (inner[idx][c] == kind) return tc.track_to_prev_oz_line(c);
        } else {
            if (idx < 0 || idx >= (int16_t)inner[0].size()) return std::nullopt;
            for (int16_t r = 0; r < (int16_t)inner.size(); ++r)
                if (inner[r][idx] == kind) return tc.track_to_prev_oz_line(r);
        }
        return std::nullopt;
    }
};

struct GridItem {
    NodeId node{};
    uint16_t source_order{0};
    Line<OriginZeroLine> row{};
    Line<OriginZeroLine> column{};
    bool is_compressible_replaced{false};
    Point<Overflow> overflow{Overflow::Visible, Overflow::Visible};
    BoxSizing box_sizing{BoxSizing::BorderBox};
    Size<Dimension> size{};
    Size<Dimension> min_size{};
    Size<Dimension> max_size{};
    std::optional<float> aspect_ratio{};
    Rect<LengthPercentage> padding{};
    Rect<LengthPercentage> border{};
    Rect<LengthPercentageAuto> margin{};
    AlignSelf align_self{};
    AlignSelf justify_self{};
    std::optional<float> baseline{};
    float baseline_shim{0.0f};
    Line<uint16_t> row_indexes{0, 0};
    Line<uint16_t> column_indexes{0, 0};
    bool crosses_flexible_row{false};
    bool crosses_flexible_column{false};
    bool crosses_intrinsic_row{false};
    bool crosses_intrinsic_column{false};
    std::optional<Size<std::optional<float>>> grid_area_size_cache{};
    Size<std::optional<float>> min_content_contribution_cache{std::nullopt, std::nullopt};
    Size<std::optional<float>> minimum_contribution_cache{std::nullopt, std::nullopt};
    Size<std::optional<float>> max_content_contribution_cache{std::nullopt, std::nullopt};
    float y_position{0.0f};
    float height{0.0f};
    bool placed{false};  // Set by placement algorithm

    Line<OriginZeroLine> placement(AbstractAxis axis) const {
        return axis == AbstractAxis::Block ? row : column;
    }
    Line<uint16_t> placement_indexes(AbstractAxis axis) const {
        return axis == AbstractAxis::Block ? row_indexes : column_indexes;
    }
    uint16_t span(AbstractAxis axis) const {
        auto p = placement(axis);
        return static_cast<uint16_t>(std::max(p.end.value - p.start.value, 0));
    }
    bool crosses_flexible_track(AbstractAxis axis) const {
        return axis == AbstractAxis::Block ? crosses_flexible_row : crosses_flexible_column;
    }
    bool crosses_intrinsic_track(AbstractAxis axis) const {
        return axis == AbstractAxis::Block ? crosses_intrinsic_row : crosses_intrinsic_column;
    }
};

} // namespace grid_detail

#endif // TAFFY_GRID

// ===========================================================================
// Section 9: Style struct
// Ref: src/style/mod.rs
// ===========================================================================

struct Style {
    Display display{Display::Flex};
    bool item_is_table{false};
    bool item_is_replaced{false};
    BoxSizing box_sizing{BoxSizing::BorderBox};
    Direction direction{Direction::Ltr};

    // Overflow
    Point<Overflow> overflow{Overflow::Visible, Overflow::Visible};
    float scrollbar_width{0.0f};

#ifdef TAFFY_FLOAT_LAYOUT
    Float float_{Float::None};
    Clear clear{Clear::None};
#endif

    // Position
    Position position{Position::Relative};
    Rect<LengthPercentageAuto> inset{
        LengthPercentageAuto::auto_(), LengthPercentageAuto::auto_(),
        LengthPercentageAuto::auto_(), LengthPercentageAuto::auto_()};

    // Size
    Size<Dimension> size{Dimension::auto_(), Dimension::auto_()};
    Size<Dimension> min_size{Dimension::auto_(), Dimension::auto_()};
    Size<Dimension> max_size{Dimension::auto_(), Dimension::auto_()};
    std::optional<float> aspect_ratio{std::nullopt};

    // Spacing
    Rect<LengthPercentageAuto> margin{
        LengthPercentageAuto(), LengthPercentageAuto(), LengthPercentageAuto(), LengthPercentageAuto()};
    Rect<LengthPercentage> padding{LengthPercentage(), LengthPercentage(), LengthPercentage(), LengthPercentage()};
    Rect<LengthPercentage> border{LengthPercentage(), LengthPercentage(), LengthPercentage(), LengthPercentage()};

    // Alignment
    std::optional<AlignItems> align_items{std::nullopt};
    std::optional<AlignSelf> align_self{std::nullopt};
#ifdef TAFFY_GRID
    std::optional<AlignItems> justify_items{std::nullopt};
    std::optional<JustifySelf> justify_self{std::nullopt};
#endif
    std::optional<AlignContent> align_content{std::nullopt};
    std::optional<JustifyContent> justify_content{std::nullopt};
    Size<LengthPercentage> gap{LengthPercentage(), LengthPercentage()};

#ifdef TAFFY_BLOCK_LAYOUT
    TextAlign text_align{TextAlign::Auto};
#endif

    // Flexbox container
#ifdef TAFFY_FLEXBOX
    FlexDirection flex_direction{FlexDirection::Row};
    FlexWrap flex_wrap{FlexWrap::NoWrap};
    Dimension flex_basis{Dimension::auto_()};
    float flex_grow{0.0f};
    float flex_shrink{1.0f};
#endif

    // Grid container
#ifdef TAFFY_GRID
    std::vector<GridTemplateComponent> grid_template_rows{};
    std::vector<GridTemplateComponent> grid_template_columns{};
    std::vector<TrackSizingFunction> grid_auto_rows{};
    std::vector<TrackSizingFunction> grid_auto_columns{};
    GridAutoFlow grid_auto_flow{GridAutoFlow::Row};
    std::vector<GridTemplateArea> grid_template_areas{};
    std::vector<std::vector<std::string>> grid_template_column_names{};
    std::vector<std::vector<std::string>> grid_template_row_names{};
    Line<GridPlacement> grid_row{GridPlacement::auto_(), GridPlacement::auto_()};
    Line<GridPlacement> grid_column{GridPlacement::auto_(), GridPlacement::auto_()};
#endif

    bool operator==(const Style& o) const {
        return display == o.display && item_is_table == o.item_is_table && item_is_replaced == o.item_is_replaced
            && box_sizing == o.box_sizing && direction == o.direction && overflow == o.overflow
            && scrollbar_width == o.scrollbar_width
#ifdef TAFFY_FLOAT_LAYOUT
            && float_ == o.float_ && clear == o.clear
#endif
            && position == o.position && inset == o.inset && size == o.size && min_size == o.min_size
            && max_size == o.max_size && aspect_ratio == o.aspect_ratio && margin == o.margin
            && padding == o.padding && border == o.border && align_items == o.align_items && align_self == o.align_self
            && align_content == o.align_content && justify_content == o.justify_content && gap == o.gap
#ifdef TAFFY_GRID
            && justify_items == o.justify_items && justify_self == o.justify_self
            && grid_template_rows == o.grid_template_rows && grid_template_columns == o.grid_template_columns
            && grid_auto_rows == o.grid_auto_rows && grid_auto_columns == o.grid_auto_columns
            && grid_auto_flow == o.grid_auto_flow && grid_template_areas == o.grid_template_areas
            && grid_template_column_names == o.grid_template_column_names
            && grid_template_row_names == o.grid_template_row_names
            && grid_row == o.grid_row && grid_column == o.grid_column
#endif
#ifdef TAFFY_BLOCK_LAYOUT
            && text_align == o.text_align
#endif
#ifdef TAFFY_FLEXBOX
            && flex_direction == o.flex_direction && flex_wrap == o.flex_wrap && flex_basis == o.flex_basis
            && flex_grow == o.flex_grow && flex_shrink == o.flex_shrink
#endif
            ;
    }
    bool operator!=(const Style& o) const { return !(*this == o); }
};

// ===========================================================================
// Section 10: Style Helpers
// Ref: src/style_helpers.rs
// ===========================================================================

// Factory functions for creating style values
inline constexpr LengthPercentage zero_lp() { return LengthPercentage::length(0.0f); }
inline constexpr LengthPercentageAuto zero_lpa() { return LengthPercentageAuto::length(0.0f); }
inline constexpr Dimension zero_dim() { return Dimension::length(0.0f); }
inline constexpr LengthPercentage length(float v) { return LengthPercentage::length(v); }
inline constexpr LengthPercentage percent(float v) { return LengthPercentage::percent(v); }
inline constexpr LengthPercentageAuto auto_lpa() { return LengthPercentageAuto::auto_(); }
inline constexpr Dimension auto_dim() { return Dimension::auto_(); }
inline constexpr Dimension dim_length(float v) { return Dimension::length(v); }
inline constexpr Dimension dim_percent(float v) { return Dimension::percent(v); }
inline constexpr Dimension fr(float v) { return Dimension::fr(v); }
inline constexpr Dimension min_content() { return Dimension::min_content(); }
inline constexpr Dimension max_content() { return Dimension::max_content(); }
inline constexpr Dimension fit_content(LengthPercentage lp) {
    return lp.tag() == CompactLength::LENGTH_TAG
        ? Dimension::fit_content_px(lp.value())
        : Dimension::fit_content_percent(lp.value());
}

#ifdef TAFFY_GRID
inline TrackSizingFunction minmax(MinTrackSizingFunction mn, MaxTrackSizingFunction mx) {
    return MinMax<MinTrackSizingFunction, MaxTrackSizingFunction>{mn, mx};
}
inline TrackSizingFunction flex_track(float v) {
    return minmax(CompactLength::length(0.0f), CompactLength::fr(v));
}
inline GridTemplateComponent repeat_grid(RepetitionCountValue count, std::vector<TrackSizingFunction> tracks,
                                          std::vector<std::vector<std::string>> line_names = {}) {
    GridTemplateRepetition rep;
    rep.count = std::move(count);
    rep.tracks = std::move(tracks);
    rep.line_names = std::move(line_names);
    return GridTemplateComponent::repeat(std::move(rep));
}
inline std::vector<TrackSizingFunction> evenly_sized_tracks(size_t n) {
    return std::vector<TrackSizingFunction>(n, flex_track(1.0f));
}
#endif

// ===========================================================================
// Section 11: Layout Types
// Ref: src/tree/layout.rs
// ===========================================================================

enum class RunMode : uint8_t { PerformLayout, ComputeSize, PerformHiddenLayout };
enum class SizingMode : uint8_t { ContentSize, InherentSize };
enum class RequestedAxis : uint8_t { Horizontal, Vertical, Both };

/// Collapsible margin set for block layout margin collapsing
struct CollapsibleMarginSet {
    float positive{0.0f};
    float negative{0.0f};

    static constexpr CollapsibleMarginSet zero() { return {0.0f, 0.0f}; }
    static constexpr CollapsibleMarginSet from_margin(float margin) {
        return margin >= 0.0f ? CollapsibleMarginSet{margin, 0.0f} : CollapsibleMarginSet{0.0f, margin};
    }
    CollapsibleMarginSet collapse_with_margin(float margin) const {
        if (margin >= 0.0f) return {std::max(positive, margin), negative};
        return {positive, std::min(negative, margin)};
    }
    CollapsibleMarginSet collapse_with_set(CollapsibleMarginSet other) const {
        return {std::max(positive, other.positive), std::min(negative, other.negative)};
    }
    float resolve() const { return positive + negative; }
    constexpr bool operator==(const CollapsibleMarginSet& o) const { return positive == o.positive && negative == o.negative; }
    constexpr bool operator!=(const CollapsibleMarginSet& o) const { return !(*this == o); }
};

/// Input to a layout computation
struct LayoutInput {
    RunMode run_mode{RunMode::PerformLayout};
    SizingMode sizing_mode{SizingMode::InherentSize};
    RequestedAxis axis{RequestedAxis::Both};
    Size<std::optional<float>> known_dimensions{};
    Size<std::optional<float>> parent_size{};
    Size<AvailableSpace> available_space{};
    Line<bool> vertical_margins_are_collapsible{false, false};
};

/// Output of a layout computation
struct LayoutOutput {
    Size<float> size{0.0f, 0.0f};
#ifdef TAFFY_CONTENT_SIZE
    Size<float> content_size{0.0f, 0.0f};
#endif
    Point<std::optional<float>> first_baselines{std::nullopt, std::nullopt};
    CollapsibleMarginSet top_margin{CollapsibleMarginSet::zero()};
    CollapsibleMarginSet bottom_margin{CollapsibleMarginSet::zero()};
    bool margins_can_collapse_through{false};

    static LayoutOutput hidden() { return {}; }

    static LayoutOutput from_sizes_and_baselines(Size<float> sz, Size<float> csz, Point<std::optional<float>> baselines) {
        LayoutOutput out;
        out.size = sz;
#ifdef TAFFY_CONTENT_SIZE
        out.content_size = csz;
#endif
        out.first_baselines = baselines;
        return out;
    }
    static LayoutOutput from_sizes(Size<float> sz, Size<float> csz) {
        return from_sizes_and_baselines(sz, csz, Point<std::optional<float>>{std::nullopt, std::nullopt});
    }
    static LayoutOutput from_outer_size(Size<float> sz) {
        return from_sizes(sz, Size<float>{0.0f, 0.0f});
    }
};

/// Final layout result stored on tree nodes
struct Layout {
    uint32_t order{0};
    Point<float> location{0.0f, 0.0f};
    Size<float> size{0.0f, 0.0f};
#ifdef TAFFY_CONTENT_SIZE
    Size<float> content_size{0.0f, 0.0f};
#endif
    Size<float> scrollbar_size{0.0f, 0.0f};
    Rect<float> border{0.0f, 0.0f, 0.0f, 0.0f};
    Rect<float> padding{0.0f, 0.0f, 0.0f, 0.0f};
    Rect<float> margin{0.0f, 0.0f, 0.0f, 0.0f};

    static constexpr Layout new_layout() { return {}; }
    static constexpr Layout with_order(uint32_t ord) { Layout l; l.order = ord; return l; }

    float content_box_width() const { return size.width - padding.left - padding.right - border.left - border.right; }
    float content_box_height() const { return size.height - padding.top - padding.bottom - border.top - border.bottom; }
    Size<float> content_box_size() const { return {content_box_width(), content_box_height()}; }
    float content_box_x() const { return location.x + border.left + padding.left; }
    float content_box_y() const { return location.y + border.top + padding.top; }
#ifdef TAFFY_CONTENT_SIZE
    float scroll_width() const { return content_size.width - content_box_width(); }
    float scroll_height() const { return content_size.height - content_box_height(); }
#endif
};

// ===========================================================================
// Section 12: Cache System
// Ref: src/tree/cache.rs
// ===========================================================================

namespace detail {
    static constexpr size_t CACHE_SIZE = 9;

    template <typename T>
    struct CacheEntry {
        Size<std::optional<float>> known_dimensions{};
        Size<AvailableSpace> available_space{};
        T content{};
    };

    inline size_t compute_cache_slot(Size<std::optional<float>> known_dimensions, Size<AvailableSpace> available_space) {
        if (known_dimensions.width.has_value() && known_dimensions.height.has_value()) return 0;
        if (known_dimensions.width.has_value()) {
            return (available_space.height.is_definite() || available_space.height == AvailableSpace::max_content()) ? 1 : 2;
        }
        if (known_dimensions.height.has_value()) {
            return (available_space.width.is_definite() || available_space.width == AvailableSpace::max_content()) ? 3 : 4;
        }
        bool wx = available_space.width.is_definite() || available_space.width == AvailableSpace::max_content();
        bool wy = available_space.height.is_definite() || available_space.height == AvailableSpace::max_content();
        if (wx && wy) return 5;
        if (wx) return 6;
        if (wy) return 7;
        return 8;
    }
} // namespace detail

enum class ClearState : uint8_t { Cleared, AlreadyEmpty };

struct Cache {
    std::optional<detail::CacheEntry<LayoutOutput>> final_layout_entry{};
    std::optional<detail::CacheEntry<Size<float>>> measure_entries[detail::CACHE_SIZE]{};
    bool is_empty_{true};

    Cache() = default;

    bool is_empty() const { return is_empty_; }

    std::optional<LayoutOutput> get(const LayoutInput& input) const {
        if (input.run_mode == RunMode::PerformHiddenLayout) return std::nullopt;
        if (input.run_mode == RunMode::PerformLayout) {
            if (final_layout_entry.has_value()) {
                auto& e = *final_layout_entry;
                if (e.known_dimensions == input.known_dimensions && e.available_space.width.is_roughly_equal(input.available_space.width)
                    && e.available_space.height.is_roughly_equal(input.available_space.height)) {
                    return e.content;
                }
            }
            return std::nullopt;
        }
        // ComputeSize
        size_t slot = detail::compute_cache_slot(input.known_dimensions, input.available_space);
        for (size_t i = 0; i < detail::CACHE_SIZE; ++i) {
            size_t idx = (slot + i) % detail::CACHE_SIZE;
            if (measure_entries[idx].has_value()) {
                auto& e = *measure_entries[idx];
                bool kw_match = e.known_dimensions == input.known_dimensions
                    || (input.known_dimensions.width.has_value() && e.known_dimensions.width == input.known_dimensions.width)
                    || (input.known_dimensions.height.has_value() && e.known_dimensions.height == input.known_dimensions.height);
                bool as_match = e.available_space.width.is_roughly_equal(input.available_space.width)
                    && e.available_space.height.is_roughly_equal(input.available_space.height);
                if (kw_match && as_match) {
                    LayoutOutput out;
                    out.size = e.content;
                    return out;
                }
            }
        }
        return std::nullopt;
    }

    void store(const LayoutInput& input, const LayoutOutput& output) {
        is_empty_ = false;
        if (input.run_mode == RunMode::PerformLayout) {
            detail::CacheEntry<LayoutOutput> entry;
            entry.known_dimensions = input.known_dimensions;
            entry.available_space = input.available_space;
            entry.content = output;
            final_layout_entry = entry;
            return;
        }
        if (input.run_mode == RunMode::ComputeSize) {
            size_t slot = detail::compute_cache_slot(input.known_dimensions, input.available_space);
            detail::CacheEntry<Size<float>> entry;
            entry.known_dimensions = input.known_dimensions;
            entry.available_space = input.available_space;
            entry.content = output.size;
            measure_entries[slot] = entry;
        }
    }

    ClearState clear() {
        if (is_empty_) return ClearState::AlreadyEmpty;
        final_layout_entry = std::nullopt;
        for (size_t i = 0; i < detail::CACHE_SIZE; ++i) measure_entries[i] = std::nullopt;
        is_empty_ = true;
        return ClearState::Cleared;
    }
};

// ===========================================================================
// Section 13: CRTP Tree Traits
// Ref: src/tree/traits.rs
// ===========================================================================

// NodeId is defined earlier (before grid types)

// CRTP base for partial tree traversal
template <typename Derived>
struct TraversePartialTreeBase {
    // Derived must implement:
    //   std::vector<NodeId> child_ids(NodeId parent) const;
    //   size_t child_count(NodeId parent) const;
    //   NodeId get_child_id(NodeId parent, size_t index) const;
};

// CRTP base for layout computation
template <typename Derived>
struct LayoutPartialTreeBase : TraversePartialTreeBase<Derived> {
    // Derived must implement:
    //   const Style& get_core_container_style(NodeId node_id) const;
    //   void set_unrounded_layout(NodeId node_id, const Layout& layout);
    //   LayoutOutput compute_child_layout(NodeId node_id, LayoutInput inputs);
};

// CRTP base for cache operations
template <typename Derived>
struct CacheTreeBase {
    // Derived must implement:
    //   std::optional<LayoutOutput> cache_get(NodeId node_id, const LayoutInput& input) const;
    //   void cache_store(NodeId node_id, const LayoutInput& input, const LayoutOutput& output);
    //   void cache_clear(NodeId node_id);
};

// CRTP base for layout rounding
template <typename Derived>
struct RoundTreeBase : TraversePartialTreeBase<Derived> {
    // Derived must implement:
    //   Layout get_unrounded_layout(NodeId node_id) const;
    //   void set_final_layout(NodeId node_id, const Layout& layout);
};

// CRTP base for debug printing
template <typename Derived>
struct PrintTreeBase : TraversePartialTreeBase<Derived> {
    // Derived must implement:
    //   const char* get_debug_label(NodeId node_id) const;
    //   Layout get_final_layout(NodeId node_id) const;
};

#ifdef TAFFY_FLEXBOX
template <typename Derived>
struct LayoutFlexboxContainerBase : LayoutPartialTreeBase<Derived> {
    // Derived must implement:
    //   const Style& get_flexbox_container_style(NodeId node_id) const;
    //   const Style& get_flexbox_child_style(NodeId child_node_id) const;
};
#endif

#ifdef TAFFY_GRID
template <typename Derived>
struct LayoutGridContainerBase : LayoutPartialTreeBase<Derived> {
    // Derived must implement:
    //   const Style& get_grid_container_style(NodeId node_id) const;
    //   const Style& get_grid_child_style(NodeId child_node_id) const;
};
#endif

#ifdef TAFFY_BLOCK_LAYOUT
template <typename Derived>
struct LayoutBlockContainerBase : LayoutPartialTreeBase<Derived> {
    // Derived must implement:
    //   const Style& get_block_container_style(NodeId node_id) const;
    //   const Style& get_block_child_style(NodeId child_node_id) const;
};
#endif

// ===========================================================================
// Section 14: Utility Functions
// Ref: src/util/math.rs, src/util/resolve.rs
// ===========================================================================

namespace detail {
    inline float f32_max(float a, float b) { return std::max(a, b); }
    inline float f32_min(float a, float b) { return std::min(a, b); }
    inline float f32_round(float v) { return std::floor(v + 0.5f); }
    inline float f32_abs(float v) { return std::abs(v); }
}

// MaybeMath: operations on optional<float> combinations
// Option<f32> + Option<f32> -> Option<f32> (None propagates)
inline std::optional<float> maybe_add(std::optional<float> a, std::optional<float> b) {
    if (a && b) return *a + *b; return std::nullopt;
}
inline std::optional<float> maybe_sub(std::optional<float> a, std::optional<float> b) {
    if (a && b) return *a - *b; return std::nullopt;
}
inline std::optional<float> maybe_max(std::optional<float> a, std::optional<float> b) {
    if (a && b) return std::max(*a, *b); return std::nullopt;
}
inline std::optional<float> maybe_min(std::optional<float> a, std::optional<float> b) {
    if (a && b) return std::min(*a, *b); return std::nullopt;
}
inline std::optional<float> maybe_clamp(std::optional<float> v, std::optional<float> mn, std::optional<float> mx) {
    if (!v) return std::nullopt;
    float r = *v;
    if (mn) r = std::max(r, *mn);
    if (mx) r = std::min(r, *mx);
    return r;
}

// f32 x Option<f32> -> f32 (None treated as identity)
inline float maybe_add_f32(float a, std::optional<float> b) { return b ? a + *b : a; }
inline float maybe_sub_f32(float a, std::optional<float> b) { return b ? a - *b : a; }

// MaybeResolve: resolve dimension types against a parent size
inline std::optional<float> maybe_resolve(LengthPercentage lp, std::optional<float> context) {
    switch (lp.tag()) {
        case CompactLength::LENGTH_TAG: return lp.value();
        case CompactLength::PERCENT_TAG: return context ? std::optional<float>((*context * lp.value()) / 100.0f) : std::nullopt;
        default: return std::nullopt;
    }
}
inline std::optional<float> maybe_resolve(LengthPercentageAuto lpa, std::optional<float> context) {
    switch (lpa.tag()) {
        case CompactLength::LENGTH_TAG: return lpa.value();
        case CompactLength::PERCENT_TAG: return context ? std::optional<float>((*context * lpa.value()) / 100.0f) : std::nullopt;
        case CompactLength::AUTO_TAG: return std::nullopt;
        default: return std::nullopt;
    }
}
inline std::optional<float> maybe_resolve(Dimension d, std::optional<float> context) {
    switch (d.tag()) {
        case CompactLength::LENGTH_TAG: return d.value();
        case CompactLength::PERCENT_TAG: return context ? std::optional<float>((*context * d.value()) / 100.0f) : std::nullopt;
        case CompactLength::AUTO_TAG: return std::nullopt;
        default: return std::nullopt;
    }
}
// ResolveOrZero: like maybe_resolve but returns 0.0 instead of nullopt
inline float resolve_or_zero(LengthPercentage lp, std::optional<float> context) {
    return maybe_resolve(lp, context).value_or(0.0f);
}
inline float resolve_or_zero(LengthPercentageAuto lpa, std::optional<float> context) {
    return maybe_resolve(lpa, context).value_or(0.0f);
}
inline float resolve_or_zero(Dimension d, std::optional<float> context) {
    return maybe_resolve(d, context).value_or(0.0f);
}
// Resolve Size/Rect versions
inline Size<std::optional<float>> maybe_resolve_size(Size<Dimension> s, Size<std::optional<float>> context) {
    return {maybe_resolve(s.width, context.width), maybe_resolve(s.height, context.height)};
}
inline Size<std::optional<float>> resolve_or_zero_size(Size<Dimension> s, Size<std::optional<float>> context) {
    return {std::optional(resolve_or_zero(s.width, context.width)), std::optional(resolve_or_zero(s.height, context.height))};
}
inline Rect<float> resolve_or_zero_rect(Rect<LengthPercentage> r, std::optional<float> context) {
    return {resolve_or_zero(r.left, context), resolve_or_zero(r.right, context),
            resolve_or_zero(r.top, context), resolve_or_zero(r.bottom, context)};
}
inline Rect<float> resolve_or_zero_rect(Rect<LengthPercentageAuto> r, std::optional<float> context) {
    return {resolve_or_zero(r.left, context), resolve_or_zero(r.right, context),
            resolve_or_zero(r.top, context), resolve_or_zero(r.bottom, context)};
}
inline Rect<std::optional<float>> maybe_resolve_rect(Rect<LengthPercentageAuto> r, std::optional<float> context) {
    return {maybe_resolve(r.left, context), maybe_resolve(r.right, context),
            maybe_resolve(r.top, context), maybe_resolve(r.bottom, context)};
}
// Overloads for float context
inline std::optional<float> maybe_resolve(LengthPercentage lp, float context) {
    return maybe_resolve(lp, std::optional<float>(context));
}

// ===========================================================================
// Section 14b: Flexbox Helpers and Alignment Utilities
// Ref: src/compute/flexbox.rs, src/compute/common/alignment.rs
// ===========================================================================

#ifdef TAFFY_FLEXBOX

// --- Size<optional<float>> free functions for MaybeMath ---

inline Size<std::optional<float>> optf32_size_none() {
    return {std::nullopt, std::nullopt};
}

inline Size<std::optional<float>> optf32_size_or(
    const Size<std::optional<float>>& s,
    const Size<std::optional<float>>& fallback) {
    return {s.width ? s.width : fallback.width, s.height ? s.height : fallback.height};
}

inline Size<std::optional<float>> optf32_size_or_f32(
    const Size<std::optional<float>>& s,
    const Size<float>& fallback) {
    return {s.width ? s.width : std::optional<float>(fallback.width),
            s.height ? s.height : std::optional<float>(fallback.height)};
}

inline Size<std::optional<float>> optf32_size_add(
    const Size<std::optional<float>>& a,
    const Size<std::optional<float>>& b) {
    return {maybe_add(a.width, b.width), maybe_add(a.height, b.height)};
}

inline Size<std::optional<float>> optf32_size_sub(
    const Size<std::optional<float>>& a,
    const Size<std::optional<float>>& b) {
    return {maybe_sub(a.width, b.width), maybe_sub(a.height, b.height)};
}

inline Size<std::optional<float>> optf32_size_clamp(
    const Size<std::optional<float>>& v,
    const Size<std::optional<float>>& mn,
    const Size<std::optional<float>>& mx) {
    return {maybe_clamp(v.width, mn.width, mx.width), maybe_clamp(v.height, mn.height, mx.height)};
}

inline Size<std::optional<float>> optf32_size_max(
    const Size<std::optional<float>>& a,
    const Size<std::optional<float>>& b) {
    return {maybe_max(a.width, b.width), maybe_max(a.height, b.height)};
}

inline Size<std::optional<float>> optf32_size_min(
    const Size<std::optional<float>>& a,
    const Size<std::optional<float>>& b) {
    return {maybe_min(a.width, b.width), maybe_min(a.height, b.height)};
}

// --- AvailableSpace MaybeMath helpers ---

inline AvailableSpace avail_add(AvailableSpace a, std::optional<float> v) {
    return v ? AvailableSpace::definite(a.unwrap_or(0.0f) + *v) : a;
}

inline AvailableSpace avail_sub(AvailableSpace a, std::optional<float> v) {
    return v ? AvailableSpace::definite(a.unwrap_or(0.0f) - *v) : a;
}

inline AvailableSpace avail_sub_f32(AvailableSpace a, float v) {
    return a.is_definite() ? AvailableSpace::definite(a.value - v) : a;
}

inline AvailableSpace avail_max(AvailableSpace a, std::optional<float> v) {
    return v ? AvailableSpace::definite(std::max(a.unwrap_or(0.0f), *v)) : a;
}

inline AvailableSpace avail_clamp(AvailableSpace a, std::optional<float> mn, std::optional<float> mx) {
    if (a.type != AvailableSpace::Definite) return a;
    float val = a.value;
    if (mn) val = std::max(val, *mn);
    if (mx) val = std::min(val, *mx);
    return AvailableSpace::definite(val);
}

// --- Size<AvailableSpace> helpers ---

inline Size<std::optional<float>> avail_into_options(const Size<AvailableSpace>& s) {
    return {s.width.into_option(), s.height.into_option()};
}

inline AvailableSpace opt_to_avail(std::optional<float> v) {
    return v ? AvailableSpace::definite(*v) : AvailableSpace::max_content();
}

inline Size<AvailableSpace> opt_to_avail_size(const Size<std::optional<float>>& s) {
    return {opt_to_avail(s.width), opt_to_avail(s.height)};
}

// --- Free functions for main/cross on Size<optional<float>> ---

inline std::optional<float> size_opt_main(const Size<std::optional<float>>& s, FlexDirection d) {
    return is_row(d) ? s.width : s.height;
}
inline std::optional<float> size_opt_cross(const Size<std::optional<float>>& s, FlexDirection d) {
    return is_row(d) ? s.height : s.width;
}
inline Size<std::optional<float>>& size_opt_set_main(Size<std::optional<float>>& s, FlexDirection d, std::optional<float> v) {
    (is_row(d) ? s.width : s.height) = v; return s;
}
inline Size<std::optional<float>>& size_opt_set_cross(Size<std::optional<float>>& s, FlexDirection d, std::optional<float> v) {
    (is_row(d) ? s.height : s.width) = v; return s;
}
inline Size<std::optional<float>> size_opt_with_main(const Size<std::optional<float>>& s, FlexDirection d, std::optional<float> v) {
    Size<std::optional<float>> r = s; size_opt_set_main(r, d, v); return r;
}
inline Size<std::optional<float>> size_opt_with_cross(const Size<std::optional<float>>& s, FlexDirection d, std::optional<float> v) {
    Size<std::optional<float>> r = s; size_opt_set_cross(r, d, v); return r;
}

// Free functions for main/cross on Size<AvailableSpace>
inline AvailableSpace size_avail_main(const Size<AvailableSpace>& s, FlexDirection d) {
    return is_row(d) ? s.width : s.height;
}
inline AvailableSpace size_avail_cross(const Size<AvailableSpace>& s, FlexDirection d) {
    return is_row(d) ? s.height : s.width;
}
inline Size<AvailableSpace>& size_avail_set_main(Size<AvailableSpace>& s, FlexDirection d, AvailableSpace v) {
    (is_row(d) ? s.width : s.height) = v; return s;
}
inline Size<AvailableSpace> size_avail_with_main(const Size<AvailableSpace>& s, FlexDirection d, AvailableSpace v) {
    Size<AvailableSpace> r = s; size_avail_set_main(r, d, v); return r;
}
inline Size<AvailableSpace> size_avail_with_cross(const Size<AvailableSpace>& s, FlexDirection d, AvailableSpace v) {
    Size<AvailableSpace> r = s; (is_row(d) ? r.height : r.width) = v; return r;
}

// --- Point<optional<float>> free function ---
inline std::optional<float> point_opt_main(const Point<std::optional<float>>& p, FlexDirection d) {
    return is_row(d) ? p.x : p.y;
}

// --- Flex direction helper ---
inline AbsoluteAxis flex_main_axis(FlexDirection d) {
    return is_row(d) ? AbsoluteAxis::Horizontal : AbsoluteAxis::Vertical;
}

// --- Alignment resolution functions ---
inline AlignItemsKeyword resolve_self_alignment_safety(AlignItems alignment, bool overflows) {
    if (alignment.is_safe() && overflows) return AlignItemsKeyword::Start;
    return alignment.keyword;
}

inline AlignContentKeyword apply_alignment_fallback(float free_space, size_t num_items, AlignContent alignment_mode) {
    auto keyword = alignment_mode.keyword;
    bool is_safe = alignment_mode.is_safe();

    if (num_items <= 1 || free_space <= 0.0f) {
        switch (keyword) {
            case AlignContentKeyword::Stretch:
            case AlignContentKeyword::SpaceBetween:
                keyword = AlignContentKeyword::FlexStart; is_safe = true; break;
            case AlignContentKeyword::SpaceAround:
            case AlignContentKeyword::SpaceEvenly:
                keyword = AlignContentKeyword::Center; is_safe = true; break;
            default: break;
        }
    }

    if (free_space <= 0.0f && is_safe) keyword = AlignContentKeyword::Start;
    return keyword;
}

inline JustifyContent apply_alignment_fallback_jc(float free_space, size_t num_items, JustifyContent alignment_mode) {
    AlignContent ac{alignment_mode.keyword, alignment_mode.safety};
    return JustifyContent{apply_alignment_fallback(free_space, num_items, ac), alignment_mode.safety};
}

inline float compute_alignment_offset(
    float free_space, size_t num_items, float gap,
    AlignContentKeyword alignment_mode, bool layout_is_flex_reversed, bool is_first)
{
    if (is_first) {
        switch (alignment_mode) {
            case AlignContentKeyword::Start: return 0.0f;
            case AlignContentKeyword::FlexStart: return layout_is_flex_reversed ? free_space : 0.0f;
            case AlignContentKeyword::End: return free_space;
            case AlignContentKeyword::FlexEnd: return layout_is_flex_reversed ? 0.0f : free_space;
            case AlignContentKeyword::Center: return free_space / 2.0f;
            case AlignContentKeyword::Stretch: return 0.0f;
            case AlignContentKeyword::SpaceBetween: return 0.0f;
            case AlignContentKeyword::SpaceAround:
                return free_space >= 0.0f ? (free_space / static_cast<float>(num_items)) / 2.0f : free_space / 2.0f;
            case AlignContentKeyword::SpaceEvenly:
                return free_space >= 0.0f ? free_space / static_cast<float>(num_items + 1) : free_space / 2.0f;
        }
        return 0.0f;
    }
    float fs = std::max(free_space, 0.0f);
    float spacing = 0.0f;
    switch (alignment_mode) {
        case AlignContentKeyword::Start: case AlignContentKeyword::FlexStart:
        case AlignContentKeyword::End: case AlignContentKeyword::FlexEnd:
        case AlignContentKeyword::Center: case AlignContentKeyword::Stretch:
            spacing = 0.0f; break;
        case AlignContentKeyword::SpaceBetween:
            spacing = fs / static_cast<float>(num_items - 1); break;
        case AlignContentKeyword::SpaceAround:
            spacing = fs / static_cast<float>(num_items); break;
        case AlignContentKeyword::SpaceEvenly:
            spacing = fs / static_cast<float>(num_items + 1); break;
    }
    return gap + spacing;
}

#endif // TAFFY_FLEXBOX

// ===========================================================================
// Section 15: Flexbox Algorithm
// Ref: src/compute/flexbox.rs, src/compute/common/alignment.rs
// ===========================================================================

#ifdef TAFFY_FLEXBOX

// --- Alignment utilities (shared with Grid) ---

inline float sum_axis_gaps(float gap, size_t num_items) {
    return num_items <= 1 ? 0.0f : gap * (num_items - 1);
}

// --- Flexbox internal structs ---

struct FlexItem {
    NodeId node{};
    uint32_t order{0};
    Size<std::optional<float>> size{};
    Size<std::optional<float>> min_size{};
    Size<std::optional<float>> max_size{};
    AlignSelf align_self{};
    Point<Overflow> overflow{Overflow::Visible, Overflow::Visible};
    float scrollbar_width{0.0f};
    float flex_shrink{1.0f};
    float flex_grow{0.0f};
    float resolved_minimum_main_size{0.0f};
    Rect<std::optional<float>> inset{};
    Rect<float> margin{0, 0, 0, 0};
    Rect<bool> margin_is_auto{false, false, false, false};
    Rect<float> padding{0, 0, 0, 0};
    Rect<float> border{0, 0, 0, 0};
    float flex_basis{0.0f};
    float inner_flex_basis{0.0f};
    float violation{0.0f};
    bool frozen{false};
    float content_flex_fraction{0.0f};
    Size<float> hypothetical_inner_size{};
    Size<float> hypothetical_outer_size{};
    Size<float> target_size{};
    Size<float> outer_target_size{};
    float baseline{0.0f};
    float offset_main{0.0f};
    float offset_cross{0.0f};

    bool is_scroll_container() const {
        return ::taffy::is_scroll_container(overflow.x) || ::taffy::is_scroll_container(overflow.y);
    }
};

struct FlexLine {
    std::vector<FlexItem> items{};
    float cross_size{0.0f};
    float offset_cross{0.0f};
};

struct FlexAlgoConstants {
    FlexDirection dir{FlexDirection::Row};
    Direction layout_direction{Direction::Ltr};
    bool is_row{true};
    bool is_column{false};
    bool is_wrap{false};
    bool is_wrap_reverse{false};
    Size<std::optional<float>> min_size{};
    Size<std::optional<float>> max_size{};
    Rect<float> margin{0, 0, 0, 0};
    Rect<float> border{0, 0, 0, 0};
    Rect<float> content_box_inset{0, 0, 0, 0};
    Point<float> scrollbar_gutter{0.0f, 0.0f};
    Size<float> gap{0.0f, 0.0f};
    AlignItems align_items{};
    AlignContent align_content{};
    std::optional<JustifyContent> justify_content{};
    Size<std::optional<float>> node_outer_size{};
    Size<std::optional<float>> node_inner_size{};
    Size<float> container_size{};
    Size<float> inner_container_size{};
};

// --- Flexbox algorithm template functions ---
// These accept any tree-like object with the required methods.

template <typename Tree>
inline FlexAlgoConstants flex_compute_constants(Tree& tree, NodeId node, const Style& style,
    Size<std::optional<float>> known_dimensions, Size<std::optional<float>> parent_size) {
    FlexAlgoConstants c;
    c.dir = style.flex_direction;
    c.is_row = ::taffy::is_row(c.dir);
    c.is_column = !c.is_row;
    c.is_wrap = style.flex_wrap == FlexWrap::Wrap || style.flex_wrap == FlexWrap::WrapReverse;
    c.is_wrap_reverse = style.flex_wrap == FlexWrap::WrapReverse;
    c.layout_direction = style.direction;

    auto padding = resolve_or_zero_rect(style.padding, parent_size.width);
    auto border = resolve_or_zero_rect(style.border, parent_size.width);
    auto pb_sum = padding.sum_axes() + border.sum_axes();
    auto box_adj = (style.box_sizing == BoxSizing::ContentBox) ? pb_sum : Size_f32_ZERO;

    c.margin = resolve_or_zero_rect(style.margin, parent_size.width);
    c.border = border;
    c.content_box_inset = padding + border;
    // Scrollbar gutter
    Point<float> sg{0, 0};
    if (style.overflow.y == Overflow::Scroll) sg.x = style.scrollbar_width;
    if (style.overflow.x == Overflow::Scroll) sg.y = style.scrollbar_width;
    c.scrollbar_gutter = sg;
    c.content_box_inset.bottom += sg.y;
    if (c.layout_direction == Direction::Ltr) c.content_box_inset.right += sg.x;
    else c.content_box_inset.left += sg.x;

    c.min_size = maybe_resolve_size(style.min_size, parent_size);
    if (style.aspect_ratio) {
        if (c.min_size.width && !c.min_size.height) c.min_size.height = *c.min_size.width / *style.aspect_ratio;
        if (!c.min_size.width && c.min_size.height) c.min_size.width = *c.min_size.height * *style.aspect_ratio;
    }
    c.max_size = maybe_resolve_size(style.max_size, parent_size);
    if (style.aspect_ratio) {
        if (c.max_size.width && !c.max_size.height) c.max_size.height = *c.max_size.width / *style.aspect_ratio;
        if (!c.max_size.width && c.max_size.height) c.max_size.width = *c.max_size.height * *style.aspect_ratio;
    }

    c.align_items = style.align_items.value_or(AlignItems_STRETCH);
    c.align_content = style.align_content.value_or(AlignContent_STRETCH);
    c.justify_content = style.justify_content;

    c.node_outer_size = known_dimensions;
    c.node_inner_size = {
        known_dimensions.width ? std::optional(*known_dimensions.width - c.content_box_inset.horizontal_axis_sum()) : std::nullopt,
        known_dimensions.height ? std::optional(*known_dimensions.height - c.content_box_inset.vertical_axis_sum()) : std::nullopt
    };

    auto inner = c.node_inner_size.width ? std::optional(*c.node_inner_size.width) : std::optional<float>(0.0f);
    c.gap = {
        resolve_or_zero(style.gap.width, inner),
        resolve_or_zero(style.gap.height, inner)
    };
    c.container_size = Size_f32_ZERO;
    c.inner_container_size = Size_f32_ZERO;
    return c;
}

template <typename Tree>
inline std::vector<FlexItem> flex_generate_items(Tree& tree, NodeId node, const FlexAlgoConstants& c) {
    std::vector<FlexItem> items;
    size_t count = tree.child_count(node);
    for (size_t i = 0; i < count; ++i) {
        NodeId child = tree.get_child_id(node, i);
        const auto& cs = tree.style(child);
        if (cs.position == Position::Absolute) continue;

        auto padding = resolve_or_zero_rect(cs.padding, c.node_inner_size.width);
        auto border = resolve_or_zero_rect(cs.border, c.node_inner_size.width);
        auto pb = (padding + border).sum_axes();
        auto box_adj = (cs.box_sizing == BoxSizing::ContentBox) ? pb : Size_f32_ZERO;

        FlexItem item;
        item.node = child;
        item.order = static_cast<uint32_t>(i);
        item.size = maybe_resolve_size(cs.size, c.node_inner_size);
        item.min_size = maybe_resolve_size(cs.min_size, c.node_inner_size);
        item.max_size = maybe_resolve_size(cs.max_size, c.node_inner_size);
        if (cs.aspect_ratio) {
            auto apply_ar = [&](Size<std::optional<float>>& s) {
                if (s.width && !s.height) s.height = *s.width / *cs.aspect_ratio;
                if (!s.width && s.height) s.width = *s.height * *cs.aspect_ratio;
            };
            apply_ar(item.size); apply_ar(item.min_size); apply_ar(item.max_size);
        }
        item.inset = {
            maybe_resolve(cs.inset.left, c.node_inner_size.width),
            maybe_resolve(cs.inset.right, c.node_inner_size.width),
            maybe_resolve(cs.inset.top, c.node_inner_size.height),
            maybe_resolve(cs.inset.bottom, c.node_inner_size.height)
        };
        item.margin = resolve_or_zero_rect(cs.margin, c.node_inner_size.width);
        item.margin_is_auto = {cs.margin.left.is_auto(), cs.margin.right.is_auto(), cs.margin.top.is_auto(), cs.margin.bottom.is_auto()};
        item.padding = padding;
        item.border = border;
        item.align_self = cs.align_self.value_or(c.align_items);
        item.overflow = cs.overflow;
        item.scrollbar_width = cs.scrollbar_width;
        item.flex_grow = cs.flex_grow;
        item.flex_shrink = cs.flex_shrink;
        items.push_back(item);
    }
    return items;
}

template <typename Tree, typename MeasureFn>
inline void flex_determine_base_size(Tree& tree, const FlexAlgoConstants& c,
    Size<AvailableSpace> available_space, std::vector<FlexItem>& items, MeasureFn& measure_fn) {
    for (auto& child : items) {
        const auto& cs = tree.style(child.node);
        auto cross_parent = optf32_cross(c.node_inner_size, c.dir);
        auto container_width = optf32_main(c.node_inner_size, c.dir);
        auto padding = resolve_or_zero_rect(cs.padding, container_width);
        auto border = resolve_or_zero_rect(cs.border, container_width);
        auto pb = (padding + border).sum_axes();
        auto box_adj = (cs.box_sizing == BoxSizing::ContentBox) ? pb : Size_f32_ZERO;
        auto flex_basis_dim = maybe_resolve(cs.flex_basis, container_width);
        float box_sizing_main = is_row(c.dir) ? box_adj.width : box_adj.height;
        auto flex_basis = flex_basis_dim ? std::optional(*flex_basis_dim + box_sizing_main) : std::optional<float>{};
        auto main_size = optf32_main(child.size, c.dir);

        float resolved_basis = 0.0f;
        if (flex_basis) resolved_basis = *flex_basis;
        else if (main_size) resolved_basis = *main_size;
        else {
            // E: size into available space treating content as max-content
            LayoutInput li;
            li.run_mode = RunMode::ComputeSize;
            li.sizing_mode = SizingMode::ContentSize;
            li.known_dimensions = optf32_with_main(child.size, c.dir, std::nullopt);
            li.parent_size = c.node_inner_size;
            li.available_space = Size<AvailableSpace>{AvailableSpace::max_content(), AvailableSpace::max_content()};
            auto output = tree.compute_node_layout(child.node, li, measure_fn);
            resolved_basis = is_row(c.dir) ? output.size.width : output.size.height;
        }

        float pb_main = child.padding.main_axis_sum(c.dir) + child.border.main_axis_sum(c.dir);
        resolved_basis = std::max(resolved_basis, pb_main);
        child.flex_basis = resolved_basis;
        child.inner_flex_basis = resolved_basis - pb_main;

        // Resolve minimum main size
        auto style_min_main = optf32_main(child.min_size, c.dir);
        if (!style_min_main) {
            // Auto min: min-content clamped by preferred and max
            LayoutInput li;
            li.run_mode = RunMode::ComputeSize;
            li.sizing_mode = SizingMode::ContentSize;
            li.known_dimensions = optf32_with_main(child.size, c.dir, std::nullopt);
            li.parent_size = c.node_inner_size;
            li.available_space = Size<AvailableSpace>{AvailableSpace::min_content(), AvailableSpace::min_content()};
            auto output = tree.compute_node_layout(child.node, li, measure_fn);
            float min_content = is_row(c.dir) ? output.size.width : output.size.height;
            float clamped = min_content;
            if (child.size.main(c.dir)) clamped = std::min(clamped, *child.size.main(c.dir));
            if (child.max_size.main(c.dir)) clamped = std::min(clamped, *child.max_size.main(c.dir));
            float pb_sum = child.padding.main_axis_sum(c.dir) + child.border.main_axis_sum(c.dir);
            child.resolved_minimum_main_size = std::max(clamped, pb_sum);
        } else {
            child.resolved_minimum_main_size = *style_min_main;
        }

        // Hypothetical sizes
        float pb_total = child.padding.main_axis_sum(c.dir) + child.border.main_axis_sum(c.dir);
        float hypo_min = std::max(child.resolved_minimum_main_size, pb_total);
        float hypo_inner = child.flex_basis;
        if (child.max_size.main(c.dir)) hypo_inner = std::min(hypo_inner, *child.max_size.main(c.dir));
        hypo_inner = std::max(hypo_inner, hypo_min);
        float hypo_outer = hypo_inner + child.margin.main_axis_sum(c.dir);
        child.hypothetical_inner_size.set_main(c.dir, hypo_inner);
        child.hypothetical_outer_size.set_main(c.dir, hypo_outer);
    }
}

inline std::vector<FlexLine> flex_collect_lines(const FlexAlgoConstants& c,
    Size<AvailableSpace> available_space, std::vector<FlexItem>& all_items) {
    std::vector<FlexLine> lines;
    if (!c.is_wrap) {
        FlexLine line;
        line.items = std::move(all_items);
        lines.push_back(std::move(line));
        return lines;
    }
    // Wrapping: collect items into lines based on available main axis space
    auto main_avail = available_space.main(c.dir);
    if (main_avail == AvailableSpace::max_content()) {
        FlexLine line;
        line.items = std::move(all_items);
        lines.push_back(std::move(line));
        return lines;
    }
    if (main_avail == AvailableSpace::min_content()) {
        for (auto& item : all_items) {
            FlexLine line;
            line.items.push_back(std::move(item));
            lines.push_back(std::move(line));
        }
        return lines;
    }
    // Definite available space
    float main_space = main_avail.value;
    float line_length = 0.0f;
    size_t line_start = 0;
    float main_gap = c.gap.main(c.dir);
    for (size_t i = 0; i < all_items.size(); ++i) {
        float gap = (i == line_start) ? 0.0f : main_gap;
        float item_main = all_items[i].hypothetical_outer_size.main(c.dir);
        if (line_length + gap + item_main > main_space && i > line_start) {
            FlexLine line;
            for (size_t j = line_start; j < i; ++j) line.items.push_back(std::move(all_items[j]));
            lines.push_back(std::move(line));
            line_start = i;
            line_length = item_main;
        } else {
            line_length += gap + item_main;
        }
    }
    if (line_start < all_items.size()) {
        FlexLine line;
        for (size_t j = line_start; j < all_items.size(); ++j) line.items.push_back(std::move(all_items[j]));
        lines.push_back(std::move(line));
    }
    return lines;
}

inline void flex_resolve_flexible_lengths(FlexLine& line, const FlexAlgoConstants& c) {
    float total_gap = sum_axis_gaps(c.gap.main(c.dir), line.items.size());
    float total_hypo = total_gap;
    for (auto& item : line.items) total_hypo += item.hypothetical_outer_size.main(c.dir);
    float inner_main = c.node_inner_size.main(c.dir).value_or(0.0f);
    bool growing = total_hypo < inner_main;
    bool shrinking = total_hypo > inner_main;
    bool exact = !growing && !shrinking;

    // Freeze inflexible items
    for (auto& item : line.items) {
        float target = item.hypothetical_inner_size.main(c.dir);
        item.target_size.set_main(c.dir, target);
        item.outer_target_size.set_main(c.dir, target + item.margin.main_axis_sum(c.dir));
        if (exact || (item.flex_grow == 0 && item.flex_shrink == 0)
            || (growing && item.flex_basis > item.hypothetical_inner_size.main(c.dir))
            || (shrinking && item.flex_basis < item.hypothetical_inner_size.main(c.dir))) {
            item.frozen = true;
        }
    }
    if (exact) return;

    // Calculate initial free space
    float used = total_gap;
    for (auto& item : line.items) {
        used += item.frozen ? item.outer_target_size.main(c.dir)
                            : item.flex_basis + item.margin.main_axis_sum(c.dir);
    }
    float initial_free = c.node_inner_size.main(c.dir).value_or(0.0f) - used;

    // Loop
    for (int iter = 0; iter < 100; ++iter) {
        bool all_frozen = true;
        for (auto& item : line.items) { if (!item.frozen) { all_frozen = false; break; } }
        if (all_frozen) break;

        float sum_grow = 0, sum_shrink = 0;
        float used2 = total_gap;
        for (auto& item : line.items) {
            if (item.frozen) used2 += item.outer_target_size.main(c.dir);
            else { used2 += item.flex_basis + item.margin.main_axis_sum(c.dir); sum_grow += item.flex_grow; sum_shrink += item.flex_shrink; }
        }

        float free_space;
        if (growing && sum_grow < 1.0f) free_space = std::min(initial_free * sum_grow, inner_main - used2);
        else if (shrinking && sum_shrink < 1.0f) free_space = std::max(initial_free * sum_shrink, inner_main - used2);
        else free_space = c.node_inner_size.main(c.dir).value_or(total_hypo) - used2;

        if (std::abs(free_space) > 1e-6f) {
            if (growing && sum_grow > 0) {
                for (auto& item : line.items) {
                    if (!item.frozen) item.target_size.set_main(c.dir, item.flex_basis + free_space * (item.flex_grow / sum_grow));
                }
            } else if (shrinking && sum_shrink > 0) {
                float sum_scaled = 0;
                for (auto& item : line.items) if (!item.frozen) sum_scaled += item.inner_flex_basis * item.flex_shrink;
                if (sum_scaled > 0) {
                    for (auto& item : line.items) {
                        if (!item.frozen) {
                            float scaled = item.inner_flex_basis * item.flex_shrink;
                            item.target_size.set_main(c.dir, item.flex_basis + free_space * (scaled / sum_scaled));
                        }
                    }
                }
            }
        }

        // Fix min/max violations
        float total_violation = 0;
        for (auto& item : line.items) {
            if (item.frozen) continue;
            float target = item.target_size.main(c.dir);
            float clamped = std::clamp(target, item.resolved_minimum_main_size,
                item.max_size.main(c.dir).value_or(std::numeric_limits<float>::infinity()));
            clamped = std::max(clamped, 0.0f);
            item.violation = clamped - target;
            item.target_size.set_main(c.dir, clamped);
            item.outer_target_size.set_main(c.dir, clamped + item.margin.main_axis_sum(c.dir));
            total_violation += item.violation;
        }

        // Freeze over-flexed items
        for (auto& item : line.items) {
            if (item.frozen) continue;
            if (total_violation > 0) item.frozen = item.violation > 0;
            else if (total_violation < 0) item.frozen = item.violation < 0;
            else item.frozen = true;
        }
    }
}

template <typename Tree, typename MeasureFn>
inline void flex_determine_hypothetical_cross(Tree& tree, FlexLine& line, const FlexAlgoConstants& c,
    Size<AvailableSpace> available_space, MeasureFn& measure_fn) {
    for (auto& child : line.items) {
        float pb_cross = child.padding.cross_axis_sum(c.dir) + child.border.cross_axis_sum(c.dir);
        float target_main = child.target_size.main(c.dir);
        Size<std::optional<float>> known = is_row(c.dir)
            ? Size<std::optional<float>>{target_main, child.size.cross(c.dir)}
            : Size<std::optional<float>>{child.size.cross(c.dir), target_main};

        float cross_size = child.size.cross(c.dir).value_or(0.0f);
        if (!child.size.cross(c.dir)) {
            LayoutInput li;
            li.run_mode = RunMode::ComputeSize;
            li.sizing_mode = SizingMode::ContentSize;
            li.known_dimensions = known;
            li.parent_size = c.node_inner_size;
            li.available_space = available_space;
            auto out = tree.compute_node_layout(child.node, li, measure_fn);
            cross_size = is_row(c.dir) ? out.size.height : out.size.width;
        }
        if (child.min_size.cross(c.dir)) cross_size = std::max(cross_size, *child.min_size.cross(c.dir));
        if (child.max_size.cross(c.dir)) cross_size = std::min(cross_size, *child.max_size.cross(c.dir));
        cross_size = std::max(cross_size, pb_cross);

        float outer_cross = cross_size + child.margin.cross_axis_sum(c.dir);
        child.hypothetical_inner_size.set_cross(c.dir, cross_size);
        child.hypothetical_outer_size.set_cross(c.dir, outer_cross);
    }
}

inline void flex_calculate_cross_size(std::vector<FlexLine>& lines, Size<std::optional<float>> node_size, const FlexAlgoConstants& c) {
    if (!c.is_wrap && node_size.cross(c.dir)) {
        float pb_cross = c.content_box_inset.cross_axis_sum(c.dir);
        float cross = *node_size.cross(c.dir);
        if (c.min_size.cross(c.dir)) cross = std::max(cross, *c.min_size.cross(c.dir));
        if (c.max_size.cross(c.dir)) cross = std::min(cross, *c.max_size.cross(c.dir));
        lines[0].cross_size = std::max(cross - pb_cross, 0.0f);
    } else {
        for (auto& line : lines) {
            float max_baseline = 0;
            for (auto& item : line.items) max_baseline = std::max(max_baseline, item.baseline);
            float line_cross = 0;
            for (auto& item : line.items) {
                float item_cross = item.hypothetical_outer_size.cross(c.dir);
                if (item.align_self == AlignItems_BASELINE && !item.margin_is_auto.cross_start(c.dir) && !item.margin_is_auto.cross_end(c.dir)) {
                    item_cross = max_baseline - item.baseline + item.hypothetical_outer_size.cross(c.dir);
                }
                line_cross = std::max(line_cross, item_cross);
            }
            line.cross_size = line_cross;
        }
        if (!c.is_wrap && lines.size() == 1) {
            float pb_cross = c.content_box_inset.cross_axis_sum(c.dir);
            float v = lines[0].cross_size;
            if (c.min_size.cross(c.dir)) v = std::max(v, *c.min_size.cross(c.dir) - pb_cross);
            if (c.max_size.cross(c.dir)) v = std::min(v, *c.max_size.cross(c.dir) - pb_cross);
            lines[0].cross_size = v;
        }
    }
}

inline void flex_handle_align_content_stretch(std::vector<FlexLine>& lines, Size<std::optional<float>> node_size, const FlexAlgoConstants& c) {
    if (c.align_content == AlignContent_STRETCH) {
        float pb_cross = c.content_box_inset.cross_axis_sum(c.dir);
        float min_inner = node_size.cross(c.dir).value_or(0.0f);
        if (c.min_size.cross(c.dir)) min_inner = std::max(min_inner, *c.min_size.cross(c.dir));
        if (c.max_size.cross(c.dir)) min_inner = std::min(min_inner, *c.max_size.cross(c.dir));
        min_inner = std::max(min_inner - pb_cross, 0.0f);
        float total_cross_gap = sum_axis_gaps(c.gap.cross(c.dir), lines.size());
        float total = 0; for (auto& l : lines) total += l.cross_size;
        if (total + total_cross_gap < min_inner) {
            float add = (min_inner - total - total_cross_gap) / lines.size();
            for (auto& l : lines) l.cross_size += add;
        }
    }
}

inline void flex_determine_used_cross_size(std::vector<FlexLine>& lines, const FlexAlgoConstants& c) {
    for (auto& line : lines) {
        for (auto& child : line.items) {
            float line_cross = line.cross_size;
            float target_cross;
            if (child.align_self == AlignItems_STRETCH
                && !child.margin_is_auto.cross_start(c.dir) && !child.margin_is_auto.cross_end(c.dir)
                && !child.size.cross(c.dir)) {
                target_cross = std::max(line_cross - child.margin.cross_axis_sum(c.dir), 0.0f);
                if (child.min_size.cross(c.dir)) target_cross = std::max(target_cross, *child.min_size.cross(c.dir));
                if (child.max_size.cross(c.dir)) target_cross = std::min(target_cross, *child.max_size.cross(c.dir));
            } else {
                target_cross = child.hypothetical_inner_size.cross(c.dir);
            }
            child.target_size.set_cross(c.dir, target_cross);
            child.outer_target_size.set_cross(c.dir, target_cross + child.margin.cross_axis_sum(c.dir));
        }
    }
}

inline void flex_distribute_remaining_free_space(std::vector<FlexLine>& lines, const FlexAlgoConstants& c) {
    for (auto& line : lines) {
        float total_gap = sum_axis_gaps(c.gap.main(c.dir), line.items.size());
        float used = total_gap;
        for (auto& item : line.items) used += item.outer_target_size.main(c.dir);
        float free_space = c.inner_container_size.main(c.dir) - used;
        int num_auto = 0;
        for (auto& item : line.items) {
            if (item.margin_is_auto.main_start(c.dir)) num_auto++;
            if (item.margin_is_auto.main_end(c.dir)) num_auto++;
        }
        if (free_space > 0 && num_auto > 0) {
            float margin = free_space / num_auto;
            for (auto& item : line.items) {
                if (item.margin_is_auto.main_start(c.dir)) { if (c.is_row) item.margin.left = margin; else item.margin.top = margin; }
                if (item.margin_is_auto.main_end(c.dir)) { if (c.is_row) item.margin.right = margin; else item.margin.bottom = margin; }
            }
        }
        // Justify content
        auto jc = c.justify_content.value_or(AlignContent_FLEX_START);
        auto jc_mode = apply_alignment_fallback(free_space, line.items.size(), jc);
        bool reversed = is_reverse(c.dir);
        float gap = c.gap.main(c.dir);
        if (reversed) {
            for (size_t i = 0; i < line.items.size(); ++i) {
                line.items[line.items.size() - 1 - i].offset_main =
                    compute_alignment_offset(free_space, line.items.size(), gap, jc_mode, true, i == 0);
            }
        } else {
            for (size_t i = 0; i < line.items.size(); ++i) {
                line.items[i].offset_main =
                    compute_alignment_offset(free_space, line.items.size(), gap, jc_mode, false, i == 0);
            }
        }
    }
}

inline void flex_resolve_cross_axis_auto_margins(std::vector<FlexLine>& lines, const FlexAlgoConstants& c) {
    for (auto& line : lines) {
        float line_cross = line.cross_size;
        float max_baseline = 0;
        for (auto& item : line.items) max_baseline = std::max(max_baseline, item.baseline);
        for (auto& child : line.items) {
            float free = line_cross - child.outer_target_size.cross(c.dir);
            if (child.margin_is_auto.cross_start(c.dir) && child.margin_is_auto.cross_end(c.dir)) {
                if (c.is_row) { child.margin.top = free / 2; child.margin.bottom = free / 2; }
                else { child.margin.left = free / 2; child.margin.right = free / 2; }
            } else if (child.margin_is_auto.cross_start(c.dir)) {
                if (c.is_row) child.margin.top = free; else child.margin.left = free;
            } else if (child.margin_is_auto.cross_end(c.dir)) {
                if (c.is_row) child.margin.bottom = free; else child.margin.right = free;
            } else {
                // Cross-axis alignment
                bool cross_reverse = c.is_column && c.layout_direction == Direction::Rtl;
                AlignItemsKeyword kw = child.align_self.keyword;
                if (child.align_self.is_safe() && free < 0) kw = AlignItemsKeyword::Start;
                float offset = 0;
                switch (kw) {
                    case AlignItemsKeyword::Start: offset = cross_reverse ? free : 0; break;
                    case AlignItemsKeyword::FlexStart: offset = (c.is_wrap_reverse ^ cross_reverse) ? free : 0; break;
                    case AlignItemsKeyword::End: offset = cross_reverse ? 0 : free; break;
                    case AlignItemsKeyword::FlexEnd: offset = (c.is_wrap_reverse ^ cross_reverse) ? 0 : free; break;
                    case AlignItemsKeyword::Center: offset = free / 2; break;
                    case AlignItemsKeyword::Baseline: offset = c.is_row ? max_baseline - child.baseline : ((c.is_wrap_reverse ^ cross_reverse) ? free : 0); break;
                    case AlignItemsKeyword::Stretch: offset = (c.is_wrap_reverse ^ cross_reverse) ? free : 0; break;
                }
                child.offset_cross = offset;
            }
        }
    }
}

inline float flex_determine_container_cross_size(std::vector<FlexLine>& lines, Size<std::optional<float>> node_size, FlexAlgoConstants& c) {
    float total_cross_gap = sum_axis_gaps(c.gap.cross(c.dir), lines.size());
    float total_line_cross = 0; for (auto& l : lines) total_line_cross += l.cross_size;
    float pb_cross = c.content_box_inset.cross_axis_sum(c.dir);
    float cross_scrollbar = c.scrollbar_gutter.cross(c.dir);
    float outer = node_size.cross(c.dir).value_or(total_line_cross + total_cross_gap + pb_cross);
    if (c.min_size.cross(c.dir)) outer = std::max(outer, *c.min_size.cross(c.dir));
    if (c.max_size.cross(c.dir)) outer = std::min(outer, *c.max_size.cross(c.dir));
    outer = std::max(outer, pb_cross - cross_scrollbar);
    float inner = std::max(outer - pb_cross, 0.0f);
    c.container_size.set_cross(c.dir, outer);
    c.inner_container_size.set_cross(c.dir, inner);
    return total_line_cross;
}

inline void flex_align_lines_per_align_content(std::vector<FlexLine>& lines, const FlexAlgoConstants& c, float total_cross) {
    float gap = c.gap.cross(c.dir);
    float total_gap = sum_axis_gaps(gap, lines.size());
    float free = c.inner_container_size.cross(c.dir) - total_cross - total_gap;
    auto mode = apply_alignment_fallback(free, lines.size(), c.align_content);
    bool reversed = c.is_wrap_reverse;
    if (reversed) {
        for (size_t i = 0; i < lines.size(); ++i)
            lines[lines.size() - 1 - i].offset_cross = compute_alignment_offset(free, lines.size(), gap, mode, true, i == 0);
    } else {
        for (size_t i = 0; i < lines.size(); ++i)
            lines[i].offset_cross = compute_alignment_offset(free, lines.size(), gap, mode, false, i == 0);
    }
}

template <typename Tree, typename MeasureFn>
inline Size<float> flex_final_layout_pass(Tree& tree, std::vector<FlexLine>& lines,
    const FlexAlgoConstants& c, MeasureFn& measure_fn, Size<float>& content_size_out) {
    float total_offset_cross = c.content_box_inset.cross_start(c.dir);
    Size<float> content_size = Size_f32_ZERO;
    auto process_line = [&](FlexLine& line) {
        float offset_main = c.layout_direction == Direction::Rtl && c.is_row
            ? c.container_size.width - c.content_box_inset.main_end(c.dir)
            : c.content_box_inset.main_start(c.dir);
        float line_offset_cross = line.offset_cross;
        auto process_item = [&](FlexItem& item) {
            LayoutInput li;
            li.run_mode = RunMode::PerformLayout;
            li.sizing_mode = SizingMode::ContentSize;
            li.known_dimensions = {item.target_size.width, item.target_size.height};
            li.parent_size = c.node_inner_size;
            li.available_space = {AvailableSpace::definite(c.container_size.width), AvailableSpace::definite(c.container_size.height)};
            auto output = tree.compute_node_layout(item.node, li, measure_fn);

            bool rtl_row = c.is_row && c.layout_direction == Direction::Rtl;
            float main_rel = 0;
            if (rtl_row) main_rel = item.inset.main_end(c.dir).value_or(0.0f);
            else main_rel = item.inset.main_start(c.dir).value_or(0.0f);

            float cross_rel = item.inset.cross_start(c.dir).value_or(0.0f);
            float off_main = rtl_row
                ? offset_main - item.offset_main - item.margin.main_end(c.dir) - main_rel - output.size.main(c.dir)
                : offset_main + item.offset_main + item.margin.main_start(c.dir) + main_rel;
            float off_cross = total_offset_cross + item.offset_cross + line_offset_cross + item.margin.cross_start(c.dir) + cross_rel;

            Layout layout_result;
            layout_result.order = item.order;
            layout_result.location = c.is_row ? Point<float>{off_main, off_cross} : Point<float>{off_cross, off_main};
            layout_result.size = output.size;
#ifdef TAFFY_CONTENT_SIZE
            layout_result.content_size = output.content_size;
#endif
            layout_result.scrollbar_size = {
                item.overflow.y == Overflow::Scroll ? item.scrollbar_width : 0.0f,
                item.overflow.x == Overflow::Scroll ? item.scrollbar_width : 0.0f
            };
            layout_result.padding = item.padding;
            layout_result.border = item.border;
            layout_result.margin = item.margin;
            tree.set_unrounded_layout(item.node, layout_result);

            if (rtl_row) offset_main -= item.offset_main + item.margin.main_axis_sum(c.dir) + output.size.main(c.dir);
            else offset_main += item.offset_main + item.margin.main_axis_sum(c.dir) + output.size.main(c.dir);
        };

        if (is_reverse(c.dir)) {
            for (auto it = line.items.rbegin(); it != line.items.rend(); ++it) process_item(*it);
        } else {
            for (auto& item : line.items) process_item(item);
        }
        total_offset_cross += line_offset_cross + line.cross_size;
    };

    if (c.is_wrap_reverse) {
        for (auto it = lines.rbegin(); it != lines.rend(); ++it) process_line(*it);
    } else {
        for (auto& line : lines) process_line(line);
    }
    return content_size;
}

#endif // TAFFY_FLEXBOX

// ===========================================================================
// Section 15b: Grid Algorithm - Explicit/Implicit Grid Resolution
// Ref: src/compute/grid/explicit_grid.rs, implicit_grid.rs
// ===========================================================================

#ifdef TAFFY_GRID

namespace grid_detail {

enum class AutoRepeatStrategy : uint8_t { MaxRepetitionsThatDoNotOverflow, MinRepetitionsThatDoOverflow };

// Compute the number of auto-repeat repetitions and total explicit track count in an axis
inline std::pair<uint16_t, uint16_t> compute_explicit_grid_size_in_axis(
    const Style& style, std::optional<float> auto_fit_container_size,
    AutoRepeatStrategy strategy, AbsoluteAxis axis) {
    auto& template_tracks = (axis == AbsoluteAxis::Horizontal) ? style.grid_template_columns : style.grid_template_rows;
    if (template_tracks.empty()) return {0, 0};

    // Count non-auto-repeating tracks
    uint16_t non_auto_count = 0;
    bool has_auto_repeat = false;
    bool all_fixed = true;
    for (auto& def : template_tracks) {
        if (def.kind == GridTemplateComponent::Single) {
            non_auto_count++;
            if (!def.single_track.min.is_length_or_percentage() && !def.single_track.min.is_zero())
                all_fixed = false;
        } else {
            if (def.repeat_val.count.kind == RepetitionCount::Count) {
                non_auto_count += def.repeat_val.count.count_val * static_cast<uint16_t>(def.repeat_val.tracks.size());
            } else {
                has_auto_repeat = true;
                for (auto& tsf : def.repeat_val.tracks) {
                    if (!tsf.min.is_length_or_percentage() && !tsf.min.is_zero()) all_fixed = false;
                }
            }
        }
    }

    if (!has_auto_repeat) return {0, non_auto_count};
    if (!all_fixed) return {0, 0}; // Invalid: auto-repeat with non-fixed tracks

    // Find auto-repeat definition
    uint16_t rep_track_count = 0;
    for (auto& def : template_tracks) {
        if (def.kind == GridTemplateComponent::Repeat && def.repeat_val.count.kind != RepetitionCount::Count) {
            rep_track_count = static_cast<uint16_t>(def.repeat_val.tracks.size());
            break;
        }
    }
    if (rep_track_count == 0) return {0, 0};

    if (!auto_fit_container_size) return {1, non_auto_count + rep_track_count};

    // Compute repetitions that fit
    float container = *auto_fit_container_size;
    float per_rep_size = 0;
    for (auto& def : template_tracks) {
        if (def.kind == GridTemplateComponent::Repeat && def.repeat_val.count.kind != RepetitionCount::Count) {
            for (auto& tsf : def.repeat_val.tracks) {
                if (tsf.max.is_length_or_percentage()) per_rep_size += tsf.max.value();
                else if (tsf.min.is_length_or_percentage()) per_rep_size += tsf.min.value();
            }
        }
    }

    uint16_t num_reps = 1;
    if (per_rep_size > 0) {
        float remaining = container - per_rep_size; // First repetition already counted
        if (remaining > 0) {
            uint16_t extra = (strategy == AutoRepeatStrategy::MaxRepetitionsThatDoNotOverflow)
                ? static_cast<uint16_t>(std::floor(remaining / per_rep_size))
                : static_cast<uint16_t>(std::ceil(remaining / per_rep_size));
            num_reps += extra;
        }
    }

    return {num_reps, non_auto_count + rep_track_count * num_reps};
}

// Initialize grid tracks from style (stores only actual tracks, not gutters)
inline void initialize_grid_tracks(
    std::vector<GridTrack>& tracks, TrackCounts counts,
    const Style& style, AbsoluteAxis axis,
    std::function<bool(size_t)> track_has_items) {
    auto& template_tracks = (axis == AbsoluteAxis::Horizontal) ? style.grid_template_columns : style.grid_template_rows;
    auto& auto_tracks = (axis == AbsoluteAxis::Horizontal) ? style.grid_auto_columns : style.grid_auto_rows;

    tracks.clear();
    size_t total = counts.len();
    tracks.reserve(total);

    auto default_auto = minmax(CompactLength::length(0), CompactLength::max_content());

    // Negative implicit tracks
    for (uint16_t i = 0; i < counts.negative_implicit; ++i) {
        TrackSizingFunction tsf = auto_tracks.empty() ? default_auto : auto_tracks[i % auto_tracks.size()];
        tracks.push_back(GridTrack::new_track(tsf.min, tsf.max));
    }

    // Explicit tracks from template
    for (auto& def : template_tracks) {
        if (def.kind == GridTemplateComponent::Single) {
            tracks.push_back(GridTrack::new_track(def.single_track.min, def.single_track.max));
        } else {
            uint16_t rep_count = 1;
            if (def.repeat_val.count.kind == RepetitionCount::Count)
                rep_count = def.repeat_val.count.count_val;
            for (uint16_t r = 0; r < rep_count; ++r) {
                for (auto& tsf : def.repeat_val.tracks) {
                    tracks.push_back(GridTrack::new_track(tsf.min, tsf.max));
                }
            }
        }
    }

    // Positive implicit tracks
    for (uint16_t i = 0; i < counts.positive_implicit; ++i) {
        TrackSizingFunction tsf = auto_tracks.empty() ? default_auto : auto_tracks[i % auto_tracks.size()];
        tracks.push_back(GridTrack::new_track(tsf.min, tsf.max));
    }
}

// Estimate grid size from child styles
inline std::pair<TrackCounts, TrackCounts> compute_grid_size_estimate(
    uint16_t explicit_col_count, uint16_t explicit_row_count,
    Direction direction, const std::vector<Style>& child_styles) {
    int16_t col_min = 0, col_max = 0, col_max_span = 0;
    int16_t row_min = 0, row_max = 0, row_max_span = 0;

    for (auto& cs : child_styles) {
        auto process_line = [&](const Line<GridPlacement>& line, int16_t explicit_count,
            int16_t& out_min, int16_t& out_max, int16_t& out_max_span) {
            auto resolve = [&](const GridPlacement& gp) -> int16_t {
                if (gp.kind == GridPlacement::LineKind) {
                    int16_t v = gp.line_val.value;
                    return v > 0 ? v - 1 : v + explicit_count;
                }
                return 0;
            };
            int16_t start = resolve(line.start);
            int16_t end_val = resolve(line.end);
            int16_t span = line.start.kind == GridPlacement::SpanKind ? line.start.span_val :
                           line.end.kind == GridPlacement::SpanKind ? line.end.span_val : 1;

            if (line.start.is_definite() && line.end.is_definite()) {
                out_min = std::min(out_min, std::min(start, end_val));
                out_max = std::max(out_max, std::max(start, end_val));
            } else if (line.start.is_definite()) {
                out_min = std::min(out_min, start);
                out_max = std::max(out_max, (int16_t)(start + span));
            } else if (line.end.is_definite()) {
                out_min = std::min(out_min, (int16_t)(end_val - span));
                out_max = std::max(out_max, end_val);
            }
            out_max_span = std::max(out_max_span, span);
        };

        if (direction == Direction::Rtl) {
            // Mirror columns for RTL
            auto mirrored_col = cs.grid_column;
            if (mirrored_col.start.is_definite() || mirrored_col.end.is_definite()) {
                int16_t end_line = explicit_col_count;
                auto mirror = [&](const GridPlacement& gp) -> GridPlacement {
                    if (gp.kind == GridPlacement::LineKind) {
                        int16_t v = gp.line_val.value;
                        return GridPlacement::line(GridLine(static_cast<int16_t>(end_line - v + (v > 0 ? 2 : 0))));
                    }
                    return gp;
                };
                Line<GridPlacement> mirrored{mirror(mirrored_col.end), mirror(mirrored_col.start)};
                process_line(mirrored, explicit_col_count, col_min, col_max, col_max_span);
            }
        } else {
            process_line(cs.grid_column, explicit_col_count, col_min, col_max, col_max_span);
        }
        process_line(cs.grid_row, explicit_row_count, row_min, row_max, row_max_span);
    }

    auto make_counts = [](int16_t mn, int16_t mx, int16_t max_span, uint16_t explicit_count) -> TrackCounts {
        uint16_t neg = mn < 0 ? static_cast<uint16_t>(-mn) : 0;
        uint16_t pos = mx > explicit_count ? static_cast<uint16_t>(mx - explicit_count) : 0;
        uint16_t total = neg + explicit_count + pos;
        if (total < max_span) pos += max_span - total;
        return TrackCounts{neg, explicit_count, pos};
    };

    return {make_counts(col_min, col_max, col_max_span, explicit_col_count),
            make_counts(row_min, row_max, row_max_span, explicit_row_count)};
}

} // namespace grid_detail

// --- Grid placement algorithm ---
namespace grid_detail {

inline AbsoluteAxis primary_axis_from_flow(GridAutoFlow flow) {
    return (flow == GridAutoFlow::Row || flow == GridAutoFlow::RowDense) ? AbsoluteAxis::Horizontal : AbsoluteAxis::Vertical;
}

inline OriginZeroLine placement_to_oz(const GridPlacement& gp, uint16_t explicit_count) {
    if (gp.kind == GridPlacement::LineKind) {
        int16_t v = gp.line_val.value;
        return OriginZeroLine{v > 0 ? static_cast<int16_t>(v - 1) : static_cast<int16_t>(v + explicit_count)};
    }
    return OriginZeroLine{0};
}

inline Line<OriginZeroLine> resolve_placement(const Line<GridPlacement>& line, uint16_t explicit_count, uint16_t& out_span) {
    bool start_def = line.start.is_definite();
    bool end_def = line.end.is_definite();
    uint16_t span = 1;
    if (line.start.kind == GridPlacement::SpanKind) span = line.start.span_val;
    else if (line.end.kind == GridPlacement::SpanKind) span = line.end.span_val;

    if (start_def && end_def) {
        auto oz_s = placement_to_oz(line.start, explicit_count);
        auto oz_e = placement_to_oz(line.end, explicit_count);
        if (oz_s.value > oz_e.value) std::swap(oz_s, oz_e);
        if (oz_s.value == oz_e.value) oz_e.value = oz_s.value + 1;
        out_span = static_cast<uint16_t>(std::max<int16_t>(oz_e.value - oz_s.value, 1));
        return {oz_s, oz_e};
    } else if (start_def) {
        auto oz_s = placement_to_oz(line.start, explicit_count);
        out_span = span;
        return {oz_s, OriginZeroLine{static_cast<int16_t>(oz_s.value + span)}};
    } else if (end_def) {
        auto oz_e = placement_to_oz(line.end, explicit_count);
        out_span = span;
        return {OriginZeroLine{static_cast<int16_t>(oz_e.value - span)}, oz_e};
    }
    out_span = span;
    return {OriginZeroLine{0}, OriginZeroLine{0}};
}

// --- Track sizing algorithm ---
inline void initialize_track_sizes(std::vector<GridTrack>& tracks, std::optional<float> axis_inner_size) {
    for (auto& track : tracks) {
        if (track.is_collapsed) continue;
        auto resolve_cl = [&](CompactLength cl) -> float {
            if (cl.tag() == CompactLength::LENGTH_TAG) return cl.value();
            if (cl.tag() == CompactLength::PERCENT_TAG && axis_inner_size) return *axis_inner_size * cl.value();
            return 0.0f;
        };
        track.base_size = resolve_cl(track.min_track_sizing_function);
        if (track.max_track_sizing_function.is_fr() || track.max_track_sizing_function.is_max_content_alike())
            track.growth_limit = std::numeric_limits<float>::infinity();
        else
            track.growth_limit = resolve_cl(track.max_track_sizing_function);
        if (track.growth_limit < track.base_size) track.growth_limit = track.base_size;
    }
}

inline void maximise_tracks(std::vector<GridTrack>& tracks, std::optional<float> axis_inner_size) {
    if (!axis_inner_size) return;
    float free_space = *axis_inner_size;
    for (auto& t : tracks) free_space -= t.base_size;
    if (free_space <= 0) return;
    bool changed = true;
    while (changed && free_space > 0.001f) {
        changed = false;
        size_t eligible = 0;
        for (auto& t : tracks)
            if (t.kind == GridTrackKind::Track && t.base_size < t.growth_limit && !t.is_collapsed) eligible++;
        if (eligible == 0) break;
        float share = free_space / eligible;
        for (auto& t : tracks) {
            if (t.kind != GridTrackKind::Track || t.is_collapsed || t.base_size >= t.growth_limit) continue;
            float ns = std::min(t.base_size + share, t.growth_limit);
            float added = ns - t.base_size;
            t.base_size = ns;
            free_space -= added;
            if (added > 0.001f) changed = true;
        }
    }
}

inline void expand_flexible_tracks(std::vector<GridTrack>& tracks, std::optional<float> axis_inner_size) {
    if (!axis_inner_size) return;
    float used = 0, total_fr = 0;
    for (auto& t : tracks) {
        if (t.is_collapsed) continue;
        if (t.is_flexible()) total_fr += t.flex_factor();
        else used += t.base_size;
    }
    if (total_fr <= 0) return;
    float free_space = *axis_inner_size - used;
    if (free_space <= 0) return;
    float fr_size = free_space / total_fr;
    for (auto& t : tracks) {
        if (t.is_flexible() && !t.is_collapsed) t.base_size = t.flex_factor() * fr_size;
    }
}

inline void compute_track_offsets(std::vector<GridTrack>& tracks) {
    float offset = 0;
    for (auto& t : tracks) { t.offset = offset; offset += t.base_size; }
}

inline void resolve_item_track_indexes(std::vector<GridItem>& items, TrackCounts col_counts, TrackCounts row_counts) {
    for (auto& item : items) {
        auto [cs, ce] = col_counts.oz_line_range_to_track_range(item.column);
        auto [rs, re] = row_counts.oz_line_range_to_track_range(item.row);
        item.column_indexes = {static_cast<uint16_t>(cs), static_cast<uint16_t>(ce)};
        item.row_indexes = {static_cast<uint16_t>(rs), static_cast<uint16_t>(re)};
    }
}

inline void track_sizing_algorithm(std::vector<GridTrack>& tracks, std::optional<float> axis_inner_size, bool has_flex) {
    initialize_track_sizes(tracks, axis_inner_size);
    if (!has_flex) maximise_tracks(tracks, axis_inner_size);
    else expand_flexible_tracks(tracks, axis_inner_size);
    compute_track_offsets(tracks);
}

inline void align_tracks(std::vector<GridTrack>& tracks, AlignContent alignment, float free_space, size_t num_tracks) {
    if (free_space <= 0 || num_tracks == 0) return;
    auto kw = apply_alignment_fallback(free_space, num_tracks, alignment);
    float off = 0;
    switch (kw) {
        case AlignContentKeyword::End: case AlignContentKeyword::FlexEnd: off = free_space; break;
        case AlignContentKeyword::Center: off = free_space / 2; break;
        default: off = 0; break;
    }
    for (auto& t : tracks) t.offset += off;
}

inline void auto_place_grid_items(
    CellOccupancyMatrix& matrix, std::vector<GridItem>& items,
    GridAutoFlow auto_flow, uint16_t explicit_col_count) {
    bool dense = is_dense(auto_flow);
    AbsoluteAxis primary = primary_axis_from_flow(auto_flow);
    int16_t cursor_col = 0, cursor_row = 0;

    for (auto& item : items) {
        if (item.placed) continue;
        uint16_t cs = static_cast<uint16_t>(std::max<int16_t>(item.column.end.value - item.column.start.value, 1));
        uint16_t rs = static_cast<uint16_t>(std::max<int16_t>(item.row.end.value - item.row.start.value, 1));

        auto try_place = [&](int16_t col_start, int16_t row_start) -> bool {
            Line<OriginZeroLine> col_span{OriginZeroLine{col_start}, OriginZeroLine{static_cast<int16_t>(col_start + cs)}};
            Line<OriginZeroLine> row_span{OriginZeroLine{row_start}, OriginZeroLine{static_cast<int16_t>(row_start + rs)}};
            if (matrix.line_area_is_unoccupied(AbsoluteAxis::Horizontal, col_span, row_span)) {
                item.column = col_span;
                item.row = row_span;
                item.placed = true;
                matrix.mark_area_as(AbsoluteAxis::Horizontal, col_span, row_span, CellOccupancyState::AutoPlaced);
                return true;
            }
            return false;
        };

        if (primary == AbsoluteAxis::Horizontal) {
            while (true) {
                if (cursor_col + cs <= explicit_col_count || explicit_col_count == 0) {
                    if (try_place(cursor_col, cursor_row)) {
                        cursor_col += cs;
                        break;
                    }
                }
                cursor_col++;
                if (cursor_col + cs > explicit_col_count && explicit_col_count > 0) {
                    cursor_col = 0;
                    cursor_row++;
                }
            }
        } else {
            while (true) {
                if (try_place(cursor_col, cursor_row)) {
                    cursor_row += rs;
                    break;
                }
                cursor_row++;
                if (explicit_col_count > 0 && cursor_col >= explicit_col_count) {
                    cursor_col = 0;
                }
            }
        }
    }
}

} // namespace grid_detail

#endif // TAFFY_GRID

// ===========================================================================
// Section 16: TaffyTree (Arena-based high-level API)
// Ref: src/tree/taffy_tree.rs
// ===========================================================================

namespace detail {
    struct NodeData {
        Style style{};
        Layout unrounded_layout{};
        Layout final_layout{};
        Cache cache{};
        bool has_context{false};
    };
    struct NodeSlot {
        uint32_t generation{0};
        NodeData data{};
        bool alive{false};
        uint32_t next_free{UINT32_MAX};
    };
} // namespace detail

enum class TaffyError : uint8_t {
    ChildIndexOutOfBounds,
    InvalidParentNode,
    InvalidChildNode,
    InvalidInputNode,
};

template <typename NodeContext = void>
class TaffyTree {
public:
    TaffyTree() { nodes_.reserve(16); }
    explicit TaffyTree(size_t capacity) { nodes_.reserve(capacity); }

    void enable_rounding() { use_rounding_ = true; }
    void disable_rounding() { use_rounding_ = false; }

    // Node creation
    NodeId new_leaf(const Style& style) {
        NodeId id = alloc_slot();
        nodes_[id.index].data.style = style;
        children_[id.index] = {};
        parents_[id.index] = NODE_ID_NONE;
        if constexpr (!std::is_void_v<NodeContext>) {
            contexts_[id.index] = NodeContext{};
        }
        return id;
    }

    template <typename NC = NodeContext, typename = std::enable_if_t<!std::is_void_v<NC>>>
    NodeId new_leaf_with_context(const Style& style, NC context) {
        NodeId id = new_leaf(style);
        contexts_[id.index] = std::move(context);
        nodes_[id.index].data.has_context = true;
        return id;
    }

    NodeId new_with_children(const Style& style, const std::vector<NodeId>& child_ids) {
        NodeId id = new_leaf(style);
        for (NodeId child : child_ids) {
            add_child(id, child);
        }
        return id;
    }

    // Tree modification
    void add_child(NodeId parent, NodeId child) {
        remove_from_parent(child);
        children_[parent.index].push_back(child);
        parents_[child.index] = parent;
        mark_dirty_recursive(parent);
    }

    void set_children(NodeId parent, const std::vector<NodeId>& new_children) {
        // Remove existing children
        for (NodeId old_child : children_[parent.index]) {
            parents_[old_child.index] = NODE_ID_NONE;
        }
        children_[parent.index].clear();
        // Reparent new children
        for (NodeId child : new_children) {
            remove_from_parent(child);
            children_[parent.index].push_back(child);
            parents_[child.index] = parent;
        }
        mark_dirty_recursive(parent);
    }

    void remove(NodeId node) {
        // Detach from parent
        remove_from_parent(node);
        // Detach children
        for (NodeId child : children_[node.index]) {
            parents_[child.index] = NODE_ID_NONE;
        }
        children_[node.index].clear();
        // Free slot
        free_slot(node);
    }

    // Style access
    void set_style(NodeId node, const Style& style) {
        nodes_[node.index].data.style = style;
        mark_dirty_recursive(node);
    }
    const Style& style(NodeId node) const { return nodes_[node.index].data.style; }

    // Layout access
    const Layout& layout(NodeId node) const {
        return use_rounding_ ? nodes_[node.index].data.final_layout : nodes_[node.index].data.unrounded_layout;
    }
    const Layout& unrounded_layout(NodeId node) const { return nodes_[node.index].data.unrounded_layout; }

    // Parent/children
    NodeId parent(NodeId child) const { return parents_[child.index]; }
    std::vector<NodeId> children(NodeId parent) const { return children_[parent.index]; }
    size_t child_count(NodeId parent) const { return children_[parent.index].size(); }
    NodeId get_child_id(NodeId parent, size_t index) const { return children_[parent.index][index]; }
    size_t total_node_count() const { return alive_count_; }

    // Dirty tracking
    void mark_dirty(NodeId node) { mark_dirty_recursive(node); }
    bool dirty(NodeId node) const { return nodes_[node.index].data.cache.is_empty(); }

    // Context access (only when NodeContext is not void)
    template <typename NC = NodeContext, typename = std::enable_if_t<!std::is_void_v<NC>>>
    NC* get_node_context(NodeId node) {
        return nodes_[node.index].data.has_context ? &contexts_[node.index] : nullptr;
    }

    // Main layout computation
    template <typename MeasureFunction>
    void compute_layout_with_measure(NodeId root, Size<AvailableSpace> available_space, MeasureFunction&& measure_fn) {
        // Compute root layout
        compute_root_layout_impl(root, available_space, std::forward<MeasureFunction>(measure_fn));
        // Round if enabled
        if (use_rounding_) {
            round_layout_impl(root, 0.0f, 0.0f);
        }
    }

    void compute_layout(NodeId root, Size<AvailableSpace> available_space) {
        compute_layout_with_measure(root, available_space,
            [](Size<std::optional<float>>, Size<AvailableSpace>, NodeId, const Style&) -> Size<float> {
                return {0.0f, 0.0f};
            });
    }

    // Debug


private:
    std::vector<detail::NodeSlot> nodes_{};
    std::vector<std::vector<NodeId>> children_{};
    std::vector<NodeId> parents_{};
    std::conditional_t<!std::is_void_v<NodeContext>, std::vector<NodeContext>, char> contexts_{};
    uint32_t free_head_{UINT32_MAX};
    size_t alive_count_{0};
    bool use_rounding_{true};

    NodeId alloc_slot() {
        uint32_t idx;
        if (free_head_ != UINT32_MAX) {
            idx = free_head_;
            free_head_ = nodes_[idx].next_free;
            nodes_[idx].generation++;
            nodes_[idx].alive = true;
            nodes_[idx].data = detail::NodeData{};
        } else {
            idx = static_cast<uint32_t>(nodes_.size());
            nodes_.push_back({0, detail::NodeData{}, true, UINT32_MAX});
            children_.push_back({});
            parents_.push_back(NODE_ID_NONE);
            if constexpr (!std::is_void_v<NodeContext>) {
                contexts_.push_back(NodeContext{});
            }
        }
        alive_count_++;
        return {idx, nodes_[idx].generation};
    }

    void free_slot(NodeId id) {
        nodes_[id.index].alive = false;
        nodes_[id.index].next_free = free_head_;
        free_head_ = id.index;
        alive_count_--;
    }

    void remove_from_parent(NodeId child) {
        NodeId p = parents_[child.index];
        if (p.is_valid()) {
            auto& pc = children_[p.index];
            pc.erase(std::remove(pc.begin(), pc.end(), child), pc.end());
            parents_[child.index] = NODE_ID_NONE;
        }
    }

    void mark_dirty_recursive(NodeId node) {
        while (node.is_valid()) {
            auto state = nodes_[node.index].data.cache.clear();
            if (state == ClearState::AlreadyEmpty) break;
            node = parents_[node.index];
        }
    }

    // Layout dispatch (public for use by algorithm template functions)
public:
    void set_unrounded_layout(NodeId node_id, const Layout& layout) {
        nodes_[node_id.index].data.unrounded_layout = layout;
    }

    template <typename MeasureFunction>
    LayoutOutput compute_node_layout(NodeId node_id, LayoutInput inputs, MeasureFunction&& measure_fn) {
        const Style& node_style = nodes_[node_id.index].data.style;
        auto& cache = nodes_[node_id.index].data.cache;
        auto cached = cache.get(inputs);
        if (cached) return *cached;

        bool has_children = !children_[node_id.index].empty();
        LayoutOutput output;

        if (inputs.run_mode == RunMode::PerformHiddenLayout) {
            output = LayoutOutput::hidden();
            Layout zero_layout = Layout::with_order(0);
            nodes_[node_id.index].data.unrounded_layout = zero_layout;
            for (NodeId child : children_[node_id.index]) {
                LayoutInput hidden_input;
                hidden_input.run_mode = RunMode::PerformHiddenLayout;
                compute_node_layout(child, hidden_input, measure_fn);
            }
        } else if (!has_children) {
            // Leaf node
            output = compute_leaf_layout_impl(inputs, node_id, node_style, measure_fn);
        } else {
#ifdef TAFFY_FLEXBOX
            if (node_style.display == Display::Flex) {
                output = compute_flexbox_layout_impl(node_id, inputs, measure_fn);
            } else
#endif
#ifdef TAFFY_GRID
            if (node_style.display == Display::Grid) {
                output = compute_grid_layout_impl(node_id, inputs, measure_fn);
            } else
#endif
#ifdef TAFFY_BLOCK_LAYOUT
            if (node_style.display == Display::Block) {
                output = compute_block_layout_impl(node_id, inputs, measure_fn);
            } else
#endif
            {
                output = LayoutOutput::hidden();
            }
        }

        cache.store(inputs, output);
        return output;
    }

    template <typename MeasureFunction>
    void compute_root_layout_impl(NodeId root, Size<AvailableSpace> available_space, MeasureFunction&& measure_fn) {
        const Style& root_style = nodes_[root.index].data.style;
        // The root's "parent" is the viewport: resolve percent sizes (e.g. 100%)
        // against the definite available space so the root fills the viewport.
        Size<std::optional<float>> parent_size = {
            available_space.width.into_option(),
            available_space.height.into_option()
        };
        Size<std::optional<float>> known_dimensions = maybe_resolve_size(root_style.size, parent_size);

        LayoutInput inputs;
        inputs.run_mode = RunMode::PerformLayout;
        inputs.sizing_mode = SizingMode::InherentSize;
        inputs.axis = RequestedAxis::Both;
        inputs.known_dimensions = known_dimensions;
        inputs.parent_size = parent_size;
        inputs.available_space = available_space;
        inputs.vertical_margins_are_collapsible = Line_bool_FALSE;

        LayoutOutput output = compute_node_layout(root, inputs, measure_fn);

        auto margin = resolve_or_zero_rect(root_style.margin, std::nullopt);
        auto border = resolve_or_zero_rect(root_style.border, std::nullopt);
        auto padding = resolve_or_zero_rect(root_style.padding, std::nullopt);

        float loc_x = margin.left;
        if (root_style.direction == Direction::Rtl) {
            float aw = available_space.width.is_definite() ? available_space.width.unwrap() : output.size.width;
            loc_x = aw - output.size.width - margin.right;
        }

        Layout layout_result;
        layout_result.order = 0;
        layout_result.location = {loc_x, margin.top};
        layout_result.size = output.size;
#ifdef TAFFY_CONTENT_SIZE
        layout_result.content_size = output.content_size;
#endif
        layout_result.scrollbar_size = {0.0f, 0.0f};
        layout_result.border = border;
        layout_result.padding = padding;
        layout_result.margin = margin;

        nodes_[root.index].data.unrounded_layout = layout_result;
    }

    void round_layout_impl(NodeId node, float cumulative_x, float cumulative_y) {
        auto& unrounded = nodes_[node.index].data.unrounded_layout;
        Layout final_layout;
        final_layout.order = unrounded.order;

        float new_x = detail::f32_round(unrounded.location.x + cumulative_x);
        float new_y = detail::f32_round(unrounded.location.y + cumulative_y);
        final_layout.location = {new_x - detail::f32_round(cumulative_x), new_y - detail::f32_round(cumulative_y)};

        float right = detail::f32_round(unrounded.location.x + cumulative_x + unrounded.size.width);
        float bottom = detail::f32_round(unrounded.location.y + cumulative_y + unrounded.size.height);
        final_layout.size = {right - new_x, bottom - new_y};
#ifdef TAFFY_CONTENT_SIZE
        final_layout.content_size = unrounded.content_size;
#endif
        final_layout.scrollbar_size = unrounded.scrollbar_size;
        final_layout.border = unrounded.border;
        final_layout.padding = unrounded.padding;
        final_layout.margin = unrounded.margin;

        nodes_[node.index].data.final_layout = final_layout;

        float child_x = new_x + final_layout.border.left + final_layout.padding.left;
        float child_y = new_y + final_layout.border.top + final_layout.padding.top;
        for (NodeId child : children_[node.index]) {
            round_layout_impl(child, child_x, child_y);
        }
    }

    template <typename MeasureFunction>
    LayoutOutput compute_leaf_layout_impl(LayoutInput inputs, NodeId node_id, const Style& style, MeasureFunction& measure_fn) {
        auto parent_size = inputs.parent_size;
        auto margin = resolve_or_zero_rect(style.margin, parent_size.width);
        auto border = resolve_or_zero_rect(style.border, parent_size.width);
        auto padding = resolve_or_zero_rect(style.padding, parent_size.width);

        Size<float> padding_border{
            padding.left + padding.right + border.left + border.right,
            padding.top + padding.bottom + border.top + border.bottom
        };

        Size<std::optional<float>> node_size;
        Size<std::optional<float>> node_min_size;
        Size<std::optional<float>> node_max_size;

        if (inputs.sizing_mode == SizingMode::InherentSize) {
            node_size = maybe_resolve_size(style.size, parent_size);
            node_min_size = maybe_resolve_size(style.min_size, parent_size);
            node_max_size = maybe_resolve_size(style.max_size, parent_size);
        }

        Size<AvailableSpace> child_available_space{
            AvailableSpace::definite(std::max(0.0f,
                (inputs.available_space.width.is_definite() ? inputs.available_space.width.unwrap() : 0.0f)
                - margin.left - margin.right - padding_border.width)),
            AvailableSpace::definite(std::max(0.0f,
                (inputs.available_space.height.is_definite() ? inputs.available_space.height.unwrap() : 0.0f)
                - margin.top - margin.bottom - padding_border.height))
        };

        if (inputs.available_space.width.type == AvailableSpace::MinContent) child_available_space.width = AvailableSpace::min_content();
        if (inputs.available_space.width.type == AvailableSpace::MaxContent) child_available_space.width = AvailableSpace::max_content();
        if (inputs.available_space.height.type == AvailableSpace::MinContent) child_available_space.height = AvailableSpace::min_content();
        if (inputs.available_space.height.type == AvailableSpace::MaxContent) child_available_space.height = AvailableSpace::max_content();

        Size<float> measured = measure_fn(inputs.known_dimensions, child_available_space, node_id, style);

        Size<float> final_size{
            inputs.known_dimensions.width.value_or(node_size.width.value_or(measured.width + padding_border.width)),
            inputs.known_dimensions.height.value_or(node_size.height.value_or(measured.height + padding_border.height))
        };

        if (node_min_size.width.has_value()) final_size.width = std::max(final_size.width, *node_min_size.width);
        if (node_min_size.height.has_value()) final_size.height = std::max(final_size.height, *node_min_size.height);
        if (node_max_size.width.has_value()) final_size.width = std::min(final_size.width, *node_max_size.width);
        if (node_max_size.height.has_value()) final_size.height = std::min(final_size.height, *node_max_size.height);

        Size<float> content_size{measured.width + padding.left + padding.right, measured.height + padding.top + padding.bottom};
        return LayoutOutput::from_sizes(final_size, content_size);
    }

    // Flexbox layout implementation
    template <typename MeasureFunction>
    LayoutOutput compute_flexbox_layout_impl(NodeId node_id, LayoutInput inputs, MeasureFunction& measure_fn) {
        const Style& style = nodes_[node_id.index].data.style;
        auto known = inputs.known_dimensions;
        auto parent_size = inputs.parent_size;
        auto padding = resolve_or_zero_rect(style.padding, parent_size.width);
        auto border = resolve_or_zero_rect(style.border, parent_size.width);
        auto pb_sum = padding.sum_axes() + border.sum_axes();
        auto box_adj = (style.box_sizing == BoxSizing::ContentBox) ? pb_sum : Size_f32_ZERO;
        auto min_sz = maybe_resolve_size(style.min_size, parent_size);
        auto max_sz = maybe_resolve_size(style.max_size, parent_size);
        auto clamped_style = inputs.sizing_mode == SizingMode::InherentSize
            ? maybe_resolve_size(style.size, parent_size) : Size<std::optional<float>>{};
        auto styled_known = known;
        if (!styled_known.width) styled_known.width = clamped_style.width;
        if (!styled_known.height) styled_known.height = clamped_style.height;
        if (!styled_known.width && min_sz.width && max_sz.width && *max_sz.width <= *min_sz.width)
            styled_known.width = min_sz.width;
        if (!styled_known.height && min_sz.height && max_sz.height && *max_sz.height <= *min_sz.height)
            styled_known.height = min_sz.height;
        if (styled_known.width) styled_known.width = std::max(*styled_known.width, pb_sum.width);
        if (styled_known.height) styled_known.height = std::max(*styled_known.height, pb_sum.height);

        if (inputs.run_mode == RunMode::ComputeSize && styled_known.width && styled_known.height)
            return LayoutOutput::from_outer_size({*styled_known.width, *styled_known.height});

        auto c = flex_compute_constants(*this, node_id, style, styled_known, parent_size);
        auto all_items = flex_generate_items(*this, node_id, c);

        auto available = determine_available_space_fn(styled_known, inputs.available_space, c);
        flex_determine_base_size(*this, c, available, all_items, measure_fn);
        auto lines = flex_collect_lines(c, available, all_items);

        // Determine container main size if unknown
        if (!c.node_inner_size.main(c.dir)) {
            float outer_main = 0;
            if (available.main(c.dir).is_definite()) {
                float max_line = 0;
                for (auto& line : lines) {
                    float line_total = sum_axis_gaps(c.gap.main(c.dir), line.items.size());
                    for (auto& item : line.items) {
                        float pb = item.padding.main_axis_sum(c.dir) + item.border.main_axis_sum(c.dir);
                        line_total += std::max(item.flex_basis, std::max(item.min_size.main(c.dir).value_or(0.0f), pb))
                                      + item.margin.main_axis_sum(c.dir);
                    }
                    max_line = std::max(max_line, line_total);
                }
                float main_inset = c.content_box_inset.main_axis_sum(c.dir);
                outer_main = lines.size() > 1 ? std::max(max_line + main_inset, available.main(c.dir).value)
                                              : max_line + main_inset;
            } else {
                float max_line = 0;
                for (auto& line : lines) {
                    float line_total = sum_axis_gaps(c.gap.main(c.dir), line.items.size());
                    for (auto& item : line.items)
                        line_total += item.flex_basis + item.margin.main_axis_sum(c.dir);
                    max_line = std::max(max_line, line_total);
                }
                outer_main = max_line + c.content_box_inset.main_axis_sum(c.dir);
            }
            if (c.min_size.main(c.dir)) outer_main = std::max(outer_main, *c.min_size.main(c.dir));
            if (c.max_size.main(c.dir)) outer_main = std::min(outer_main, *c.max_size.main(c.dir));
            float main_inset = c.content_box_inset.main_axis_sum(c.dir);
            outer_main = std::max(outer_main, main_inset - c.scrollbar_gutter.main(c.dir));
            float inner_main = std::max(outer_main - main_inset, 0.0f);
            c.container_size.set_main(c.dir, outer_main);
            c.inner_container_size.set_main(c.dir, inner_main);
            c.node_inner_size.set_main(c.dir, inner_main);
            c.node_outer_size.set_main(c.dir, outer_main);
            // Re-resolve gap
            auto new_gap = c.node_inner_size.width ? std::optional(resolve_or_zero(LengthPercentage::length(c.gap.width), c.node_inner_size.width)) : std::optional<float>(c.gap.width);
            c.gap.set_main(c.dir, new_gap.value_or(0.0f));
        } else {
            c.inner_container_size.set_main(c.dir, *c.node_inner_size.main(c.dir));
            c.container_size.set_main(c.dir, *c.node_inner_size.main(c.dir) + c.content_box_inset.main_axis_sum(c.dir));
        }

        for (auto& line : lines) flex_resolve_flexible_lengths(line, c);
        for (auto& line : lines) flex_determine_hypothetical_cross(*this, line, c, available, measure_fn);
        flex_calculate_cross_size(lines, c.node_outer_size, c);
        flex_handle_align_content_stretch(lines, c.node_outer_size, c);
        flex_determine_used_cross_size(lines, c);
        flex_distribute_remaining_free_space(lines, c);
        flex_resolve_cross_axis_auto_margins(lines, c);
        float total_cross = flex_determine_container_cross_size(lines, c.node_outer_size, c);

        if (inputs.run_mode == RunMode::ComputeSize)
            return LayoutOutput::from_outer_size(c.container_size);

        flex_align_lines_per_align_content(lines, c, total_cross);
        Size<float> content_size = Size_f32_ZERO;
        flex_final_layout_pass(*this, lines, c, measure_fn, content_size);

        // Hidden children
        for (size_t i = 0; i < children_[node_id.index].size(); ++i) {
            NodeId child = children_[node_id.index][i];
            const auto& cs = nodes_[child.index].data.style;
            if (cs.display == Display::None) {
                set_unrounded_layout(child, Layout::with_order(i));
                LayoutInput hi; hi.run_mode = RunMode::PerformHiddenLayout;
                compute_node_layout(child, hi, measure_fn);
            }
        }

        // First baseline
        std::optional<float> first_baseline;
        if (!lines.empty() && !lines[0].items.empty()) {
            auto& first = lines[0].items[0];
            float offset_v = c.is_row ? first.offset_cross : first.offset_main;
            first_baseline = offset_v + first.baseline;
        }

        return LayoutOutput::from_sizes_and_baselines(c.container_size, content_size,
            {std::nullopt, first_baseline});
    }

    static Size<AvailableSpace> determine_available_space_fn(
        Size<std::optional<float>> known, Size<AvailableSpace> outer, const FlexAlgoConstants& c) {
        auto width = known.width
            ? AvailableSpace::definite(*known.width - c.content_box_inset.horizontal_axis_sum())
            : maybe_sub(maybe_sub(outer.width, c.margin.horizontal_axis_sum()), c.content_box_inset.horizontal_axis_sum());
        auto height = known.height
            ? AvailableSpace::definite(*known.height - c.content_box_inset.vertical_axis_sum())
            : maybe_sub(maybe_sub(outer.height, c.margin.vertical_axis_sum()), c.content_box_inset.vertical_axis_sum());
        return {width, height};
    }
    // Grid layout implementation placeholder
    template <typename MeasureFunction>
    LayoutOutput compute_grid_layout_impl(NodeId node_id, LayoutInput inputs, MeasureFunction& measure_fn) {
        using namespace grid_detail;
        const Style& style = nodes_[node_id.index].data.style;
        auto parent_size = inputs.parent_size;
        auto padding = resolve_or_zero_rect(style.padding, parent_size.width);
        auto border = resolve_or_zero_rect(style.border, parent_size.width);
        auto pb_sum = padding.sum_axes() + border.sum_axes();

        // Resolve container size
        auto style_size = maybe_resolve_size(style.size, parent_size);
        Size<std::optional<float>> known = inputs.known_dimensions;
        if (!known.width && style_size.width) known.width = style_size.width;
        if (!known.height && style_size.height) known.height = style_size.height;
        Size<std::optional<float>> node_inner = {
            known.width ? std::optional(*known.width - pb_sum.width) : std::nullopt,
            known.height ? std::optional(*known.height - pb_sum.height) : std::nullopt
        };

        // Compute explicit grid
        auto [auto_col_reps, explicit_col_count] = compute_explicit_grid_size_in_axis(
            style, node_inner.width, AutoRepeatStrategy::MaxRepetitionsThatDoNotOverflow, AbsoluteAxis::Horizontal);
        auto [auto_row_reps, explicit_row_count] = compute_explicit_grid_size_in_axis(
            style, node_inner.height, AutoRepeatStrategy::MaxRepetitionsThatDoNotOverflow, AbsoluteAxis::Vertical);

        // Estimate implicit grid from children
        std::vector<Style> child_styles;
        for (size_t i = 0; i < children_[node_id.index].size(); ++i) {
            child_styles.push_back(nodes_[children_[node_id.index][i].index].data.style);
        }
        auto [col_counts, row_counts] = compute_grid_size_estimate(
            explicit_col_count, explicit_row_count, style.direction, child_styles);

        // Create grid items
        std::vector<GridItem> items;
        for (size_t i = 0; i < children_[node_id.index].size(); ++i) {
            NodeId child_id = children_[node_id.index][i];
            const auto& cs = nodes_[child_id.index].data.style;
            if (cs.position == Position::Absolute) continue;

            // Resolve placement
            uint16_t col_span = 1, row_span = 1;
            auto col_oz = resolve_placement(cs.grid_column, explicit_col_count, col_span);
            auto row_oz = resolve_placement(cs.grid_row, explicit_row_count, row_span);
            bool col_def = cs.grid_column.start.is_definite() || cs.grid_column.end.is_definite();
            bool row_def = cs.grid_row.start.is_definite() || cs.grid_row.end.is_definite();

            GridItem item;
            item.node = child_id;
            item.source_order = static_cast<uint16_t>(i);
            item.column = col_def ? col_oz : Line<OriginZeroLine>{OriginZeroLine{0}, OriginZeroLine{static_cast<int16_t>(col_span)}};
            item.row = row_def ? row_oz : Line<OriginZeroLine>{OriginZeroLine{0}, OriginZeroLine{static_cast<int16_t>(row_span)}};
            item.placed = col_def && row_def;
            item.size = cs.size;
            item.min_size = cs.min_size;
            item.max_size = cs.max_size;
            item.aspect_ratio = cs.aspect_ratio;
            item.padding = cs.padding;
            item.border = cs.border;
            item.margin = cs.margin;
            item.align_self = cs.align_self.value_or(style.align_items.value_or(AlignItems_STRETCH));
            item.justify_self = cs.justify_self.value_or(style.justify_items.value_or(AlignItems_STRETCH));
            item.overflow = cs.overflow;
            item.box_sizing = cs.box_sizing;
            items.push_back(item);
        }

        // Place items
        auto matrix = CellOccupancyMatrix::with_track_counts(col_counts, row_counts);
        // Mark definitely placed items in matrix
        for (auto& item : items) {
            if (item.placed)
                matrix.mark_area_as(AbsoluteAxis::Horizontal, item.column, item.row, CellOccupancyState::DefinitelyPlaced);
        }
        auto_place_grid_items(matrix, items, style.grid_auto_flow, explicit_col_count);

        // Update counts based on placed items
        TrackCounts final_cols = col_counts, final_rows = row_counts;
        for (auto& item : items) {
            int16_t col_end = item.column.end.value;
            int16_t row_end = item.row.end.value;
            if (col_end > (int16_t)(final_cols.explicit_ + final_cols.positive_implicit))
                final_cols.positive_implicit = col_end - final_cols.explicit_;
            if (row_end > (int16_t)(final_rows.explicit_ + final_rows.positive_implicit))
                final_rows.positive_implicit = row_end - final_rows.explicit_;
        }

        // Resolve item track indexes
        resolve_item_track_indexes(items, final_cols, final_rows);

        // Initialize tracks
        std::vector<GridTrack> columns, rows;
        auto track_has_items_col = [&](size_t idx) -> bool {
            for (auto& item : items)
                if (item.column_indexes.start <= idx && idx < item.column_indexes.end) return true;
            return false;
        };
        auto track_has_items_row = [&](size_t idx) -> bool {
            for (auto& item : items)
                if (item.row_indexes.start <= idx && idx < item.row_indexes.end) return true;
            return false;
        };
        initialize_grid_tracks(columns, final_cols, style, AbsoluteAxis::Horizontal, track_has_items_col);
        initialize_grid_tracks(rows, final_rows, style, AbsoluteAxis::Vertical, track_has_items_row);

        // Check for flex tracks
        bool has_flex_col = false, has_flex_row = false;
        for (auto& t : columns) if (t.is_flexible()) has_flex_col = true;
        for (auto& t : rows) if (t.is_flexible()) has_flex_row = true;

        // Run track sizing
        track_sizing_algorithm(columns, node_inner.width, has_flex_col);
        track_sizing_algorithm(rows, node_inner.height, has_flex_row);

        // Compute container size
        float total_col_size = 0;
        for (auto& t : columns) if (t.kind == GridTrackKind::Track) total_col_size += t.base_size;
        float total_row_size = 0;
        for (auto& t : rows) if (t.kind == GridTrackKind::Track) total_row_size += t.base_size;

        float container_w = known.width.value_or(total_col_size + pb_sum.width);
        float container_h = known.height.value_or(total_row_size + pb_sum.height);

        // Align tracks
        float free_col = container_w - pb_sum.width - total_col_size;
        float free_row = container_h - pb_sum.height - total_row_size;
        size_t num_col_tracks = 0, num_row_tracks = 0;
        for (auto& t : columns) if (t.kind == GridTrackKind::Track) num_col_tracks++;
        for (auto& t : rows) if (t.kind == GridTrackKind::Track) num_row_tracks++;

        align_tracks(columns, style.justify_content.value_or(AlignContent_START), free_col, num_col_tracks);
        align_tracks(rows, style.align_content.value_or(AlignContent_START), free_row, num_row_tracks);

        // Position items
        for (auto& item : items) {
            float x = padding.left + border.left;
            float y = padding.top + border.top;
            float w = 0, h = 0;

            // Compute item grid area from track offsets
            if (item.column_indexes.start < columns.size() && item.column_indexes.end <= columns.size()) {
                x += columns[item.column_indexes.start].offset;
                for (uint16_t i = item.column_indexes.start; i < item.column_indexes.end; ++i)
                    w += columns[i].base_size;
            }
            if (item.row_indexes.start < rows.size() && item.row_indexes.end <= rows.size()) {
                y += rows[item.row_indexes.start].offset;
                for (uint16_t i = item.row_indexes.start; i < item.row_indexes.end; ++i)
                    h += rows[i].base_size;
            }

            // Subtract margins/padding/border from available space for child
            auto margin = resolve_or_zero_rect(item.margin, container_w - pb_sum.width);
            auto child_pb = resolve_or_zero_rect(item.padding, w);
            auto child_border = resolve_or_zero_rect(item.border, w);
            Size<float> child_size{w - margin.left - margin.right, h - margin.top - margin.bottom};

            LayoutInput child_inputs;
            child_inputs.run_mode = RunMode::PerformLayout;
            child_inputs.sizing_mode = SizingMode::InherentSize;
            child_inputs.known_dimensions = {child_size.width, child_size.height};
            child_inputs.parent_size = {container_w - pb_sum.width, container_h - pb_sum.height};
            child_inputs.available_space = {AvailableSpace::definite(child_size.width), AvailableSpace::definite(child_size.height)};
            auto output = compute_node_layout(item.node, child_inputs, measure_fn);

            Layout layout_result;
            layout_result.order = item.source_order;
            layout_result.location = {x + margin.left, y + margin.top};
            layout_result.size = output.size;
#ifdef TAFFY_CONTENT_SIZE
            layout_result.content_size = output.content_size;
#endif
            layout_result.scrollbar_size = {0, 0};
            layout_result.padding = child_pb;
            layout_result.border = child_border;
            layout_result.margin = margin;
            set_unrounded_layout(item.node, layout_result);
        }

        // Hidden children
        for (size_t i = 0; i < children_[node_id.index].size(); ++i) {
            NodeId child_id = children_[node_id.index][i];
            if (nodes_[child_id.index].data.style.display == Display::None) {
                set_unrounded_layout(child_id, Layout::with_order(i));
                LayoutInput hi; hi.run_mode = RunMode::PerformHiddenLayout;
                compute_node_layout(child_id, hi, measure_fn);
            }
        }

        return LayoutOutput::from_outer_size({container_w, container_h});
    }
    // Block layout implementation
    template <typename MeasureFunction>
    LayoutOutput compute_block_layout_impl(NodeId node_id, LayoutInput inputs, MeasureFunction& measure_fn) {
        const Style& style = nodes_[node_id.index].data.style;
        auto parent_size = inputs.parent_size;
        auto padding = resolve_or_zero_rect(style.padding, parent_size.width);
        auto border = resolve_or_zero_rect(style.border, parent_size.width);
        auto pb_sum = padding.sum_axes() + border.sum_axes();
        auto margin = resolve_or_zero_rect(style.margin, parent_size.width);

        // Resolve container size
        auto style_size = maybe_resolve_size(style.size, parent_size);
        auto min_size = maybe_resolve_size(style.min_size, parent_size);
        auto max_size = maybe_resolve_size(style.max_size, parent_size);

        Size<std::optional<float>> known = inputs.known_dimensions;
        if (!known.width && style_size.width) known.width = style_size.width;
        if (!known.height && style_size.height) known.height = style_size.height;

        float container_content_width = known.width
            ? std::max(*known.width - pb_sum.width, 0.0f)
            : (inputs.available_space.width.is_definite()
                ? std::max(inputs.available_space.width.value - margin.left - margin.right - pb_sum.width, 0.0f)
                : 0.0f);

        // Position children vertically (block flow)
        float y_offset = 0;
        float prev_margin_bottom = 0;
        float max_child_width = 0;

        for (size_t i = 0; i < children_[node_id.index].size(); ++i) {
            NodeId child_id = children_[node_id.index][i];
            const auto& cs = nodes_[child_id.index].data.style;

            if (cs.display == Display::None) {
                set_unrounded_layout(child_id, Layout::with_order(static_cast<uint32_t>(i)));
                LayoutInput hi; hi.run_mode = RunMode::PerformHiddenLayout;
                compute_node_layout(child_id, hi, measure_fn);
                continue;
            }

            // Resolve child margin for collapsing
            auto child_margin = resolve_or_zero_rect(cs.margin, container_content_width);
            float collapsed_margin = std::max(prev_margin_bottom, child_margin.top);
            y_offset += collapsed_margin;

            // Child layout
            auto child_pb = resolve_or_zero_rect(cs.padding, container_content_width);
            auto child_border = resolve_or_zero_rect(cs.border, container_content_width);
            float child_content_width = container_content_width - child_pb.horizontal_axis_sum() - child_border.horizontal_axis_sum()
                                        - child_margin.left - child_margin.right;

            LayoutInput child_inputs;
            child_inputs.run_mode = inputs.run_mode;
            child_inputs.sizing_mode = SizingMode::InherentSize;
            child_inputs.known_dimensions = {child_content_width, std::nullopt};
            child_inputs.parent_size = {container_content_width, known.height};
            child_inputs.available_space = {
                AvailableSpace::definite(container_content_width),
                inputs.available_space.height
            };
            child_inputs.vertical_margins_are_collapsible = Line_bool_FALSE;

            auto output = compute_node_layout(child_id, child_inputs, measure_fn);

            Layout layout_result;
            layout_result.order = static_cast<uint32_t>(i);
            layout_result.location = {padding.left + border.left + child_margin.left, padding.top + border.top + y_offset};
            layout_result.size = output.size;
#ifdef TAFFY_CONTENT_SIZE
            layout_result.content_size = output.content_size;
#endif
            layout_result.scrollbar_size = {0, 0};
            layout_result.padding = child_pb;
            layout_result.border = child_border;
            layout_result.margin = child_margin;
            set_unrounded_layout(child_id, layout_result);

            y_offset += output.size.height;
            prev_margin_bottom = child_margin.bottom;
            max_child_width = std::max(max_child_width, output.size.width + child_margin.left + child_margin.right);
        }

        // Final container size
        float final_width = known.width.value_or(max_child_width + pb_sum.width);
        float final_height = known.height.value_or(y_offset + prev_margin_bottom + pb_sum.height);
        if (min_size.width) final_width = std::max(final_width, *min_size.width);
        if (min_size.height) final_height = std::max(final_height, *min_size.height);
        if (max_size.width) final_width = std::min(final_width, *max_size.width);
        if (max_size.height) final_height = std::min(final_height, *max_size.height);

#ifdef TAFFY_CONTENT_SIZE
        Size<float> content_size{max_child_width, y_offset + prev_margin_bottom};
        return LayoutOutput::from_sizes({final_width, final_height}, content_size);
#else
        return LayoutOutput::from_outer_size({final_width, final_height});
#endif
    }

    void print_tree(NodeId root) const {
        printf("TREE\n");
        print_tree_impl(root, false, "");
    }

    void print_tree_impl(NodeId node, bool has_sibling, const std::string& lines_prefix) const {
        const auto& l = use_rounding_ ? nodes_[node.index].data.final_layout : nodes_[node.index].data.unrounded_layout;
        const auto& s = nodes_[node.index].data.style;
        const char* label = "UNKNOWN";
        if (s.display == Display::None) label = "NONE";
        else if (children_[node.index].empty()) label = "LEAF";
        else if (s.display == Display::Flex) label = is_row(s.flex_direction) ? "FLEX ROW" : "FLEX COL";
        else if (s.display == Display::Grid) label = "GRID";
        else if (s.display == Display::Block) label = "BLOCK";

        const char* fork_str = has_sibling ? "\xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 " : "\xe2\x94\x94\xe2\x94\x80\xe2\x94\x80 ";
        printf("%s%s %s [x: %-6.1f y: %-6.1f w: %-6.1f h: %-6.1f border: l:%.1f r:%.1f t:%.1f b:%.1f padding: l:%.1f r:%.1f t:%.1f b:%.1f]\n",
            lines_prefix.c_str(), fork_str, label,
            l.location.x, l.location.y, l.size.width, l.size.height,
            l.border.left, l.border.right, l.border.top, l.border.bottom,
            l.padding.left, l.padding.right, l.padding.top, l.padding.bottom);

        const char* bar = has_sibling ? "\xe2\x94\x82   " : "    ";
        std::string new_prefix = lines_prefix + bar;
        size_t num_children = children_[node.index].size();
        for (size_t i = 0; i < num_children; ++i) {
            print_tree_impl(children_[node.index][i], i < num_children - 1, new_prefix);
        }
    }
};

} // namespace taffy

#endif // TAFFY_HPP
