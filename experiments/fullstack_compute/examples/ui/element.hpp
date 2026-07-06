// element.hpp — Element base class for the WCN UI library
// Requires taffy.hpp and fullstack_compute rendering headers.
// Standard: C++17
#pragma once
#ifndef WCN_UI_ELEMENT_HPP
#define WCN_UI_ELEMENT_HPP

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// ── Forward declare taffy types ──────────────────────────────────────────
namespace taffy {
struct Style;
struct Layout;
}

// ── Forward declare FS types ─────────────────────────────────────────────
struct FS_Core;
struct FS_ImageHandle;
struct FS_TextMetrics;

namespace wcn_ui {

// ── Color ────────────────────────────────────────────────────────────────
struct Color {
    uint8_t r = 0, g = 0, b = 0, a = 255;
    constexpr Color() = default;
    constexpr Color(uint8_t r_, uint8_t g_, uint8_t b_, uint8_t a_ = 255)
        : r(r_), g(g_), b(b_), a(a_) {}
    static constexpr Color rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
        return {r, g, b, a};
    }
    static constexpr Color hex(uint32_t hex) {
        const uint8_t a = (hex >> 24) & 0xFF;
        return {
            static_cast<uint8_t>((hex >> 16) & 0xFF),
            static_cast<uint8_t>((hex >> 8) & 0xFF),
            static_cast<uint8_t>(hex & 0xFF),
            a != 0 ? a : static_cast<uint8_t>(255)
        };
    }
    static constexpr Color transparent() { return {0,0,0,0}; }
    static constexpr Color white()   { return {255,255,255,255}; }
    static constexpr Color black()   { return {0,0,0,255}; }
    static constexpr Color red()     { return {255,0,0,255}; }
    static constexpr Color green()   { return {0,255,0,255}; }
    static constexpr Color blue()    { return {0,0,255,255}; }

    constexpr uint32_t to_rgba8() const {
        return (static_cast<uint32_t>(a) << 24)
             | (static_cast<uint32_t>(b) << 16)
             | (static_cast<uint32_t>(g) << 8)
             | static_cast<uint32_t>(r);
    }
};

// ── Event types ──────────────────────────────────────────────────────────
enum class EventType : uint8_t {
    None,
    MouseMove,
    MouseDown,
    MouseUp,
    Click,
    MouseEnter,
    MouseLeave,
    KeyDown,
    KeyUp,
    Char,
    Scroll,
    Resize,
    Focus,
    Blur
};

struct MouseEvent {
    float x = 0, y = 0;
    float screen_x = 0, screen_y = 0;
    uint8_t button = 0;
    bool pressed = false;
};

struct KeyEvent {
    int key = 0;
    int scancode = 0;
    int action = 0;
    int mods = 0;
    unsigned int codepoint = 0;
};

struct ScrollEvent {
    float dx = 0, dy = 0;
};

struct Event {
    EventType type = EventType::None;
    MouseEvent mouse{};
    KeyEvent key{};
    ScrollEvent scroll{};
    bool handled = false;

    static Event make_mouse_move(float lx, float ly, float sx, float sy) {
        Event e; e.type = EventType::MouseMove;
        e.mouse = {lx, ly, sx, sy}; return e;
    }
    static Event make_mouse_down(float sx, float sy, uint8_t btn) {
        Event e; e.type = EventType::MouseDown;
        e.mouse = {sx, sy, sx, sy, btn, true}; return e;
    }
    static Event make_mouse_up(float sx, float sy, uint8_t btn) {
        Event e; e.type = EventType::MouseUp;
        e.mouse = {sx, sy, sx, sy, btn, false}; return e;
    }
    static Event make_click(float sx, float sy) {
        Event e; e.type = EventType::Click;
        e.mouse = {sx, sy, sx, sy}; return e;
    }
    static Event make_scroll(float dx, float dy) {
        Event e; e.type = EventType::Scroll;
        e.scroll = {dx, dy}; return e;
    }
    static Event make_key(int k, int sc, int act, int mods_) {
        Event e; e.type = EventType::KeyDown;
        e.key = {k, sc, act, mods_}; return e;
    }
};

// ── Forward declaration ─────────────────────────────────────────────────
class App;
class Element;

// ── Element base class ───────────────────────────────────────────────────
class Element : public std::enable_shared_from_this<Element> {
public:
    Element() = default;
    virtual ~Element() = default;

    Element(const Element&) = delete;
    Element& operator=(const Element&) = delete;

    // ── Tree management ─────────────────────────────────────────────────
    void add_child(std::shared_ptr<Element> child) {
        child->parent_ = this;
        children_.push_back(std::move(child));
    }

    void remove_child(Element* child) {
        auto it = std::find_if(children_.begin(), children_.end(),
            [child](const auto& p) { return p.get() == child; });
        if (it != children_.end()) {
            (*it)->parent_ = nullptr;
            children_.erase(it);
        }
    }

    Element* parent() const { return parent_; }
    const std::vector<std::shared_ptr<Element>>& children() const { return children_; }
    std::vector<std::shared_ptr<Element>>& children() { return children_; }

    // ── Taffy node handle ───────────────────────────────────────────────
    uint64_t taffy_handle() const { return taffy_handle_; }
    void set_taffy_handle(uint64_t h) { taffy_handle_ = h; }

    // App back-pointer (set by App::build_node) so elements can request relayout.
    void set_app(App* a) { app_ = a; }

    // ── Layout ──────────────────────────────────────────────────────────
    float x()      const { return bounds_.x; }
    float y()      const { return bounds_.y; }
    float width()  const { return bounds_.w; }
    float height() const { return bounds_.h; }

    float inner_x()      const { return bounds_.x + margin_left() + border_left() + padding_left(); }
    float inner_y()      const { return bounds_.y + margin_top()  + border_top()  + padding_top(); }
    float inner_width()  const { return std::max(0.0f, bounds_.w - margin_left() - margin_right() - border_left() - border_right() - padding_left() - padding_right()); }
    float inner_height() const { return std::max(0.0f, bounds_.h - margin_top()   - margin_bottom()  - border_top()   - border_bottom()  - padding_top()   - padding_bottom()); }

    float margin_left()   const { return margin_.left; }
    float margin_right()  const { return margin_.right; }
    float margin_top()    const { return margin_.top; }
    float margin_bottom() const { return margin_.bottom; }
    float border_left()   const { return border_.left; }
    float border_right()  const { return border_.right; }
    float border_top()    const { return border_.top; }
    float border_bottom() const { return border_.bottom; }
    float padding_left()  const { return padding_.left; }
    float padding_right() const { return padding_.right; }
    float padding_top()   const { return padding_.top; }
    float padding_bottom()const { return padding_.bottom; }

    // ── Style helpers ───────────────────────────────────────────────────
    void set_background(Color c) { bg_color_ = c; }
    virtual Color background() const { return bg_color_; }
    void set_border_color(Color c) { border_color_ = c; }
    Color border_color() const { return border_color_; }
    void set_border_radius(float r) { border_radius_ = r; }
    float border_radius() const { return border_radius_; }
    void set_opacity(float o) { opacity_ = std::clamp(o, 0.0f, 1.0f); }
    float opacity() const { return opacity_; }
    void set_visible(bool v) { visible_ = v; }
    bool visible() const { return visible_; }
    void set_id(const std::string& sid) { id_ = sid; }
    const std::string& id() const { return id_; }

    // ── Event handling ──────────────────────────────────────────────────
    virtual bool on_event(const Event& e) {
        (void)e;
        return false;
    }

    virtual bool hit_test(float px, float py) const {
        return px >= 0 && px < bounds_.w && py >= 0 && py < bounds_.h;
    }

    // ── Lifecycle hooks ─────────────────────────────────────────────────
    virtual void on_create(App* app)  { (void)app; }
    virtual void on_mount(App* app)   { (void)app; }
    virtual void on_unmount(App* app) { (void)app; }
    virtual void build(App* app) { (void)app; }
    virtual void on_layout(App* app) { (void)app; }

    // Render phase — override for custom content between background and border.
    // Base render_element handles background fill and border stroke automatically.
    virtual void on_render(App* app, FS_Core* fs) {
        (void)app;
        (void)fs;
    }

    // Measure intrinsic content size for layout (leaf nodes only).
    // known_width <= 0 means unknown -> measure natural single-line size.
    // known_width > 0 means the content width is constrained; text may wrap to it.
    // out_w/out_h receive the measured content size (excluding padding/border).
    virtual void measure_content(FS_Core* fs, float known_width, float& out_w, float& out_h) {
        (void)fs; (void)known_width;
        out_w = 0.0f; out_h = 0.0f;
    }

    virtual void on_bounds_changed() {}

    // ── Internal (called by App) ────────────────────────────────────────
    void apply_layout(const taffy::Layout& lay);
    void propagate_bounds();

    float screen_x() const { return screen_bounds_.x; }
    float screen_y() const { return screen_bounds_.y; }
    float screen_width() const { return screen_bounds_.w; }
    float screen_height() const { return screen_bounds_.h; }

protected:
    struct Rect { float left=0, right=0, top=0, bottom=0; };
    struct Bounds { float x=0, y=0, w=0, h=0; };

    Bounds bounds_{};
    Bounds screen_bounds_{};
    Rect margin_{}, border_{}, padding_{};
    Color bg_color_ = Color::transparent();
    Color border_color_ = Color::transparent();
    float border_radius_ = 0.0f;
    float opacity_ = 1.0f;
    bool visible_ = true;
    std::string id_;

    uint64_t taffy_handle_ = 0;

    Element* parent_ = nullptr;
    std::vector<std::shared_ptr<Element>> children_;
    App* app_ = nullptr;

    // Request a relayout from the owning App (defined in app.hpp).
    void request_relayout();
};

} // namespace wcn_ui

#endif // WCN_UI_ELEMENT_HPP
