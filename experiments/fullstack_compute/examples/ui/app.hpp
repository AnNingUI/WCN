// app.hpp — Application class: FS_Core ownership, taffy layout, event loop, render bridge
#pragma once
#ifndef WCN_UI_APP_HPP
#define WCN_UI_APP_HPP

#include "element.hpp"
#include "container.hpp"
#include "text.hpp"
#include "button.hpp"
#include "transition.hpp"
#include "../taffy.hpp"

// fullstack_compute is a C library — wrap its headers in extern "C" for C++ linkage
#ifdef __cplusplus
extern "C" {
#endif

#include "../../impl/fullstack_glfw_backend.h"
#include "../../impl/fullstack_stb_image_backend.h"
#include "../../impl/fullstack_stb_font_backend.h"
#include "../../include/fullstack_core.h"
#include "../../include/fullstack_core_debug.h"
#include "../../include/fullstack_effects.h"

#ifdef __cplusplus
}
#endif
#include <cmath>
#include <cstdio>
#include <memory>
#include <unordered_map>
#include <vector>

namespace wcn_ui {


// ── Helper: convert uint64_t handle ↔ taffy::NodeId ─────────────────────
static inline taffy::NodeId handle_to_nodeid(uint64_t h) {
    return taffy::NodeId{uint32_t(h >> 32), uint32_t(h & 0xFFFFFFFFu)};
}
static inline uint64_t nodeid_to_handle(taffy::NodeId id) {
    return (uint64_t(id.index) << 32) | uint64_t(id.generation);
}

namespace detail {

inline taffy::Style element_to_taffy_style(const Container& el) { return el.style(); }
inline taffy::Style element_to_taffy_style(const Text& el) { return el.style(); }
inline taffy::Style element_to_taffy_style(const Button& el) { return el.style(); }

} // namespace detail

// ── App ──────────────────────────────────────────────────────────────────
class App : public std::enable_shared_from_this<App> {
public:
    App() = default;
    ~App() { shutdown(); }

    App(const App&) = delete;
    App& operator=(const App&) = delete;

    bool init(uint32_t width, uint32_t height, const char* title) {
        width_ = width;
        height_ = height;

        if (!fs_glfw_backend_init(&glfw_, width, height, title)) {
            std::fprintf(stderr, "App: fs_glfw_backend_init failed\n");
            return false;
        }

        int fb_w = 0, fb_h = 0;
        glfwGetFramebufferSize(glfw_.window, &fb_w, &fb_h);
        if (fb_w > 0 && fb_h > 0) {
            width_ = (uint32_t)fb_w;
            height_ = (uint32_t)fb_h;
        }

        fs_ = fs_glfw_backend_core(&glfw_);
        if (!fs_) {
            std::fprintf(stderr, "App: fs_glfw_backend_core returned null\n");
            return false;
        }

        static const FS_ImageBackend* stb_image = fs_get_stb_image_backend();
        if (stb_image && !fs_core_set_image_backend(fs_, stb_image)) {
            std::fprintf(stderr, "App: fs_core_set_image_backend failed\n");
            return false;
        }

        static const FS_FontBackend* stb_font = fs_get_stb_font_backend();
        if (stb_font && !fs_core_set_font_backend(fs_, stb_font)) {
            std::fprintf(stderr, "App: fs_core_set_font_backend failed\n");
            return false;
        }

        // Note: fs_effects_init is already called inside fs_core_init -> fs_glfw_backend_init.
        // Do NOT call it again here -- it would overwrite core->effects with a new resource
        // that lacks the presentation pipeline, causing a black screen on the first frame.

        const char* default_font_path = nullptr;
        #if defined(_WIN32)
            default_font_path = "C:\\Windows\\Fonts\\arial.ttf";
        #elif defined(__APPLE__)
            default_font_path = "/System/Library/Fonts/Helvetica.ttc";
        #else
            default_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
        #endif
        if (default_font_path) {
            if (!fs_core_load_font_file(fs_, default_font_path)) {
                std::fprintf(stderr, "App: fs_core_load_font_file(\"%s\") failed (non-fatal)\n", default_font_path);
            }
        }

        install_input_callbacks();

        return true;
    }

    void shutdown() {
        fs_glfw_backend_shutdown(&glfw_);
    }

    void set_root(std::shared_ptr<Element> root) {
        root_ = std::move(root);
        rebuild_tree();
    }

    Element* root() const { return root_.get(); }

    void rebuild_tree() {
        taffy_tree_ = std::make_unique<taffy::TaffyTree<>>();
        element_to_node_.clear();
        node_to_element_.clear();

        if (root_) {
            root_->on_create(this);
            root_->build(this);
            build_node(root_.get());
        }
    }

    bool poll_window_size() {
        if (!glfw_.window) return false;
        int fb_w = 0, fb_h = 0;
        glfwGetFramebufferSize(glfw_.window, &fb_w, &fb_h);
        if (fb_w <= 0 || fb_h <= 0) return false;
        uint32_t new_w = (uint32_t)fb_w;
        uint32_t new_h = (uint32_t)fb_h;
        if (new_w != width_ || new_h != height_) {
            width_ = new_w;
            height_ = new_h;
            needs_relayout_ = true;
            return true;
        }
        return false;
    }

    void perform_layout() {
        if (!root_ || !taffy_tree_) return;
        root_->on_layout(this);

        taffy::Size<taffy::AvailableSpace> available = {
            taffy::AvailableSpace::definite(static_cast<float>(width_)),
            taffy::AvailableSpace::definite(static_cast<float>(height_))
        };

        // Measure function: lets leaf nodes (Text/Button) report their intrinsic
        // content size to taffy so text is actually accounted for in the layout.
        auto measure_fn = [this](taffy::Size<std::optional<float>> known_dims,
                                 taffy::Size<taffy::AvailableSpace> available,
                                 taffy::NodeId node_id,
                                 const taffy::Style& style) -> taffy::Size<float> {
            Element* el = this->node_to_element(node_id);
            if (!el) return {0.0f, 0.0f};

            // Determine the content-box width available for text wrapping.
            // taffy passes child_available_space with the node's own
            // padding/border already subtracted (see compute_leaf_layout_impl).
            float content_w = -1.0f;  // -1 = unknown/max-content -> single-line size
            if (known_dims.width.has_value()) {
                // Node width is fully determined: content width = node width - padding - border.
                auto pb = taffy::resolve_or_zero_rect(style.padding, known_dims.width);
                auto bb = taffy::resolve_or_zero_rect(style.border, known_dims.width);
                content_w = std::max(0.0f, *known_dims.width - pb.left - pb.right - bb.left - bb.right);
            } else if (available.width.is_definite()) {
                content_w = std::max(0.0f, available.width.unwrap());
            } else if (available.width.type == taffy::AvailableSpace::MinContent) {
                content_w = -2.0f;  // min-content sentinel -> widest word
            }

            float w = 0.0f, h = 0.0f;
            el->measure_content(fs_, content_w, w, h);
            return {w, h};
        };
        taffy_tree_->compute_layout_with_measure(root_element_node(), available, measure_fn);
        apply_layout_to_element(root_.get());
        needs_relayout_ = false;

        if (resized_this_frame_) {
            notify_resize(root_.get());
            resized_this_frame_ = false;
        }
    }

    void render() {
        if (!root_ || !fs_) return;

        fs_core_begin_commands(fs_);
        fs_transform_reset(fs_);
        fs_style_reset(fs_);
        render_element(root_.get());
        if (!fs_glfw_backend_present(&glfw_, 0.94f, 0.94f, 0.94f, 1.0f)) {
            std::fprintf(stderr, "App: fs_glfw_backend_present failed\n");
        }
    }

    void run() {
        running_ = true;
        std::fprintf(stderr, "App: entering run loop\n");

        while (running_ && !fs_glfw_backend_should_close(&glfw_)) {
            fs_glfw_backend_poll_events();
            process_input_events();

            if (poll_window_size()) {
                resized_this_frame_ = true;
            }

            if (first_frame_) {
                first_frame_ = false;
                perform_layout();
            } else if (needs_relayout_) {
                perform_layout();
            }

            render();
        }
    }

    void quit() { running_ = false; }

    // ── Input: GLFW callbacks ───────────────────────────────────────────
    void install_input_callbacks() {
        if (!glfw_.window) return;
        glfwSetWindowUserPointer(glfw_.window, this);
        glfwSetMouseButtonCallback(glfw_.window, &App::mouse_button_callback);
        glfwSetCursorPosCallback(glfw_.window, &App::cursor_pos_callback);
    }

    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        auto* self = static_cast<App*>(glfwGetWindowUserPointer(window));
        if (!self) return;
        // Use the last cursor position reported by cursor_pos_callback.
        // glfwGetCursorPos can return stale/zero values when the button
        // event is processed in a different poll cycle than the motion.
        float sx = self->mouse_x_;
        float sy = self->mouse_y_;
        // GLFW button 0 == left; only forward left clicks as clicks/presses.
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            if (action == GLFW_PRESS) {
                self->pending_events_.push_back(Event::make_mouse_down(sx, sy, 0));
            } else if (action == GLFW_RELEASE) {
                self->pending_events_.push_back(Event::make_mouse_up(sx, sy, 0));
                self->pending_events_.push_back(Event::make_click(sx, sy));
            }
        }
        (void)mods;
    }

    static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
        auto* self = static_cast<App*>(glfwGetWindowUserPointer(window));
        if (!self) return;
        float sx = float(xpos), sy = float(ypos);
        self->mouse_x_ = sx;
        self->mouse_y_ = sy;
        self->pending_events_.push_back(Event::make_mouse_move(0, 0, sx, sy));
    }

    void process_input_events() {
        if (pending_events_.empty()) return;
        for (const auto& ev : pending_events_) {
            if (ev.type == EventType::MouseMove ||
                ev.type == EventType::MouseDown ||
                ev.type == EventType::MouseUp) {
                hit_test_and_dispatch(ev, root_.get());
            } else {
                dispatch_event(ev);
            }
        }
        pending_events_.clear();
    }

    FS_Core* fs() const { return fs_; }
    FS_GlfwBackend* glfw() { return &glfw_; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    taffy::TaffyTree<>& taffy() { return *taffy_tree_; }

    // ── Transition engine hooks ────────────────────────────────────────
    // Render the current root tree into an arbitrary offscreen target
    // (WGPUTexture + view) instead of the swapchain. Used by Router to
    // capture page textures. Clears to the given rgba (0..1).
    void render_to_target(WGPUTexture target_texture, WGPUTextureView target_view,
                          float cr, float cg, float cb, float ca) {
        if (!root_ || !fs_ || !target_view) return;
        perform_layout();
        fs_core_begin_commands(fs_);
        fs_transform_reset(fs_);
        fs_style_reset(fs_);
        render_element(root_.get());
        // Encode into the offscreen target (bypassing the swapchain present).
        WGPUCommandEncoderDescriptor ed{};
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(glfw_.device, &ed);
        if (!encoder) return;
        fs_core_encode(fs_, encoder, target_texture, target_view, cr, cg, cb, ca);
        WGPUCommandBufferDescriptor cbd{};
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cbd);
        if (cmd) { wgpuQueueSubmit(glfw_.queue, 1, &cmd); wgpuCommandBufferRelease(cmd); }
        wgpuCommandEncoderRelease(encoder);
    }

    // Render a fullscreen transition quad composing currentView + nextView
    // into the swapchain and present. Called by Router each frame of a transition.
    void render_transition_frame(Transition* tr, uint32_t w, uint32_t h, float progress,
                                 WGPUTextureView current_view, WGPUTextureView next_view,
                                 WGPUSampler sampler) {
        if (!tr || !glfw_.window || !glfw_.surface) return;
        tr->ensure_pipeline(glfw_.device, glfw_.surface_format);
        tr->bind_textures(glfw_.device, glfw_.queue, current_view, next_view, sampler);

        // Acquire swapchain texture
        WGPUSurfaceTexture st{};
        wgpuSurfaceGetCurrentTexture(glfw_.surface, &st);
        if (st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
            st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
            if (st.texture) { wgpuTextureRelease(st.texture); st.texture = nullptr; }
            return;
        }
        WGPUTextureViewDescriptor vd{};
        vd.format = WGPUTextureFormat_Undefined;
        vd.dimension = WGPUTextureViewDimension_2D;
        vd.aspect = WGPUTextureAspect_All;
        vd.baseMipLevel = 0; vd.mipLevelCount = 1;
        vd.baseArrayLayer = 0; vd.arrayLayerCount = 1;
        WGPUTextureView swap_view = wgpuTextureCreateView(st.texture, &vd);
        if (!swap_view) { if (st.texture) wgpuTextureRelease(st.texture); return; }

        WGPUCommandEncoderDescriptor ed{};
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(glfw_.device, &ed);
        if (!encoder) { wgpuTextureViewRelease(swap_view); if (st.texture) wgpuTextureRelease(st.texture); return; }

        WGPURenderPassColorAttachment att{};
        att.view = swap_view;
        att.loadOp = WGPULoadOp_Clear;
        att.storeOp = WGPUStoreOp_Store;
        att.clearValue = { 0.0f, 0.0f, 0.0f, 1.0f };
        WGPURenderPassDescriptor rpd{};
        rpd.colorAttachmentCount = 1;
        rpd.colorAttachments = &att;
        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &rpd);
        if (pass) {
            tr->record(pass, w, h, progress);
            wgpuRenderPassEncoderEnd(pass);
            wgpuRenderPassEncoderRelease(pass);
        }
        wgpuTextureViewRelease(swap_view);

        WGPUCommandBufferDescriptor cbd{};
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cbd);
        if (cmd) { wgpuQueueSubmit(glfw_.queue, 1, &cmd); wgpuCommandBufferRelease(cmd); }
        wgpuCommandEncoderRelease(encoder);

        wgpuSurfacePresent(glfw_.surface);
        if (st.texture) wgpuTextureRelease(st.texture);
    }

    void dispatch_event(Event e, Element* target = nullptr) {
        if (target) {
            target->on_event(e);
        } else if (root_) {
            if (e.type == EventType::MouseMove ||
                e.type == EventType::MouseDown ||
                e.type == EventType::MouseUp) {
                hit_test_and_dispatch(e, root_.get());
            } else {
                root_->on_event(e);
            }
        }
    }

    taffy::NodeId element_to_node(Element* el) const {
        auto it = element_to_node_.find(el);
        return it != element_to_node_.end() ? handle_to_nodeid(it->second) : taffy::NODE_ID_NONE;
    }

    Element* node_to_element(taffy::NodeId id) const {
        auto it = node_to_element_.find(nodeid_to_handle(id));
        return it != node_to_element_.end() ? it->second : nullptr;
    }

    // Mark an element's taffy node (and its ancestors) dirty and request a
    // relayout on the next frame. Called when content (e.g. text) changes.
    void invalidate_element(Element* el) {
        if (!el || !taffy_tree_) return;
        taffy::NodeId nid = element_to_node(el);
        if (nid != taffy::NODE_ID_NONE) {
            taffy_tree_->mark_dirty(nid);
        }
        needs_relayout_ = true;
    }

    using ResizeCallback = std::function<void(uint32_t w, uint32_t h)>;
    void set_on_resize(ResizeCallback cb) { resize_callback_ = std::move(cb); }

protected:
    FS_GlfwBackend glfw_{};
    FS_Core* fs_ = nullptr;
    uint32_t width_ = 0, height_ = 0;
    bool running_ = false;
    bool first_frame_ = true;
    bool needs_relayout_ = true;
    bool resized_this_frame_ = false;

private:

    std::shared_ptr<Element> root_;
    std::unique_ptr<taffy::TaffyTree<>> taffy_tree_;

    std::unordered_map<Element*, uint64_t> element_to_node_;
    std::unordered_map<uint64_t, Element*> node_to_element_;

    float mouse_x_ = 0, mouse_y_ = 0;
    Element* hovered_element_ = nullptr;
    ResizeCallback resize_callback_;
    std::vector<Event> pending_events_;

    taffy::NodeId root_element_node() const {
        return element_to_node(root_.get());
    }

    taffy::NodeId build_node(Element* el) {
        std::vector<taffy::NodeId> child_nodes;
        for (auto& child : el->children()) {
            taffy::NodeId cid = build_node(child.get());
            if (cid != taffy::NODE_ID_NONE) {
                child_nodes.push_back(cid);
            }
        }

        taffy::Style style;
        if (auto* container = dynamic_cast<Container*>(el)) {
            style = detail::element_to_taffy_style(*container);
        } else if (auto* text = dynamic_cast<Text*>(el)) {
            style = detail::element_to_taffy_style(*text);
        } else if (auto* button = dynamic_cast<Button*>(el)) {
            style = detail::element_to_taffy_style(*button);
        } else {
            style.display = taffy::Display::Flex;
            style.flex_direction = taffy::FlexDirection::Row;
        }

        taffy::NodeId node_id;
        if (child_nodes.empty()) {
            node_id = taffy_tree_->new_leaf(style);
        } else {
            node_id = taffy_tree_->new_with_children(style, child_nodes);
        }

        el->set_taffy_handle(nodeid_to_handle(node_id));
        el->set_app(this);
        element_to_node_[el] = nodeid_to_handle(node_id);
        node_to_element_[nodeid_to_handle(node_id)] = el;

        return node_id;
    }

    void apply_layout_to_element(Element* el) {
        taffy::NodeId nid = handle_to_nodeid(el->taffy_handle());
        if (nid == taffy::NODE_ID_NONE) return;
        const auto& lay = taffy_tree_->layout(nid);
        el->apply_layout(lay);
        for (auto& child : el->children()) {
            apply_layout_to_element(child.get());
        }
    }

    void notify_resize(Element* el) {
        el->on_bounds_changed();
        for (auto& child : el->children()) {
            notify_resize(child.get());
        }
    }

    // ── Base rendering: background → content → children → border ────────
    void render_element(Element* el) {
        if (!el->visible()) return;

        float x = el->screen_x();
        float y = el->screen_y();
        float w = el->screen_width();
        float h = el->screen_height();
        float r = el->border_radius();

        // 1. Background fill
        Color bg = el->background();
        if (bg.a > 0) {
            fs_cmd_rect(fs_, x, y, w, h, r, bg.to_rgba8());
        }

        // 2. Custom content (text, icons, etc.)
        el->on_render(this, fs_);

        // 3. Children
        for (auto& child : el->children()) {
            render_element(child.get());
        }

        // 4. Border stroke (on top)
        Color bc = el->border_color();
        if (bc.a > 0 && el->border_left() > 0) {
            fs_cmd_rect_stroke(fs_, x, y, w, h, r, el->border_left(), bc.to_rgba8());
        }
    }

    Element* hit_test_tree(Element* el, float sx, float sy) {
        if (!el->visible()) return nullptr;
        for (auto it = el->children().rbegin(); it != el->children().rend(); ++it) {
            Element* found = hit_test_tree(it->get(), sx, sy);
            if (found) return found;
        }
        float lx = sx - el->screen_x();
        float ly = sy - el->screen_y();
        if (el->hit_test(lx, ly)) return el;
        return nullptr;
    }

    void hit_test_and_dispatch(const Event& e, Element* root) {
        float sx = e.mouse.screen_x;
        float sy = e.mouse.screen_y;
        Element* target = hit_test_tree(root, sx, sy);


        if (target != hovered_element_) {
            if (hovered_element_) {
                Event leave = Event::make_mouse_move(0,0,0,0);
                leave.type = EventType::MouseLeave;
                hovered_element_->on_event(leave);
            }
            if (target) {
                Event enter = Event::make_mouse_move(0,0,0,0);
                enter.type = EventType::MouseEnter;
                target->on_event(enter);
            }
            hovered_element_ = target;
        }

        if (target) {
            float lx = sx - target->screen_x();
            float ly = sy - target->screen_y();
            Event local = e;
            local.mouse.x = lx;
            local.mouse.y = ly;
            target->on_event(local);
        }
    }
};

// ── Render implementations ───────────────────────────────────────────────

inline void Element::apply_layout(const taffy::Layout& lay) {
    bounds_.x = lay.location.x;
    bounds_.y = lay.location.y;
    bounds_.w = lay.size.width;
    bounds_.h = lay.size.height;
    margin_  = {lay.margin.left, lay.margin.right, lay.margin.top, lay.margin.bottom};
    border_  = {lay.border.left, lay.border.right, lay.border.top, lay.border.bottom};
    padding_ = {lay.padding.left, lay.padding.right, lay.padding.top, lay.padding.bottom};
    propagate_bounds();
}

inline void Element::request_relayout() {
    if (app_) app_->invalidate_element(this);
}

inline void Element::propagate_bounds() {
    Element* p = parent_;
    if (p) {
        screen_bounds_.x = p->screen_bounds_.x + bounds_.x;
        screen_bounds_.y = p->screen_bounds_.y + bounds_.y;
    } else {
        screen_bounds_.x = bounds_.x;
        screen_bounds_.y = bounds_.y;
    }
    screen_bounds_.w = bounds_.w;
    screen_bounds_.h = bounds_.h;
    for (auto& child : children_) child->propagate_bounds();
}

// Container: no custom rendering — background/border handled by render_element
inline void Container::on_render(App* app, FS_Core* fs) { (void)app; (void)fs; }

// Text: draws text centered vertically within inner area.
// Uses SCREEN coordinates (screen_x/y + insets) since fs_cmd_text_utf8
// draws in absolute window coords, not local element coords.
inline void Text::on_render(App* app, FS_Core* fs) {
    (void)app;
    if (text_.empty()) return;
    FS_TextMetrics metrics{};
    if (!fs_measure_text_utf8(fs, font_size_, text_.c_str(), 0, &metrics)) return;

    float sx = screen_x() + margin_left() + border_left() + padding_left();
    float sy = screen_y() + margin_top()  + border_top()  + padding_top();
    float iw = inner_width();
    float ih = inner_height();

    float asc = metrics.em_height_ascent;
    float desc = metrics.em_height_descent;
    float baseline_y = sy + (ih + asc - desc) * 0.5f;
    float max_w = iw;

    fs_cmd_text_utf8(fs, sx, baseline_y, font_size_, text_.c_str(), color_.to_rgba8(), max_w);
}

// Button: draws label text centered within inner area.
// Uses SCREEN coordinates (screen_x/y + insets) since fs_cmd_text_utf8
// draws in absolute window coords, not local element coords.
inline void Button::on_render(App* app, FS_Core* fs) {
    (void)app;
    if (label_.empty()) return;

    FS_TextMetrics metrics{};
    fs_measure_text_utf8(fs, font_size_, label_.c_str(), 0, &metrics);
    float asc = metrics.em_height_ascent;
    float desc = metrics.em_height_descent;

    float sx = screen_x() + margin_left() + border_left() + padding_left();
    float sy = screen_y() + margin_top()  + border_top()  + padding_top();
    float iw = inner_width();

    float cx = sx + (iw - metrics.width) * 0.5f;
    float cy = sy + (inner_height() + asc - desc) * 0.5f;

    fs_cmd_text_utf8(fs, cx, cy, font_size_, label_.c_str(), text_color_.to_rgba8(), iw);
}

namespace detail {
// Min-content width of a UTF-8 string = width of its widest whitespace-delimited
// token (text can always wrap at word boundaries). Returns 0 on failure.
inline float measure_min_content_width(FS_Core* fs, float font_size, const std::string& s) {
    float best = 0.0f;
    size_t i = 0, n = s.size();
    while (i < n) {
        while (i < n && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
        size_t start = i;
        while (i < n && !(s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
        if (i > start) {
            std::string token(s, start, i - start);
            FS_TextMetrics m{};
            if (fs_measure_text_utf8(fs, font_size, token.c_str(), 0.0f, &m) && m.width > best)
                best = m.width;
        }
    }
    return best;
}
} // namespace detail

// Text: measure text content size so taffy gives text nodes a real layout size.
// known_width > 0  -> wrap to that width (definite).
// known_width == 0 or -1 -> max-content (single line).
// known_width <= -2 -> min-content (widest word).
inline void Text::measure_content(FS_Core* fs, float known_width, float& out_w, float& out_h) {
    out_w = 0.0f; out_h = 0.0f;
    if (!fs || text_.empty()) return;
    FS_TextMetrics m{};
    if (known_width <= -2.0f) {
        out_w = detail::measure_min_content_width(fs, font_size_, text_);
        // Height of one line; the real (wrapped) height is re-measured later
        // once taffy assigns a definite width.
        if (fs_measure_text_utf8(fs, font_size_, text_.c_str(), 0.0f, &m))
            out_h = m.em_height_ascent + m.em_height_descent;
        return;
    }
    float mw = known_width > 0.0f ? known_width : 0.0f;
    if (!fs_measure_text_utf8(fs, font_size_, text_.c_str(), mw, &m)) return;
    out_w = m.width;
    uint32_t lines = m.line_count > 0u ? m.line_count : 1u;
    out_h = (m.em_height_ascent + m.em_height_descent) * static_cast<float>(lines);
}

// Button: measure label text; taffy adds padding/border around it automatically.
inline void Button::measure_content(FS_Core* fs, float known_width, float& out_w, float& out_h) {
    out_w = 0.0f; out_h = 0.0f;
    if (!fs || label_.empty()) return;
    FS_TextMetrics m{};
    if (known_width <= -2.0f) {
        out_w = detail::measure_min_content_width(fs, font_size_, label_);
        if (fs_measure_text_utf8(fs, font_size_, label_.c_str(), 0.0f, &m))
            out_h = m.em_height_ascent + m.em_height_descent;
        return;
    }
    float mw = known_width > 0.0f ? known_width : 0.0f;
    if (!fs_measure_text_utf8(fs, font_size_, label_.c_str(), mw, &m)) return;
    out_w = m.width;
    uint32_t lines = m.line_count > 0u ? m.line_count : 1u;
    out_h = (m.em_height_ascent + m.em_height_descent) * static_cast<float>(lines);
}

} // namespace wcn_ui

#endif // WCN_UI_APP_HPP
