// router.hpp — Page router that drives GPU transitions between Element trees.
//
// Each Page is an Element subtree. When navigating from page A to page B:
//   1. Render A offscreen -> currentTex (one frame, cached)
//   2. Render B offscreen -> nextTex (one frame)
//   3. Run the selected Transition shader for `duration` seconds, composing
//      currentTex + nextTex into the swapchain.
//   4. On finish, B becomes the active page and is rendered normally.
#pragma once
#ifndef WCN_UI_ROUTER_HPP
#define WCN_UI_ROUTER_HPP

#include "transition.hpp"
#include "app.hpp"

namespace wcn_ui {

class Router {
public:
    void init(App* app) {
        app_ = app;
        if (app_) {
            device_ = app_->glfw()->device;
            queue_ = app_->glfw()->queue;
            surface_format_ = app_->glfw()->surface_format;
        }
    }

    // Register a named page (an Element subtree) with a default transition.
    void add_page(const std::string& name, std::shared_ptr<Element> root,
                  std::shared_ptr<Transition> transition = nullptr) {
        pages_[name] = { std::move(root), std::move(transition) };
        if (current_page_.empty()) current_page_ = name;
    }

    const std::string& current_page() const { return current_page_; }
    bool is_transitioning() const { return state_ == State::Transitioning; }

    // Navigate to a page. Uses the target page's registered transition, or
    // falls back to `fallback`. If mid-transition, queues the new target.
    void navigate(const std::string& to, std::shared_ptr<Transition> fallback = nullptr) {
        if (pages_.find(to) == pages_.end()) return;
        if (state_ == State::Transitioning) {
            // Allow interrupting: snap to target and re-transition from current.
            finish_transition(true);
        }
        if (to == current_page_) return;
        next_page_ = to;
        auto& pe = pages_[to];
        active_transition_ = pe.transition ? pe.transition : fallback;
        if (!active_transition_) active_transition_ = std::make_shared<LiquidMorphTransition>();
        start_transition();
    }

    // Per-frame: if transitioning, capture textures + run the transition render
    // pass instead of the normal render. Returns true if a transition is active
    // (so App should skip its normal render path this frame).
    bool update_and_render(double now) {
        if (state_ != State::Transitioning) return false;
        if (!app_ || !device_ || !queue_) { finish_transition(false); return false; }

        uint32_t w = app_->width();
        uint32_t h = app_->height();
        if (w == 0 || h == 0) return false;

        // (Re)create offscreen targets on first use / resize
        if (cur_.width != w || cur_.height != h) {
            cur_.create(device_, w, h, surface_format_);
            nxt_.create(device_, w, h, surface_format_);
            need_capture_ = true;
        }

        // Capture phase: render current page -> cur_, next page -> nxt_.
        // Done once at the start of the transition (cached).
        if (need_capture_) {
            capture_page(pages_[current_page_].root, cur_);
            capture_page(pages_[next_page_].root, nxt_);
            need_capture_ = false;
            transition_start_time_ = now;
        }

        // Ensure transition pipeline built for the surface format
        if (!active_transition_->ensure_pipeline(device_, surface_format_)) {
            finish_transition(false); return false;
        }

        // Ensure sampler
        if (!sampler_) {
            WGPUSamplerDescriptor sd{};
            sd.addressModeU = WGPUAddressMode_ClampToEdge;
            sd.addressModeV = WGPUAddressMode_ClampToEdge;
            sd.magFilter = WGPUFilterMode_Linear; sd.minFilter = WGPUFilterMode_Linear;
            sd.maxAnisotropy = 1;
            sampler_ = wgpuDeviceCreateSampler(device_, &sd);
        }

        // Progress with easing
        float raw = float((now - transition_start_time_) / active_transition_->duration());
        if (raw >= 1.0f) { finish_transition(false); return true; }
        float p = active_transition_->ease(std::clamp(raw, 0.0f, 1.0f));

        active_transition_->bind_textures(device_, queue_, cur_.view, nxt_.view, sampler_);

        // Composite into the swapchain via the backend's present path.
        // We hijack the render by drawing a fullscreen quad over a cleared target,
        // then present. We use fs_core_begin_commands + a manual render pass that
        // renders into the swapchain texture, then submit + present.
        render_composite(w, h, p);
        return true;
    }

    // When not transitioning, render the current page normally via App.
    std::shared_ptr<Element> current_root() {
        auto it = pages_.find(current_page_);
        return it != pages_.end() ? it->second.root : nullptr;
    }

private:
    struct PageEntry { std::shared_ptr<Element> root; std::shared_ptr<Transition> transition; };

    enum class State { Idle, Transitioning };

    App* app_ = nullptr;
    WGPUDevice device_ = nullptr;
    WGPUQueue queue_ = nullptr;
    WGPUTextureFormat surface_format_ = WGPUTextureFormat_Undefined;

    std::unordered_map<std::string, PageEntry> pages_;
    std::string current_page_;
    std::string next_page_;
    State state_ = State::Idle;

    std::shared_ptr<Transition> active_transition_;
    OffscreenTarget cur_, nxt_;
    WGPUSampler sampler_ = nullptr;
    bool need_capture_ = true;
    double transition_start_time_ = 0.0;

    void start_transition() {
        if (!app_) return;
        // Ensure the current page is laid out (so capture sees correct geometry).
        state_ = State::Transitioning;
        need_capture_ = true;
    }

    void finish_transition(bool snapped) {
        (void)snapped;
        if (!next_page_.empty()) current_page_ = next_page_;
        next_page_.clear();
        state_ = State::Idle;
        active_transition_.reset();
        if (app_) {
            // Hand off to App: make the new page the live root.
            auto root = current_root();
            if (root) app_->set_root(root);
        }
    }

    // Render a page's Element tree into an offscreen target.
    // We temporarily swap App's root to the page so layout + render traversal
    // target it, then render_to_target encodes into the offscreen texture.
    void capture_page(std::shared_ptr<Element> page_root, OffscreenTarget& target) {
        if (!app_ || !page_root || !target.view) return;
        app_->set_root(page_root);
        app_->render_to_target(target.texture, target.view, 0.0f, 0.0f, 0.0f, 0.0f);
    }

    void render_composite(uint32_t w, uint32_t h, float progress) {
        if (!app_) return;
        // Draw a fullscreen transition quad into the swapchain via App helper.
        app_->render_transition_frame(active_transition_.get(), w, h, progress,
                                      cur_.view, nxt_.view, sampler_);
    }
};

} // namespace wcn_ui

#endif // WCN_UI_ROUTER_HPP