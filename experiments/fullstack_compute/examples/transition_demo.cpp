// transition_demo.cpp — GPU-driven page transition engine demo.
//
// Three pages, each with a distinct visual identity. Press 1/2/3 (or click
// the on-screen buttons) to navigate; each route uses a different transition:
//   home  -> analytics  : LiquidMorph
//   home  -> settings   : RealityTear
//   any   -> home       : InkSpread
//
// Architecture: Router owns pages (Element subtrees). On navigate it captures
// each page to an offscreen RGBA8 texture via fs_core_encode, then runs a WGSL
// fullscreen shader that blends current+next by progress, presented to the
// swapchain each frame.

#include "ui/app.hpp"
#include "ui/transition.hpp"
#include "ui/router.hpp"

#include <chrono>
#include <cstdio>
#include <memory>

// ── Forward declarations ──────────────────────────────────────────────
class TransitionApp;
static std::shared_ptr<wcn_ui::Element> make_home_page(TransitionApp* app);
static std::shared_ptr<wcn_ui::Element> make_analytics_page(TransitionApp* app);
static std::shared_ptr<wcn_ui::Element> make_settings_page(TransitionApp* app);
static std::shared_ptr<wcn_ui::Element> make_gallery_page(TransitionApp* app);

// ── App subclass with button routing ────────────────────────────────────
class TransitionApp : public wcn_ui::App {
public:
    wcn_ui::Router router;

    void navigate_to(const std::string& name) {
        if (name == "home") {
            router.navigate("home", std::make_shared<wcn_ui::InkSpreadTransition>());
        } else {
            router.navigate(name);
        }
    }

    void setup_router() {
        router.init(this);
        router.add_page("home", make_home_page(this));
        router.add_page("analytics", make_analytics_page(this),
                        std::make_shared<wcn_ui::LiquidMorphTransition>());
        router.add_page("settings", make_settings_page(this),
                        std::make_shared<wcn_ui::RealityTearTransition>());
        router.add_page("gallery", make_gallery_page(this),
                        std::make_shared<wcn_ui::GlassShatterTransition>());
        set_root(router.current_root());
    }

    void run_with_transitions() {
        running_ = true;
        std::fprintf(stderr, "TransitionApp: running. Click buttons to navigate.\n");
        auto t0 = std::chrono::steady_clock::now();
        while (running_ && !fs_glfw_backend_should_close(&glfw_)) {
            fs_glfw_backend_poll_events();
            process_input_events();

            auto now = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            if (router.update_and_render(now)) {
                continue;
            }
            if (poll_window_size()) resized_this_frame_ = true;
            if (first_frame_) { first_frame_ = false; perform_layout(); }
            else if (needs_relayout_) perform_layout();
            render();
        }
    }
};

// ── Page builders (defined after TransitionApp so they can reference it) ─
using TA = TransitionApp;

// Helper: create a row of navigation buttons
static std::shared_ptr<wcn_ui::Element> make_nav_row(TA* app,
    const std::vector<std::pair<std::string, std::string>>& buttons) {
    auto row = std::make_shared<wcn_ui::HStack>(12);
    for (auto& [label, page] : buttons) {
        auto btn = std::make_shared<wcn_ui::Button>(label);
        btn->set_on_click([app, page](wcn_ui::Button&) { app->navigate_to(page); });
        btn->set_font_size(14);
        row->add_child(btn);
    }
    return row;
}

static std::shared_ptr<wcn_ui::Element> make_home_page(TA* app) {
    auto root = std::make_shared<wcn_ui::Container>();
    auto& s = root->style_mut();
    s.display = taffy::Display::Flex;
    s.flex_direction = taffy::FlexDirection::Column;
    s.align_items = taffy::AlignItems{taffy::AlignItemsKeyword::Center, taffy::AlignmentSafety::Unsafe};
    s.justify_content = taffy::JustifyContent{taffy::AlignContentKeyword::Center, taffy::AlignmentSafety::Unsafe};
    s.size = {taffy::Dimension::percent(100), taffy::Dimension::percent(100)};
    s.gap = {taffy::LengthPercentage::length(0), taffy::LengthPercentage::length(20)};
    root->set_background(wcn_ui::Color::hex(0x1A1A2E));

    auto title = std::make_shared<wcn_ui::Text>("WCN Transition Engine");
    title->set_font_size(40);
    title->set_color(wcn_ui::Color::white());
    root->add_child(title);

    auto sub = std::make_shared<wcn_ui::Text>("Click a button to switch pages  •  each route has a different GPU transition");
    sub->set_font_size(16);
    sub->set_color(wcn_ui::Color::hex(0x8888AA));
    root->add_child(sub);

    auto row = std::make_shared<wcn_ui::HStack>(16);
    auto b1 = std::make_shared<wcn_ui::Button>("Analytics (Liquid)");
    auto b2 = std::make_shared<wcn_ui::Button>("Settings (Tear)");
    auto b3 = std::make_shared<wcn_ui::Button>("Gallery (Glass)");
    b1->set_on_click([app](wcn_ui::Button&) { app->navigate_to("analytics"); });
    b2->set_on_click([app](wcn_ui::Button&) { app->navigate_to("settings"); });
    b3->set_on_click([app](wcn_ui::Button&) { app->navigate_to("gallery"); });
    row->add_child(b1); row->add_child(b2); row->add_child(b3);
    root->add_child(row);
    return root;
}

static std::shared_ptr<wcn_ui::Element> make_analytics_page(TA* app) {
    auto root = std::make_shared<wcn_ui::Container>();
    auto& s = root->style_mut();
    s.display = taffy::Display::Flex;
    s.flex_direction = taffy::FlexDirection::Column;
    s.padding = {taffy::LengthPercentage::length(40), taffy::LengthPercentage::length(40),
                 taffy::LengthPercentage::length(40), taffy::LengthPercentage::length(40)};
    s.gap = {taffy::LengthPercentage::length(0), taffy::LengthPercentage::length(20)};
    s.size = {taffy::Dimension::percent(100), taffy::Dimension::percent(100)};
    root->set_background(wcn_ui::Color::hex(0x0F3460));

    auto title = std::make_shared<wcn_ui::Text>("Analytics");
    title->set_font_size(34); title->set_color(wcn_ui::Color::white());
    root->add_child(title);

    auto row = std::make_shared<wcn_ui::HStack>(16);
    for (int i = 0; i < 3; i++) {
        auto card = std::make_shared<wcn_ui::Container>();
        auto& cs = card->style_mut();
        cs.display = taffy::Display::Flex; cs.flex_direction = taffy::FlexDirection::Column;
        cs.padding = {taffy::LengthPercentage::length(20), taffy::LengthPercentage::length(20),
                      taffy::LengthPercentage::length(20), taffy::LengthPercentage::length(20)};
        cs.flex_grow = 1; cs.flex_basis = taffy::Dimension::length(0);
        card->set_background(wcn_ui::Color::hex(0x16213E));
        card->set_border_radius(8);
        char buf[32]; std::snprintf(buf, sizeof(buf), "Metric %d", i + 1);
        auto t = std::make_shared<wcn_ui::Text>(buf);
        t->set_font_size(22); t->set_color(wcn_ui::Color::hex(0xE94560));
        card->add_child(t);
        auto v = std::make_shared<wcn_ui::Text>("+ 12.4 %");
        v->set_font_size(28); v->set_color(wcn_ui::Color::white());
        card->add_child(v);
        row->add_child(card);
    }
    root->add_child(row);

    root->add_child(make_nav_row(app, {
        {"Home (Ink)", "home"},
        {"Settings (Tear)", "settings"},
        {"Gallery (Glass)", "gallery"}
    }));
    return root;
}

static std::shared_ptr<wcn_ui::Element> make_settings_page(TA* app) {
    auto root = std::make_shared<wcn_ui::Container>();
    auto& s = root->style_mut();
    s.display = taffy::Display::Flex;
    s.flex_direction = taffy::FlexDirection::Column;
    s.padding = {taffy::LengthPercentage::length(40), taffy::LengthPercentage::length(40),
                 taffy::LengthPercentage::length(40), taffy::LengthPercentage::length(40)};
    s.gap = {taffy::LengthPercentage::length(0), taffy::LengthPercentage::length(16)};
    s.size = {taffy::Dimension::percent(100), taffy::Dimension::percent(100)};
    root->set_background(wcn_ui::Color::hex(0x2D2D2D));

    auto title = std::make_shared<wcn_ui::Text>("Settings");
    title->set_font_size(34); title->set_color(wcn_ui::Color::hex(0xF5F5F5));
    root->add_child(title);

    const char* labels[] = {"Theme: Dark", "Language: English", "GPU: WebGPU (wgpu_native)", "Renderer: fullstack_compute"};
    for (auto& l : labels) {
        auto item = std::make_shared<wcn_ui::Container>();
        auto& is = item->style_mut();
        is.display = taffy::Display::Flex; is.flex_direction = taffy::FlexDirection::Row;
        is.align_items = taffy::AlignItems{taffy::AlignItemsKeyword::Center, taffy::AlignmentSafety::Unsafe};
        is.padding = {taffy::LengthPercentage::length(16), taffy::LengthPercentage::length(16),
                      taffy::LengthPercentage::length(14), taffy::LengthPercentage::length(14)};
        item->set_background(wcn_ui::Color::hex(0x3A3A3A));
        item->set_border_radius(6);
        auto t = std::make_shared<wcn_ui::Text>(l);
        t->set_font_size(18); t->set_color(wcn_ui::Color::hex(0xD6D6D6));
        item->add_child(t);
        root->add_child(item);
    }
    root->add_child(make_nav_row(app, {
        {"Home (Ink)", "home"},
        {"Analytics (Liquid)", "analytics"},
        {"Gallery (Glass)", "gallery"}
    }));
    return root;
}

static std::shared_ptr<wcn_ui::Element> make_gallery_page(TA* app) {
    auto root = std::make_shared<wcn_ui::Container>();
    auto& s = root->style_mut();
    s.display = taffy::Display::Flex;
    s.flex_direction = taffy::FlexDirection::Column;
    s.padding = {taffy::LengthPercentage::length(40), taffy::LengthPercentage::length(40),
                 taffy::LengthPercentage::length(40), taffy::LengthPercentage::length(40)};
    s.gap = {taffy::LengthPercentage::length(0), taffy::LengthPercentage::length(20)};
    s.size = {taffy::Dimension::percent(100), taffy::Dimension::percent(100)};
    root->set_background(wcn_ui::Color::hex(0x22223B));

    auto title = std::make_shared<wcn_ui::Text>("Gallery");
    title->set_font_size(34); title->set_color(wcn_ui::Color::hex(0xF2E9E4));
    root->add_child(title);

    auto grid = std::make_shared<wcn_ui::HStack>(16);
    uint32_t colors[] = {0xC9ADA7, 0xE9806F, 0xA7C7E7, 0x8E9AAF};
    for (int i = 0; i < 4; i++) {
        auto tile = std::make_shared<wcn_ui::Container>();
        auto& ts = tile->style_mut();
        ts.display = taffy::Display::Flex; ts.flex_direction = taffy::FlexDirection::Column;
        ts.flex_grow = 1; ts.flex_basis = taffy::Dimension::length(0);
        ts.size = {taffy::Dimension::auto_(), taffy::Dimension::length(140)};
        tile->set_background(wcn_ui::Color::hex(colors[i]));
        tile->set_border_radius(10);
        grid->add_child(tile);
    }
    root->add_child(grid);

    root->add_child(make_nav_row(app, {
        {"Home (Ink)", "home"},
        {"Analytics (Liquid)", "analytics"},
        {"Settings (Tear)", "settings"}
    }));
    return root;
}


int main() {
    auto app = std::make_shared<TransitionApp>();
    if (!app->init(1280, 800, "WCN GPU Transition Engine")) {
        std::fprintf(stderr, "init failed\n");
        return 1;
    }
    app->setup_router();
    app->run_with_transitions();
    return 0;
}