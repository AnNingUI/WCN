// button.hpp — Interactive Button element
#pragma once
#ifndef WCN_UI_BUTTON_HPP
#define WCN_UI_BUTTON_HPP

#include "element.hpp"
#include "container.hpp"
#include "../taffy.hpp"
#include <functional>
#include <string>

namespace wcn_ui {

class Button : public Element {
public:
    using ClickCallback = std::function<void(Button&)>;

    explicit Button(std::string label = {})
        : label_(std::move(label)) {
        style_.display = taffy::Display::Flex;
        style_.flex_direction = taffy::FlexDirection::Row;
        style_.align_items = taffy::AlignItems{taffy::AlignItemsKeyword::Center, taffy::AlignmentSafety::Unsafe};
        style_.justify_content = taffy::JustifyContent{taffy::AlignContentKeyword::Center, taffy::AlignmentSafety::Unsafe};
        style_.padding = {
            taffy::LengthPercentage::length(12),
            taffy::LengthPercentage::length(12),
            taffy::LengthPercentage::length(6),
            taffy::LengthPercentage::length(6)
        };
        style_.border = {
            taffy::LengthPercentage::length(1),
            taffy::LengthPercentage::length(1),
            taffy::LengthPercentage::length(1),
            taffy::LengthPercentage::length(1)
        };
        set_background(Color::hex(0x4A90D9));
        set_border_color(Color::hex(0x357ABD));
        set_border_radius(4);
    }

    void set_label(const std::string& l) { label_ = l; request_relayout(); }
    const std::string& label() const { return label_; }

    void set_font_size(float px) { font_size_ = px; request_relayout(); }
    float font_size() const { return font_size_; }

    void set_text_color(Color c) { text_color_ = c; }
    Color text_color() const { return text_color_; }

    void set_on_click(ClickCallback cb) { on_click_ = std::move(cb); }

    // ── State ───────────────────────────────────────────────────────────
    bool is_hovered() const { return hovered_; }
    bool is_pressed() const { return pressed_; }

    // Override background() so render_element picks up hover/press colors
    Color background() const override {
        if (pressed_)  return Color::hex(0x2A6FB0);
        if (hovered_)  return Color::hex(0x5BA0E8);
        return bg_color_;
    }

    bool on_event(const Event& e) override {
        switch (e.type) {
            case EventType::MouseEnter:
                hovered_ = true;
                return true;
            case EventType::MouseLeave:
                hovered_ = false;
                pressed_ = false;
                return true;
            case EventType::MouseDown:
                if (e.mouse.button == 0) {
                    pressed_ = true;
                    return true;
                }
                break;
            case EventType::MouseUp:
                if (e.mouse.button == 0 && pressed_) {
                    pressed_ = false;
                    if (on_click_) on_click_(*this);
                    return true;
                }
                break;
            default: break;
        }
        return false;
    }

    void on_render(App* app, FS_Core* fs) override;

    // Measure label text so the button sizes to fit its content.
    void measure_content(FS_Core* fs, float known_width, float& out_w, float& out_h) override;

    taffy::Style& style_mut() { return style_; }
    const taffy::Style& style() const { return style_; }

private:
    std::string label_;
    float font_size_ = 14.0f;
    Color text_color_ = Color::white();
    ClickCallback on_click_;
    bool hovered_ = false;
    bool pressed_ = false;
    taffy::Style style_;
};

} // namespace wcn_ui

#endif // WCN_UI_BUTTON_HPP
