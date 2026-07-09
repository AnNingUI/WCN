// text.hpp — Text element (renders a text string via fullstack_compute)
#pragma once
#ifndef WCN_UI_TEXT_HPP
#define WCN_UI_TEXT_HPP

#include "element.hpp"
#include "../taffy.hpp"
#include <string>

namespace wcn_ui {

class Text : public Element {
public:
    explicit Text(std::string text = {})
        : text_(std::move(text)) {}

    void set_text(const std::string& t) { text_ = t; dirty_ = true; request_relayout(); }
    const std::string& text() const { return text_; }
    void set_font_size(float px) { font_size_ = px; dirty_ = true; request_relayout(); }
    float font_size() const { return font_size_; }
    void set_color(Color c) { color_ = c; }
    Color text_color() const { return color_; }
    void set_font_family(const std::string& f) { font_family_ = f; dirty_ = true; request_relayout(); }
    const std::string& font_family() const { return font_family_; }

    void on_create(App* app) override {
        Element::on_create(app);
        style_.display = taffy::Display::Flex;
        style_.flex_direction = taffy::FlexDirection::Row;
        style_.align_items = taffy::AlignItems{taffy::AlignItemsKeyword::Center, taffy::AlignmentSafety::Unsafe};
    }

    void on_layout(App* app) override {
        Element::on_layout(app);
    }

    void on_render(App* app, FS_Core* fs) override;

    // Measure text content for taffy layout (so text nodes get a real size).
    void measure_content(FS_Core* fs, float known_width, float& out_w, float& out_h) override;

    static taffy::Style default_style() {
        taffy::Style s;
        s.display = taffy::Display::Flex;
        s.flex_direction = taffy::FlexDirection::Row;
        s.align_items = taffy::AlignItems{taffy::AlignItemsKeyword::Center, taffy::AlignmentSafety::Unsafe};
        return s;
    }

    taffy::Style& style_mut() { return style_; }
    const taffy::Style& style() const { return style_; }

private:
    std::string text_;
    float font_size_ = 16.0f;
    Color color_ = Color::black();
    std::string font_family_ = "sans-serif";
    bool dirty_ = true;
    taffy::Style style_ = default_style();
};

} // namespace wcn_ui

#endif // WCN_UI_TEXT_HPP
