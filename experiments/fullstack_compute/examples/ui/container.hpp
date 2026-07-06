// container.hpp — Container element (flex container with children)
#pragma once
#ifndef WCN_UI_CONTAINER_HPP
#define WCN_UI_CONTAINER_HPP

#include "element.hpp"
#include "../taffy.hpp"
#include <algorithm>

namespace wcn_ui {

class Container : public Element {
public:
    Container() = default;
    explicit Container(const taffy::Style& style) : style_(style) {}

    void set_style(const taffy::Style& s) { style_ = s; }
    const taffy::Style& style() const { return style_; }
    taffy::Style& style_mut() { return style_; }

    void on_create(App* app) override {
        Element::on_create(app);
        app_ = app;
    }

    // Build taffy nodes recursively
    void build(App* app) override {
        Element::build(app);

        // Build children first
        for (auto& child : children_) {
            child->build(app);
        }
    }

    void on_layout(App* app) override {
        Element::on_layout(app);
    }

    void on_render(App* app, FS_Core* fs) override;

protected:
    taffy::Style style_;
};

class HStack : public Container {
public:
    explicit HStack(float gap = 0) {
        style_.display = taffy::Display::Flex;
        style_.flex_direction = taffy::FlexDirection::Row;
        style_.gap = {taffy::LengthPercentage::length(gap), taffy::LengthPercentage::length(0)};
    }
};

class VStack : public Container {
public:
    explicit VStack(float gap = 0) {
        style_.display = taffy::Display::Flex;
        style_.flex_direction = taffy::FlexDirection::Column;
        style_.gap = {taffy::LengthPercentage::length(0), taffy::LengthPercentage::length(gap)};
    }
};

class ZStack : public Container {
public:
    ZStack() {
        style_.display = taffy::Display::Flex;
        style_.flex_direction = taffy::FlexDirection::Row;
    }
};

} // namespace wcn_ui

#endif // WCN_UI_CONTAINER_HPP
