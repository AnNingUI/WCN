// ui_demo.cpp — WCN UI Library demo
// Demonstrates Element hierarchy, taffy layout, fullstack_compute rendering,
// and automatic window resize adaptation.

#include "ui/app.hpp"
#include <cstdio>
#include <cstdlib>

// ── Custom elements — just set style properties, rendering is automatic ──

class HeaderBar : public wcn_ui::Container {
public:
    HeaderBar() {
        style_.display = taffy::Display::Flex;
        style_.flex_direction = taffy::FlexDirection::Row;
        style_.align_items = taffy::AlignItems{taffy::AlignItemsKeyword::Center, taffy::AlignmentSafety::Unsafe};
        style_.padding = {
            taffy::LengthPercentage::length(16),
            taffy::LengthPercentage::length(16),
            taffy::LengthPercentage::length(12),
            taffy::LengthPercentage::length(12)
        };
        style_.size = {taffy::Dimension::auto_(), taffy::Dimension::length(56)};
        set_background(wcn_ui::Color::hex(0x2C3E50));
    }
};

class Sidebar : public wcn_ui::Container {
public:
    Sidebar() {
        style_.display = taffy::Display::Flex;
        style_.flex_direction = taffy::FlexDirection::Column;
        style_.padding = {
            taffy::LengthPercentage::length(8),
            taffy::LengthPercentage::length(8),
            taffy::LengthPercentage::length(8),
            taffy::LengthPercentage::length(8)
        };
        style_.size = {taffy::Dimension::length(200), taffy::Dimension::auto_()};
        style_.gap = {taffy::LengthPercentage::length(0), taffy::LengthPercentage::length(4)};
        set_background(wcn_ui::Color::hex(0x34495E));
    }
};

class ContentArea : public wcn_ui::Container {
public:
    ContentArea() {
        style_.display = taffy::Display::Flex;
        style_.flex_direction = taffy::FlexDirection::Column;
        style_.padding = {
            taffy::LengthPercentage::length(24),
            taffy::LengthPercentage::length(24),
            taffy::LengthPercentage::length(24),
            taffy::LengthPercentage::length(24)
        };
        style_.gap = {taffy::LengthPercentage::length(0), taffy::LengthPercentage::length(16)};
        set_background(wcn_ui::Color::hex(0xECF0F1));
    }
};

class Card : public wcn_ui::Container {
public:
    Card() {
        style_.display = taffy::Display::Flex;
        style_.flex_direction = taffy::FlexDirection::Column;
        style_.padding = {
            taffy::LengthPercentage::length(16),
            taffy::LengthPercentage::length(16),
            taffy::LengthPercentage::length(16),
            taffy::LengthPercentage::length(16)
        };
        style_.gap = {taffy::LengthPercentage::length(0), taffy::LengthPercentage::length(8)};
        style_.border = {
            taffy::LengthPercentage::length(1),
            taffy::LengthPercentage::length(1),
            taffy::LengthPercentage::length(1),
            taffy::LengthPercentage::length(1)
        };
        set_background(wcn_ui::Color::white());
        set_border_color(wcn_ui::Color::hex(0xBDC3C7));
        set_border_radius(6);
    }
};

// ── Counter element with state ───────────────────────────────────────────
class CounterApp : public wcn_ui::Container {
public:
    CounterApp() {
        style_.display = taffy::Display::Flex;
        style_.flex_direction = taffy::FlexDirection::Column;
        style_.align_items = taffy::AlignItems{taffy::AlignItemsKeyword::Center, taffy::AlignmentSafety::Unsafe};
        style_.gap = {taffy::LengthPercentage::length(0), taffy::LengthPercentage::length(12)};
        style_.padding = {
            taffy::LengthPercentage::length(16),
            taffy::LengthPercentage::length(16),
            taffy::LengthPercentage::length(16),
            taffy::LengthPercentage::length(16)
        };
    }

    void build(wcn_ui::App* app) override {
        Container::build(app);

        auto title = std::make_shared<wcn_ui::Text>("Counter Demo");
        title->set_font_size(24);
        title->set_color(wcn_ui::Color::hex(0x2C3E50));
        add_child(title);

        count_text_ = std::make_shared<wcn_ui::Text>("0");
        count_text_->set_font_size(48);
        count_text_->set_color(wcn_ui::Color::hex(0x4A90D9));
        add_child(count_text_);

        auto row = std::make_shared<wcn_ui::HStack>(12);
        auto dec_btn = std::make_shared<wcn_ui::Button>("- Decrement");
        dec_btn->set_on_click([this](wcn_ui::Button&) { count_--; update_label(); });
        row->add_child(dec_btn);

        auto inc_btn = std::make_shared<wcn_ui::Button>("+ Increment");
        inc_btn->set_on_click([this](wcn_ui::Button&) { count_++; update_label(); });
        row->add_child(inc_btn);

        auto reset_btn = std::make_shared<wcn_ui::Button>("Reset");
        reset_btn->set_on_click([this](wcn_ui::Button&) { count_ = 0; update_label(); });
        row->add_child(reset_btn);

        add_child(row);
    }

    void update_label() {
        if (count_text_) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%d", count_);
            count_text_->set_text(buf);
        }
    }

private:
    int count_ = 0;
    std::shared_ptr<wcn_ui::Text> count_text_;
};

// ── Main ─────────────────────────────────────────────────────────────────
int main() {
    auto app = std::make_shared<wcn_ui::App>();

    if (!app->init(1280, 800, "WCN UI Library Demo")) {
        std::fprintf(stderr, "Failed to initialize App\n");
        return 1;
    }

    auto root = std::make_shared<wcn_ui::Container>();
    {
        auto& rs = root->style_mut();
        rs.display = taffy::Display::Flex;
        rs.flex_direction = taffy::FlexDirection::Column;
        rs.align_items = taffy::AlignItems{taffy::AlignItemsKeyword::Stretch, taffy::AlignmentSafety::Unsafe};
        rs.size = {taffy::Dimension::percent(100), taffy::Dimension::percent(100)};
    }

    // Header bar
    auto header = std::make_shared<HeaderBar>();
    {
        auto title = std::make_shared<wcn_ui::Text>("WCN UI Library");
        title->set_font_size(20);
        title->set_color(wcn_ui::Color::white());
        header->add_child(title);
    }
    root->add_child(header);

    // Body: sidebar + content (flex row, fills remaining vertical space)
    auto body = std::make_shared<wcn_ui::Container>();
    {
        auto& bs = body->style_mut();
        bs.display = taffy::Display::Flex;
        bs.flex_direction = taffy::FlexDirection::Row;
        bs.align_items = taffy::AlignItems{taffy::AlignItemsKeyword::Stretch, taffy::AlignmentSafety::Unsafe};
        bs.size = {taffy::Dimension::percent(100), taffy::Dimension::auto_()};
        bs.flex_grow = 1;
    }

    // Sidebar
    auto sidebar = std::make_shared<Sidebar>();
    {
        const char* items[] = {"Dashboard", "Analytics", "Settings", "Profile", "Help"};
        for (auto& item : items) {
            auto btn = std::make_shared<wcn_ui::Button>(item);
            btn->set_background(wcn_ui::Color::transparent());
            btn->set_border_color(wcn_ui::Color::transparent());
            btn->set_text_color(wcn_ui::Color::hex(0xECF0F1));
            btn->style_mut().justify_content = {
                taffy::AlignContentKeyword::FlexStart, taffy::AlignmentSafety::Unsafe
            };
            btn->style_mut().padding = {
                taffy::LengthPercentage::length(12),
                taffy::LengthPercentage::length(12),
                taffy::LengthPercentage::length(8),
                taffy::LengthPercentage::length(8)
            };
            const char* item_copy = item;
            btn->set_on_click([item_copy](wcn_ui::Button& b) {
                std::printf("Clicked: %s\n", item_copy);
                (void)b;
            });
            sidebar->add_child(btn);
        }
    }
    body->add_child(sidebar);

    // Content area (flex-grow fills remaining horizontal space)
    auto content = std::make_shared<ContentArea>();
    content->style_mut().flex_grow = 1;
    {
        auto page_title = std::make_shared<wcn_ui::Text>("Welcome to WCN UI");
        page_title->set_font_size(28);
        page_title->set_color(wcn_ui::Color::hex(0x2C3E50));
        content->add_child(page_title);

        auto desc = std::make_shared<wcn_ui::Text>(
            "A C++ UI library built on taffy layout + fullstack_compute GPU rendering. "
            "Resize the window — the Flexbox layout adapts automatically."
        );
        desc->set_font_size(14);
        desc->set_color(wcn_ui::Color::hex(0x7F8C8D));
        content->add_child(desc);

        auto cards_row = std::make_shared<wcn_ui::HStack>(16);
        cards_row->style_mut().size = {taffy::Dimension::percent(100), taffy::Dimension::auto_()};

        for (int i = 1; i <= 3; i++) {
            auto card = std::make_shared<Card>();
            char title_buf[32];
            std::snprintf(title_buf, sizeof(title_buf), "Card %d", i);
            auto card_title = std::make_shared<wcn_ui::Text>(title_buf);
            card_title->set_font_size(18);
            card_title->set_color(wcn_ui::Color::hex(0x2C3E50));
            card->add_child(card_title);

            auto card_desc = std::make_shared<wcn_ui::Text>(
                "This card uses flex-grow to share available space. "
                "Resize the window to see it adapt."
            );
            card_desc->set_font_size(13);
            card_desc->set_color(wcn_ui::Color::hex(0x95A5A6));
            card->add_child(card_desc);

            card->style_mut().flex_grow = 1;
            card->style_mut().flex_basis = taffy::Dimension::length(0);
            cards_row->add_child(card);
        }
        content->add_child(cards_row);

        auto counter = std::make_shared<CounterApp>();
        {
            auto& cs = counter->style_mut();
            cs.size = {taffy::Dimension::auto_(), taffy::Dimension::auto_()};
            cs.align_self = taffy::AlignSelf{taffy::AlignItemsKeyword::Center, taffy::AlignmentSafety::Unsafe};
        }
        content->add_child(counter);
    }
    body->add_child(content);

    root->add_child(body);

    app->set_root(root);

    app->set_on_resize([](uint32_t w, uint32_t h) {
        std::printf("Window resized: %u x %u\n", w, h);
    });

    app->run();
    return 0;
}
