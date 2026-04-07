/**
 * clay_layout_demo.c — Clay layout engine × WCN showcase
 *
 * Rebuilt to mirror the component structure and interaction style of the
 * zig-build-test Clay demo while keeping the WCN backend and render bridge.
 */

#include "../impl/fullstack_glfw_backend.h"
#include "../impl/fullstack_stb_font_backend.h"
#include "../impl/fullstack_stb_image_backend.h"
#include "fullstack_clay.h"
#include "clay.h"

#include "lato_regular_font_embedded.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#endif

static const Clay_Color C_PRIMARY = {70, 130, 180, 255};
static const Clay_Color C_SECONDARY = {106, 90, 205, 255};
static const Clay_Color C_ACCENT = {220, 20, 60, 255};
static const Clay_Color C_BG = {245, 245, 245, 255};
static const Clay_Color C_CARD = {255, 255, 255, 255};
static const Clay_Color C_TEXT = {51, 51, 51, 255};
static const Clay_Color C_BORDER = {200, 200, 200, 255};

static FS_Core* g_measure_core = NULL;

static Clay_Dimensions fsclay_measure_text(
    Clay_StringSlice text,
    Clay_TextElementConfig* config,
    void* userData
) {
    (void)userData;
    if (!g_measure_core || !text.chars || text.length <= 0) {
        return (Clay_Dimensions){0, 0};
    }

    size_t copy_len = (size_t)text.length < 4095 ? (size_t)text.length : 4095;
    char buf[4096];
    memcpy(buf, text.chars, copy_len);
    buf[copy_len] = '\0';

    FS_TextMetrics metrics = {0};
    float font_size = (float)(config ? config->fontSize : 14);
    if (!fs_measure_text_utf8(g_measure_core, font_size, buf, 0.0f, &metrics)) {
        float w = (float)text.length * font_size * 0.6f;
        return (Clay_Dimensions){w, font_size};
    }

    return (Clay_Dimensions){
        metrics.width,
        metrics.em_height_ascent + metrics.em_height_descent
    };
}

static void clay_error_handler(Clay_ErrorData error) {
    fprintf(
        stderr,
        "Clay Error: %.*s\n",
        (int)error.errorText.length,
        error.errorText.chars ? error.errorText.chars : ""
    );
}

static void SetConsoleUTF8(void) {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

typedef void (*ButtonClickListener)(void);

typedef struct {
    Clay_String text;
    Clay_Color backgroundColor;
    Clay_ElementId buttonId;
    ButtonClickListener on_click;
} ButtonData;

static bool sidebarVisible = false;
static float sidebarOffset = -300.0f;
static float previousTime = 0.0f;
static float deltaTime = 0.0f;
static char g_width_card_text[128];
static FS_ImageHandle g_demo_image_handle = {0};
static bool g_demo_image_ready = false;

static const uint8_t k_sample_tga_2x2[] = {
    0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x02, 0x00, 0x18, 0x00,
    0x00, 0x00, 0xFF, 0x00, 0xFF, 0x00,
    0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF
};

static const Clay_TextElementConfig TXT_HEADER_TITLE = {
    .fontId = 0,
    .fontSize = 28,
    .textColor = {255, 255, 255, 255}
};

static const Clay_TextElementConfig TXT_BUTTON = {
    .fontId = 0,
    .fontSize = 14,
    .textColor = {255, 255, 255, 255}
};

static const Clay_TextElementConfig TXT_SECTION_TITLE = {
    .fontId = 0,
    .fontSize = 20,
    .textColor = {70, 130, 180, 255}
};

static const Clay_TextElementConfig TXT_CARD_TITLE = {
    .fontId = 0,
    .fontSize = 18,
    .textColor = {70, 130, 180, 255}
};

static const Clay_TextElementConfig TXT_BODY = {
    .fontId = 0,
    .fontSize = 14,
    .textColor = {51, 51, 51, 255}
};

static const Clay_TextElementConfig TXT_NAV_ACTIVE = {
    .fontId = 0,
    .fontSize = 14,
    .textColor = {255, 255, 255, 255}
};

static Clay_Color DarkenColor(Clay_Color color, float factor) {
    Clay_Color darkColor;
    darkColor.r = color.r * factor;
    darkColor.g = color.g * factor;
    darkColor.b = color.b * factor;
    darkColor.a = color.a;
    return darkColor;
}

static float GetCurrentTimeInSeconds(void) {
#if defined(_WIN32)
    static LARGE_INTEGER frequency = {0};
    static int frequency_initialized = 0;
    if (!frequency_initialized) {
        QueryPerformanceFrequency(&frequency);
        frequency_initialized = 1;
    }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (float)now.QuadPart / (float)frequency.QuadPart;
#else
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (float)now.tv_sec + (float)now.tv_nsec / 1000000000.0f;
#endif
}

static float GetDeltaTime(void) {
    float currentTime = GetCurrentTimeInSeconds();
    deltaTime = currentTime - previousTime;
    previousTime = currentTime;
    return deltaTime;
}

static void ToggleSidebar(void) {
    sidebarVisible = !sidebarVisible;
}

static void OnRendererButtonClick(void) {
    fprintf(stderr, "[Clay x WCN] Renderer button pressed\n");
}

static void OnEffectsButtonClick(void) {
    fprintf(stderr, "[Clay x WCN] Effects button pressed\n");
}

static void OnLayoutButtonClick(void) {
    fprintf(stderr, "[Clay x WCN] Layout button pressed\n");
}

static void CardComponent(Clay_String title, Clay_String content) {
    CLAY_AUTO_ID({
        .layout = {
            .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(150, 220)},
            .padding = CLAY_PADDING_ALL(20),
            .childGap = 12,
            .layoutDirection = CLAY_TOP_TO_BOTTOM,
        },
        .backgroundColor = C_CARD,
        .cornerRadius = CLAY_CORNER_RADIUS(10),
        .border = {.color = C_BORDER, .width = CLAY_BORDER_OUTSIDE(1)}
    }) {
        CLAY_AUTO_ID({
            .layout = {
                .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(24)}
            }
        }) {
            CLAY_TEXT(title, TXT_CARD_TITLE);
        }

        CLAY_AUTO_ID({
            .layout = {
                .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0, 0)}
            }
        }) {
            CLAY_TEXT(content, TXT_BODY);
        }
    }
}

static void ButtonComponent(ButtonData* data) {
    Clay_String text = data->text;
    Clay_Color backgroundColor = data->backgroundColor;
    Clay_ElementId buttonId = data->buttonId;
    ButtonClickListener on_click = data->on_click;

    static bool isClicked = false;

    Clay_Context* context = Clay_GetCurrentContext();
    Clay_PointerDataInteractionState state = context->pointerInfo.state;
    bool isHovered = Clay_PointerOver(buttonId);
    bool isPressed =
        isHovered &&
        (state == CLAY_POINTER_DATA_PRESSED ||
         state == CLAY_POINTER_DATA_PRESSED_THIS_FRAME);

    if (state == CLAY_POINTER_DATA_RELEASED_THIS_FRAME ||
        state == CLAY_POINTER_DATA_RELEASED) {
        isClicked = false;
    }

    Clay_Color buttonColor =
        isPressed ? DarkenColor(backgroundColor, 0.7f) : backgroundColor;

    if (isHovered &&
        state == CLAY_POINTER_DATA_PRESSED_THIS_FRAME &&
        on_click != NULL &&
        !isClicked) {
        on_click();
        isClicked = true;
    }

    CLAY(buttonId, {
        .layout = {
            .sizing = {CLAY_SIZING_FIXED(120), CLAY_SIZING_FIXED(40)},
            .padding = CLAY_PADDING_ALL(10),
            .childAlignment = {CLAY_ALIGN_X_CENTER, CLAY_ALIGN_Y_CENTER}
        },
        .backgroundColor = buttonColor,
        .cornerRadius = CLAY_CORNER_RADIUS(6)
    }) {
        CLAY_TEXT(text, TXT_BUTTON);
    }
}

static void HeaderComponent(Clay_String title) {
    CLAY_AUTO_ID({
        .floating = {
            .attachTo = CLAY_ATTACH_TO_ROOT,
        },
        .layout = {
            .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(80)},
            .padding = {20, 20, 0, 0},
            .childAlignment = {CLAY_ALIGN_X_LEFT, CLAY_ALIGN_Y_CENTER},
            .layoutDirection = CLAY_LEFT_TO_RIGHT,
            .childGap = 20,
        },
        .backgroundColor = C_PRIMARY
    }) {
        ButtonData menuButton = {
            .text = CLAY_STRING("☰"),
            .backgroundColor = C_ACCENT,
            .buttonId = CLAY_ID("MenuButton"),
            .on_click = ToggleSidebar,
        };
        ButtonComponent(&menuButton);
        CLAY_TEXT(title, TXT_HEADER_TITLE);
    }

    CLAY_AUTO_ID({
        .layout = {
            .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(80)}
        }
    }) {}
}

static void AnimatedSidebar(void) {
    float targetOffset = sidebarVisible ? 0.0f : -300.0f;
    float animationSpeed = 8.0f;
    sidebarOffset += (targetOffset - sidebarOffset) * animationSpeed * deltaTime;

    CLAY(CLAY_ID("AnimatedSidebar"), {
        .floating = {
            .attachTo = CLAY_ATTACH_TO_ROOT,
            .offset = {.x = sidebarOffset, .y = 80},
            .zIndex = 100,
        },
        .layout = {
            .layoutDirection = CLAY_TOP_TO_BOTTOM,
            .sizing = {CLAY_SIZING_FIXED(300), CLAY_SIZING_GROW(0)},
            .padding = CLAY_PADDING_ALL(16),
            .childGap = 16,
        },
        .backgroundColor = {255, 255, 255, 215},
        .border = {.color = C_BORDER, .width = {0, 1, 1, 0}}
    }) {
        CLAY_TEXT(CLAY_STRING("WCN Sections"), TXT_CARD_TITLE);

        static const char* sidebarItems[] = {
            "Overview",
            "Pipeline",
            "Effects",
            "Text",
            "Layout",
            "Diagnostics",
        };

        for (int i = 0; i < 6; i++) {
            Clay_String itemText = {
                .isStaticallyAllocated = false,
                .length = (int32_t)strlen(sidebarItems[i]),
                .chars = sidebarItems[i],
            };

            CLAY_AUTO_ID({
                .layout = {
                    .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(40)},
                    .padding = CLAY_PADDING_ALL(10),
                },
                .backgroundColor = i == 0 ? C_PRIMARY : C_BG,
                .cornerRadius = CLAY_CORNER_RADIUS(5)
            }) {
                CLAY_TEXT(itemText, i == 0 ? TXT_NAV_ACTIVE : TXT_BODY);
            }
        }
    }
}

static void ResponsiveCardGrid(void) {
    Clay_Context* context = Clay_GetCurrentContext();
    int windowWidth = (int)context->layoutDimensions.width;
    Clay_LayoutDirection direction =
        (windowWidth < 768) ? CLAY_TOP_TO_BOTTOM : CLAY_LEFT_TO_RIGHT;
    uint16_t gap = (windowWidth < 768) ? 15 : 20;
    Clay_SizingAxis cardWidth =
        (windowWidth < 768) ? CLAY_SIZING_GROW(0) : CLAY_SIZING_PERCENT(0.5f);
    Clay_SizingAxis gridHeight =
        (windowWidth < 768) ? CLAY_SIZING_GROW(0) : CLAY_SIZING_FIXED(180);

    snprintf(
        g_width_card_text,
        sizeof(g_width_card_text),
        "Viewport width: %d px — %s card flow is active.",
        windowWidth,
        direction == CLAY_LEFT_TO_RIGHT ? "horizontal" : "vertical"
    );

    Clay_String widthString = {
        .isStaticallyAllocated = false,
        .length = (int32_t)strlen(g_width_card_text),
        .chars = g_width_card_text,
    };

    CLAY_AUTO_ID({
        .layout = {
            .sizing = {CLAY_SIZING_GROW(0), gridHeight},
            .layoutDirection = direction,
            .childGap = gap,
        }
    }) {
        CLAY_AUTO_ID({
            .layout = {
                .sizing = {cardWidth, CLAY_SIZING_GROW(0)}
            }
        }) {
            CardComponent(CLAY_STRING("Responsive Layout"), widthString);
        }

        CLAY_AUTO_ID({
            .layout = {
                .sizing = {cardWidth, CLAY_SIZING_GROW(0)}
            }
        }) {
            CardComponent(
                CLAY_STRING("WCN Renderer"),
                CLAY_STRING("Clay render commands are measured, emitted, and drawn through the existing fullstack_clay bridge into the WCN pipeline.")
            );
        }
    }
}

static void FeatureItem(Clay_String text) {
    CLAY_AUTO_ID({
        .layout = {
            .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(20, 48)},
            .layoutDirection = CLAY_LEFT_TO_RIGHT,
            .childGap = 10,
            .childAlignment = {CLAY_ALIGN_X_LEFT, CLAY_ALIGN_Y_CENTER},
        }
    }) {
        CLAY_AUTO_ID({
            .layout = {
                .sizing = {CLAY_SIZING_FIXED(10), CLAY_SIZING_FIXED(10)}
            },
            .backgroundColor = C_ACCENT,
            .cornerRadius = CLAY_CORNER_RADIUS(5)
        }) {}

        CLAY_AUTO_ID({
            .layout = {
                .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0, 0)}
            }
        }) {
            CLAY_TEXT(text, TXT_BODY);
        }
    }
}

static void ImageTestCard(void) {
    CLAY_AUTO_ID({
        .layout = {
            .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(180, 260)},
            .padding = CLAY_PADDING_ALL(20),
            .childGap = 12,
            .layoutDirection = CLAY_TOP_TO_BOTTOM,
        },
        .backgroundColor = C_CARD,
        .cornerRadius = CLAY_CORNER_RADIUS(10),
        .border = {.color = C_BORDER, .width = CLAY_BORDER_OUTSIDE(1)}
    }) {
        CLAY_AUTO_ID({
            .layout = {
                .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(24)}
            }
        }) {
            CLAY_TEXT(CLAY_STRING("Embedded Image"), TXT_CARD_TITLE);
        }

        CLAY_AUTO_ID({
            .layout = {
                .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(120)},
                .childAlignment = {CLAY_ALIGN_X_CENTER, CLAY_ALIGN_Y_CENTER},
            },
            .backgroundColor = C_BG,
            .cornerRadius = CLAY_CORNER_RADIUS(8)
        }) {
            if (g_demo_image_ready) {
                CLAY_AUTO_ID({
                    .layout = {
                        .sizing = {CLAY_SIZING_FIXED(120), CLAY_SIZING_FIXED(120)}
                    },
                    .aspectRatio = { .aspectRatio = 1.0f },
                    .image = { .imageData = &g_demo_image_handle },
                    .cornerRadius = CLAY_CORNER_RADIUS(8)
                }) {}
            } else {
                CLAY_TEXT(CLAY_STRING("Image upload failed"), TXT_BODY);
            }
        }

        CLAY_AUTO_ID({
            .layout = {
                .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0, 0)}
            }
        }) {
            CLAY_TEXT(CLAY_STRING("Testing Clay image elements through imageData -> FS_ImageHandle rendering."), TXT_BODY);
        }
    }
}

int main(void) {
    SetConsoleUTF8();

    FS_GlfwBackend backend;
    if (!fs_glfw_backend_init(&backend, 1024, 768, "Clay x WCN")) {
        fprintf(stderr, "Failed to init backend\n");
        return 1;
    }

    FS_Core* core = fs_glfw_backend_core(&backend);
    g_measure_core = core;

    if (!fs_core_set_font_backend(core, fs_get_stb_font_backend())) {
        fprintf(stderr, "Warning: stb-font backend unavailable\n");
    }
    fs_core_set_image_backend(core, fs_get_stb_image_backend());

    bool font_ready = false;
    if (sizeof(k_font_lato_regular_data) > 0) {
        font_ready = fs_core_load_font_memory(
            core,
            (const uint8_t*)k_font_lato_regular_data,
            sizeof(k_font_lato_regular_data)
        );
        if (font_ready) {
            fprintf(stderr, "Font loaded OK\n");
        }
    }

    if (!font_ready) {
        const char* paths[] = {
            "assets/NotoSerifSC-VF.ttf",
            "../assets/NotoSerifSC-VF.ttf",
            "../../assets/NotoSerifSC-VF.ttf",
            ".lookme/use.gpu/public/fonts/Lato-Regular.ttf",
        };
        for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); i++) {
            if (fs_core_load_font_file(core, paths[i])) {
                font_ready = true;
                break;
            }
        }
    }

    if (!font_ready) {
        fprintf(stderr, "Warning: no font loaded\n");
    }

    g_demo_image_ready = fs_core_decode_image_memory(
        core,
        k_sample_tga_2x2,
        sizeof(k_sample_tga_2x2),
        &g_demo_image_handle
    );
    if (!g_demo_image_ready) {
        fprintf(stderr, "Warning: embedded image decode/upload failed\n");
    }

    uint32_t clay_mem_size = Clay_MinMemorySize();
    void* clay_mem = calloc(1, clay_mem_size);
    if (!clay_mem) {
        fs_glfw_backend_shutdown(&backend);
        return 1;
    }

    Clay_Arena arena = Clay_CreateArenaWithCapacityAndMemory(clay_mem_size, clay_mem);
    Clay_Context* clay_ctx = Clay_Initialize(
        arena,
        (Clay_Dimensions){1024, 768},
        (Clay_ErrorHandler){clay_error_handler, NULL}
    );
    if (!clay_ctx) {
        fprintf(stderr, "Clay_Initialize FAILED\n");
        free(clay_mem);
        fs_glfw_backend_shutdown(&backend);
        return 1;
    }

    Clay_SetMaxElementCount(8192);
    fsclay_init(fsclay_measure_text, NULL);

    previousTime = GetCurrentTimeInSeconds();

    while (!fs_glfw_backend_should_close(&backend)) {
        fs_glfw_backend_poll_events();

        int win_w = 0;
        int win_h = 0;
        glfwGetWindowSize(backend.window, &win_w, &win_h);

        float scroll_x = 0.0f;
        float scroll_y = 0.0f;
        fs_glfw_backend_take_scroll_delta(&backend, &scroll_x, &scroll_y);

        double mx = 0.0;
        double my = 0.0;
        glfwGetCursorPos(backend.window, &mx, &my);
        int lmb = glfwGetMouseButton(backend.window, GLFW_MOUSE_BUTTON_LEFT);

        Clay_SetPointerState(
            (Clay_Vector2){(float)mx, (float)my},
            lmb == GLFW_PRESS
        );

        deltaTime = GetDeltaTime();
        Clay_UpdateScrollContainers(true, (Clay_Vector2){scroll_x, scroll_y}, deltaTime);

        fs_core_begin_commands(core);
        fs_transform_reset(core);
        fs_style_reset(core);
        fs_path_begin(core);

        float bw = (float)win_w;
        float bh = (float)win_h;
        fs_cmd_rect(core, 0.0f, 0.0f, bw, bh, 0.0f, fsclay_color(C_BG));

        Clay_SetLayoutDimensions((Clay_Dimensions){bw, bh});
        Clay_BeginLayout();

        CLAY(CLAY_ID("MainContainer"), {
            .layout = {
                .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0)},
                .layoutDirection = CLAY_TOP_TO_BOTTOM,
            },
            .backgroundColor = C_BG
        }) {
            HeaderComponent(CLAY_STRING("Clay x WCN Demo"));
            AnimatedSidebar();

            CLAY_AUTO_ID({
                .layout = {
                    .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0)},
                    .padding = {20, 20, 20, 20},
                    .childGap = 20,
                    .layoutDirection = CLAY_TOP_TO_BOTTOM,
                }
            }) {
                ResponsiveCardGrid();

                CLAY_AUTO_ID({
                    .layout = {
                        .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(60)},
                        .layoutDirection = CLAY_LEFT_TO_RIGHT,
                        .childGap = 15,
                        .childAlignment = {CLAY_ALIGN_X_LEFT, CLAY_ALIGN_Y_CENTER},
                    }
                }) {
                    ButtonComponent(&(ButtonData){
                        .text = CLAY_STRING("Renderer"),
                        .backgroundColor = C_PRIMARY,
                        .buttonId = CLAY_ID("RendererButton"),
                        .on_click = OnRendererButtonClick,
                    });
                    ButtonComponent(&(ButtonData){
                        .text = CLAY_STRING("Effects"),
                        .backgroundColor = C_SECONDARY,
                        .buttonId = CLAY_ID("EffectsButton"),
                        .on_click = OnEffectsButtonClick,
                    });
                    ButtonComponent(&(ButtonData){
                        .text = CLAY_STRING("Layout"),
                        .backgroundColor = C_ACCENT,
                        .buttonId = CLAY_ID("LayoutButton"),
                        .on_click = OnLayoutButtonClick,
                    });
                }

                Clay_Context* context = Clay_GetCurrentContext();
                int currentWidth = (int)context->layoutDimensions.width;
                bool stackedLayout = currentWidth < 768;

                CLAY_AUTO_ID({
                    .layout = {
                        .sizing = {
                            CLAY_SIZING_GROW(0),
                            stackedLayout ? CLAY_SIZING_FIT(0, 0) : CLAY_SIZING_GROW(0)
                        },
                        .padding = {20, 20, 20, 20},
                        .childGap = 15,
                        .layoutDirection = CLAY_TOP_TO_BOTTOM,
                    },
                    .backgroundColor = C_CARD,
                    .cornerRadius = CLAY_CORNER_RADIUS(10),
                    .border = {.color = C_BORDER, .width = CLAY_BORDER_OUTSIDE(1)}
                }) {
                    CLAY_AUTO_ID({
                        .layout = {
                            .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(30)}
                        }
                    }) {
                        CLAY_TEXT(CLAY_STRING("WCN Features"), TXT_SECTION_TITLE);
                    }

                    Clay_LayoutDirection featureLayoutDirection =
                        stackedLayout ? CLAY_TOP_TO_BOTTOM : CLAY_LEFT_TO_RIGHT;

                    CLAY_AUTO_ID({
                        .layout = {
                            .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0)},
                            .layoutDirection = featureLayoutDirection,
                            .childGap = 15,
                        }
                    }) {
                        CLAY_AUTO_ID({
                            .layout = {
                                .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0)},
                                .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                .childGap = 12,
                            }
                        }) {
                            FeatureItem(CLAY_STRING("Clay 0.14 declarations stay aligned with the local header semantics."));
                            FeatureItem(CLAY_STRING("GLFW pointer input feeds the same button interaction flow as the reference demo."));
                        }

                        CLAY_AUTO_ID({
                            .layout = {
                                .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0)},
                                .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                .childGap = 12,
                            }
                        }) {
                            FeatureItem(CLAY_STRING("Render commands are bridged into WCN through fullstack_clay."));
                            FeatureItem(CLAY_STRING("Sidebar toggle and card breakpoint behavior mirror zig-build-test."));
                        }
                    }

                    ImageTestCard();
                }
            }
        }

        Clay_RenderCommandArray rcmds = Clay_EndLayout(0.0f);
        fsclay_render_commands(core, &rcmds);

        fs_glfw_backend_present(
            &backend,
            (float)C_BG.r / 255.0f,
            (float)C_BG.g / 255.0f,
            (float)C_BG.b / 255.0f,
            1.0f
        );
    }

    free(clay_mem);
    fsclay_shutdown();
    fs_glfw_backend_shutdown(&backend);
    return 0;
}
