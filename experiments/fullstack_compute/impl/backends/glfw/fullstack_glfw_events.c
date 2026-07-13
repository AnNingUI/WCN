#include "fullstack_glfw_app_backend_private.h"

#include <string.h>

void fs_glfw_emit(FS_GlfwAppWindow* window, FS_AppEvent* event) {
    if (!window || !event || !window->backend || !window->backend->host) return;
    event->struct_size = sizeof(*event);
    event->window_id = window->stable_id;
    (void)window->backend->host->enqueue_event(
        window->backend->host, event, NULL);
}

static FS_GlfwAppWindow* fs_glfw_callback_window(GLFWwindow* native) {
    return native ? (FS_GlfwAppWindow*)glfwGetWindowUserPointer(native) : NULL;
}

static void fs_glfw_window_size_callback(GLFWwindow* native, int width, int height) {
    FS_GlfwAppWindow* window = fs_glfw_callback_window(native);
    if (!window) return;
    fs_glfw_refresh_metrics(window);
    FS_AppEvent event = FS_APP_EVENT_INIT;
    event.type = FS_APP_EVENT_WINDOW_RESIZED;
    event.data.window.logical_width = width > 0 ? (uint32_t)width : 0;
    event.data.window.logical_height = height > 0 ? (uint32_t)height : 0;
    event.data.window.framebuffer_width = window->metrics.framebuffer_width;
    event.data.window.framebuffer_height = window->metrics.framebuffer_height;
    event.data.window.scale_x = window->metrics.scale_x;
    event.data.window.scale_y = window->metrics.scale_y;
    fs_glfw_emit(window, &event);
    if (window->previous_window_size)
        window->previous_window_size(native, width, height);
}

static void fs_glfw_framebuffer_size_callback(GLFWwindow* native,
                                               int width, int height) {
    FS_GlfwAppWindow* window = fs_glfw_callback_window(native);
    if (!window) return;
    fs_glfw_refresh_metrics(window);
    window->configured = false;
    FS_AppEvent event = FS_APP_EVENT_INIT;
    event.type = FS_APP_EVENT_FRAMEBUFFER_RESIZED;
    event.data.window.logical_width = window->metrics.logical_width;
    event.data.window.logical_height = window->metrics.logical_height;
    event.data.window.framebuffer_width = width > 0 ? (uint32_t)width : 0;
    event.data.window.framebuffer_height = height > 0 ? (uint32_t)height : 0;
    event.data.window.scale_x = window->metrics.scale_x;
    event.data.window.scale_y = window->metrics.scale_y;
    fs_glfw_emit(window, &event);
    if (window->previous_framebuffer_size)
        window->previous_framebuffer_size(native, width, height);
}

static void fs_glfw_content_scale_callback(GLFWwindow* native,
                                            float scale_x, float scale_y) {
    FS_GlfwAppWindow* window = fs_glfw_callback_window(native);
    if (!window) return;
    fs_glfw_refresh_metrics(window);
    FS_AppEvent event = FS_APP_EVENT_INIT;
    event.type = FS_APP_EVENT_SCALE_CHANGED;
    event.data.window.logical_width = window->metrics.logical_width;
    event.data.window.logical_height = window->metrics.logical_height;
    event.data.window.framebuffer_width = window->metrics.framebuffer_width;
    event.data.window.framebuffer_height = window->metrics.framebuffer_height;
    event.data.window.scale_x = scale_x;
    event.data.window.scale_y = scale_y;
    fs_glfw_emit(window, &event);
    if (window->previous_content_scale)
        window->previous_content_scale(native, scale_x, scale_y);
}

static void fs_glfw_cursor_callback(GLFWwindow* native, double x, double y) {
    FS_GlfwAppWindow* window = fs_glfw_callback_window(native);
    if (!window) return;
    FS_AppEvent event = FS_APP_EVENT_INIT;
    event.type = FS_APP_EVENT_POINTER_MOVE;
    event.data.pointer.pointer_id = 1;
    event.data.pointer.logical_x = (float)x;
    event.data.pointer.logical_y = (float)y;
    event.data.pointer.framebuffer_x = (float)x * window->metrics.scale_x;
    event.data.pointer.framebuffer_y = (float)y * window->metrics.scale_y;
    fs_glfw_emit(window, &event);
    if (window->previous_cursor_position)
        window->previous_cursor_position(native, x, y);
}

static void fs_glfw_mouse_button_callback(GLFWwindow* native, int button,
                                           int action, int modifiers) {
    FS_GlfwAppWindow* window = fs_glfw_callback_window(native);
    if (!window) return;
    double x = 0, y = 0;
    glfwGetCursorPos(native, &x, &y);
    FS_AppEvent event = FS_APP_EVENT_INIT;
    event.type = FS_APP_EVENT_POINTER_BUTTON;
    event.data.pointer.pointer_id = 1;
    event.data.pointer.logical_x = (float)x;
    event.data.pointer.logical_y = (float)y;
    event.data.pointer.framebuffer_x = (float)x * window->metrics.scale_x;
    event.data.pointer.framebuffer_y = (float)y * window->metrics.scale_y;
    event.data.pointer.button = button >= 0 ? (uint32_t)button : 0;
    event.data.pointer.action = action == GLFW_PRESS ? 1u : 0u;
    event.data.pointer.modifiers = (uint32_t)modifiers;
    fs_glfw_emit(window, &event);
    if (window->previous_mouse_button)
        window->previous_mouse_button(native, button, action, modifiers);
}

static void fs_glfw_scroll_callback(GLFWwindow* native, double x, double y) {
    FS_GlfwAppWindow* window = fs_glfw_callback_window(native);
    if (!window) return;
    FS_AppEvent event = FS_APP_EVENT_INIT;
    event.type = FS_APP_EVENT_POINTER_SCROLL;
    event.data.scroll.x = (float)x;
    event.data.scroll.y = (float)y;
    fs_glfw_emit(window, &event);
    if (window->previous_scroll) window->previous_scroll(native, x, y);
}

static FS_AppKeyAction fs_glfw_key_action(int action) {
    if (action == GLFW_PRESS) return FS_APP_KEY_PRESS;
    if (action == GLFW_REPEAT) return FS_APP_KEY_REPEAT;
    return FS_APP_KEY_RELEASE;
}

static void fs_glfw_key_callback(GLFWwindow* native, int key, int scancode,
                                 int action, int modifiers) {
    FS_GlfwAppWindow* window = fs_glfw_callback_window(native);
    if (!window) return;
    FS_AppEvent event = FS_APP_EVENT_INIT;
    event.type = FS_APP_EVENT_KEY;
    event.data.key.key = key >= 0 ? (uint32_t)key : 0;
    event.data.key.physical_key = scancode >= 0 ? (uint32_t)scancode : 0;
    event.data.key.action = fs_glfw_key_action(action);
    event.data.key.modifiers = (uint32_t)modifiers;
    fs_glfw_emit(window, &event);
    if (window->previous_key)
        window->previous_key(native, key, scancode, action, modifiers);
}

static uint32_t fs_glfw_utf8_encode(uint32_t codepoint, char output[4]) {
    if (codepoint <= 0x7fu) { output[0] = (char)codepoint; return 1; }
    if (codepoint <= 0x7ffu) {
        output[0] = (char)(0xc0u | (codepoint >> 6));
        output[1] = (char)(0x80u | (codepoint & 0x3fu));
        return 2;
    }
    if (codepoint <= 0xffffu) {
        output[0] = (char)(0xe0u | (codepoint >> 12));
        output[1] = (char)(0x80u | ((codepoint >> 6) & 0x3fu));
        output[2] = (char)(0x80u | (codepoint & 0x3fu));
        return 3;
    }
    output[0] = (char)(0xf0u | (codepoint >> 18));
    output[1] = (char)(0x80u | ((codepoint >> 12) & 0x3fu));
    output[2] = (char)(0x80u | ((codepoint >> 6) & 0x3fu));
    output[3] = (char)(0x80u | (codepoint & 0x3fu));
    return 4;
}

static void fs_glfw_character_callback(GLFWwindow* native, unsigned int codepoint) {
    FS_GlfwAppWindow* window = fs_glfw_callback_window(native);
    if (!window) return;
    char utf8[4];
    uint32_t size = fs_glfw_utf8_encode(codepoint, utf8);
    FS_AppEvent event = FS_APP_EVENT_INIT;
    event.type = FS_APP_EVENT_TEXT_INPUT;
    if (fs_app_event_set_utf8(&event, utf8, size,
                              window->backend->host->allocator, NULL) ==
        FS_RESULT_OK) {
        fs_glfw_emit(window, &event);
        fs_app_event_release(&event);
    }
    if (window->previous_character)
        window->previous_character(native, codepoint);
}

static void fs_glfw_close_callback(GLFWwindow* native) {
    FS_GlfwAppWindow* window = fs_glfw_callback_window(native);
    if (!window) return;
    FS_AppEvent event = FS_APP_EVENT_INIT;
    event.type = FS_APP_EVENT_QUIT_REQUESTED;
    fs_glfw_emit(window, &event);
    if (window->previous_close) window->previous_close(native);
}

void fs_glfw_events_attach(FS_GlfwAppWindow* window) {
    window->previous_user_pointer = glfwGetWindowUserPointer(window->window);
    glfwSetWindowUserPointer(window->window, window);
    window->previous_window_size = glfwSetWindowSizeCallback(
        window->window, fs_glfw_window_size_callback);
    window->previous_framebuffer_size = glfwSetFramebufferSizeCallback(
        window->window, fs_glfw_framebuffer_size_callback);
    window->previous_content_scale = glfwSetWindowContentScaleCallback(
        window->window, fs_glfw_content_scale_callback);
    window->previous_cursor_position = glfwSetCursorPosCallback(
        window->window, fs_glfw_cursor_callback);
    window->previous_mouse_button = glfwSetMouseButtonCallback(
        window->window, fs_glfw_mouse_button_callback);
    window->previous_scroll = glfwSetScrollCallback(
        window->window, fs_glfw_scroll_callback);
    window->previous_key = glfwSetKeyCallback(window->window, fs_glfw_key_callback);
    window->previous_character = glfwSetCharCallback(
        window->window, fs_glfw_character_callback);
    window->previous_close = glfwSetWindowCloseCallback(
        window->window, fs_glfw_close_callback);
}

void fs_glfw_events_detach(FS_GlfwAppWindow* window) {
    if (!window || !window->window) return;
    glfwSetWindowSizeCallback(window->window, window->previous_window_size);
    glfwSetFramebufferSizeCallback(window->window, window->previous_framebuffer_size);
    glfwSetWindowContentScaleCallback(window->window, window->previous_content_scale);
    glfwSetCursorPosCallback(window->window, window->previous_cursor_position);
    glfwSetMouseButtonCallback(window->window, window->previous_mouse_button);
    glfwSetScrollCallback(window->window, window->previous_scroll);
    glfwSetKeyCallback(window->window, window->previous_key);
    glfwSetCharCallback(window->window, window->previous_character);
    glfwSetWindowCloseCallback(window->window, window->previous_close);
    glfwSetWindowUserPointer(window->window, window->previous_user_pointer);
}
