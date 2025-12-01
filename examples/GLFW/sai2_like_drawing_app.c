#include <WCN/WCN.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define WCN_GLFW_IMPLEMENTATION
#include "../../impl/wcn_glfw_impl.h"

// Drawing tool types
typedef enum {
    TOOL_PEN,
    TOOL_BRUSH,
    TOOL_ERASER,
    TOOL_BUCKET
} ToolType;

// Application state
typedef struct {
    ToolType current_tool;
    uint32_t pen_color;
    float pen_size;
    bool is_drawing;
    float last_x, last_y;
    float canvas_width, canvas_height;
} AppState;

// Mouse button callback
static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    AppState* app_state = (AppState*)glfwGetWindowUserPointer(window);
    
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            app_state->is_drawing = true;
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            app_state->last_x = (float)xpos;
            app_state->last_y = (float)ypos;
        } else if (action == GLFW_RELEASE) {
            app_state->is_drawing = false;
        }
    }
}

// Cursor position callback
static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    AppState* app_state = (AppState*)glfwGetWindowUserPointer(window);
    
    if (app_state->is_drawing) {
        // In a real implementation, we would draw a line from (last_x, last_y) to (xpos, ypos)
        // For simplicity, we'll just update the last position
        app_state->last_x = (float)xpos;
        app_state->last_y = (float)ypos;
    }
}

// Key callback for tool switching
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    AppState* app_state = (AppState*)glfwGetWindowUserPointer(window);
    
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        switch (key) {
            case GLFW_KEY_1:
                app_state->current_tool = TOOL_PEN;
                printf("Switched to Pen tool\n");
                break;
            case GLFW_KEY_2:
                app_state->current_tool = TOOL_BRUSH;
                printf("Switched to Brush tool\n");
                break;
            case GLFW_KEY_3:
                app_state->current_tool = TOOL_ERASER;
                printf("Switched to Eraser tool\n");
                break;
            case GLFW_KEY_4:
                app_state->current_tool = TOOL_BUCKET;
                printf("Switched to Bucket tool\n");
                break;
            case GLFW_KEY_R:
                app_state->pen_color = 0xFFFF0000; // Red
                printf("Color changed to Red\n");
                break;
            case GLFW_KEY_G:
                app_state->pen_color = 0xFF00FF00; // Green
                printf("Color changed to Green\n");
                break;
            case GLFW_KEY_B:
                app_state->pen_color = 0xFF0000FF; // Blue
                printf("Color changed to Blue\n");
                break;
            case GLFW_KEY_K:
                app_state->pen_color = 0xFF000000; // Black
                printf("Color changed to Black\n");
                break;
            case GLFW_KEY_EQUAL: // Plus key
                app_state->pen_size += 1.0f;
                if (app_state->pen_size > 50.0f) app_state->pen_size = 50.0f;
                printf("Pen size increased to %.1f\n", app_state->pen_size);
                break;
            case GLFW_KEY_MINUS:
                app_state->pen_size -= 1.0f;
                if (app_state->pen_size < 1.0f) app_state->pen_size = 1.0f;
                printf("Pen size decreased to %.1f\n", app_state->pen_size);
                break;
        }
    }
}

// Draw the UI overlay
static void draw_ui_overlay(WCN_Context* ctx, AppState* app_state, uint32_t width, uint32_t height) {
    // Draw tool indicator
    wcn_save(ctx);
    wcn_set_fill_style(ctx, 0xA0000000); // Semi-transparent black
    wcn_fill_rect(ctx, 10, 10, 200, 100);
    
    wcn_set_fill_style(ctx, 0xFFFFFFFF); // White text
    const char* tool_name = "Unknown";
    switch (app_state->current_tool) {
        case TOOL_PEN: tool_name = "Pen"; break;
        case TOOL_BRUSH: tool_name = "Brush"; break;
        case TOOL_ERASER: tool_name = "Eraser"; break;
        case TOOL_BUCKET: tool_name = "Bucket"; break;
    }
    
    // In a real implementation, we would render text here
    // For now, we'll just draw colored rectangles to represent the UI
    
    // Draw color preview
    wcn_set_fill_style(ctx, app_state->pen_color);
    wcn_fill_rect(ctx, 20, 40, 30, 30);
    
    // Draw size indicator
    wcn_set_fill_style(ctx, 0xFFFFFFFF);
    wcn_fill_rect(ctx, 60, 40, (int)app_state->pen_size, (int)app_state->pen_size);
    
    wcn_restore(ctx);
}

// Draw a simple brush stroke (simplified for this example)
static void draw_brush_stroke(WCN_Context* ctx, float x1, float y1, float x2, float y2, uint32_t color, float size) {
    wcn_save(ctx);
    wcn_set_stroke_style(ctx, color);
    wcn_set_line_width(ctx, size);
    wcn_set_line_cap(ctx, WCN_LINE_CAP_ROUND);
    wcn_set_line_join(ctx, WCN_LINE_JOIN_ROUND);
    
    wcn_begin_path(ctx);
    wcn_move_to(ctx, x1, y1);
    wcn_line_to(ctx, x2, y2);
    wcn_stroke(ctx);
    
    wcn_restore(ctx);
}

int main(void) {
    printf("=== SAI2-like Drawing App ===\n");
    printf("Controls:\n");
    printf("  1 - Pen tool\n");
    printf("  2 - Brush tool\n");
    printf("  3 - Eraser tool\n");
    printf("  4 - Bucket tool\n");
    printf("  R/G/B/K - Change color to Red/Green/Blue/blacK\n");
    printf("  +/- - Increase/decrease pen size\n");
    printf("  Left mouse - Draw\n");
    printf("\n");

    // Create window
    WCN_GLFW_Window* window = wcn_glfw_create_window(1024, 768, "SAI2-like Drawing App");
    if (!window) {
        fprintf(stderr, "Failed to create window\n");
        return -1;
    }

    // Get WCN context
    WCN_Context* ctx = wcn_glfw_get_context(window);
    
    // Set up application state
    AppState app_state = {
        .current_tool = TOOL_PEN,
        .pen_color = 0xFF000000, // Black
        .pen_size = 5.0f,
        .is_drawing = false,
        .last_x = 0,
        .last_y = 0,
        .canvas_width = 1024,
        .canvas_height = 768
    };
    
    // Set GLFW callbacks
    glfwSetWindowUserPointer(window->window, &app_state);
    glfwSetMouseButtonCallback(window->window, mouse_button_callback);
    glfwSetCursorPosCallback(window->window, cursor_position_callback);
    glfwSetKeyCallback(window->window, key_callback);

    printf("Window created successfully\n");
    printf("Starting render loop...\n");

    // Main loop
    int frame_count = 0;
    while (!wcn_glfw_window_should_close(window)) {
        wcn_glfw_poll_events();
        
        // Begin rendering frame
        WCN_GLFW_RenderFrame frame;
        if (wcn_glfw_begin_frame(window, &frame)) {
            uint32_t width, height;
            wcn_glfw_get_size(window, &width, &height);
            
            // Clear canvas with white background
            wcn_set_fill_style(ctx, 0xFFFFFFFF);
            wcn_fill_rect(ctx, 0, 0, (float)width, (float)height);
            
            // Handle drawing
            if (app_state.is_drawing) {
                double xpos, ypos;
                glfwGetCursorPos(window->window, &xpos, &ypos);
                
                // Draw a line from last position to current position
                if (app_state.last_x != 0 || app_state.last_y != 0) {
                    uint32_t color = app_state.pen_color;
                    // If eraser tool, use white color
                    if (app_state.current_tool == TOOL_ERASER) {
                        color = 0xFFFFFFFF;
                    }
                    
                    draw_brush_stroke(ctx, app_state.last_x, app_state.last_y, (float)xpos, (float)ypos, color, app_state.pen_size);
                }
                
                // Update last position
                app_state.last_x = (float)xpos;
                app_state.last_y = (float)ypos;
            }
            
            // Draw UI overlay
            draw_ui_overlay(ctx, &app_state, width, height);
            
            // End rendering frame
            wcn_glfw_end_frame(window, &frame);
        }
        
        frame_count++;
    }

    // Cleanup
    wcn_glfw_destroy_window(window);
    
    printf("Application closed. Rendered %d frames\n", frame_count);
    return 0;
}