#include "fullstack_glfw_app_backend.h"
#include "fullstack_render.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static FS_ColorSpace output_color_space(WGPUTextureFormat format) {
    return format == WGPUTextureFormat_BGRA8UnormSrgb ||
           format == WGPUTextureFormat_RGBA8UnormSrgb
        ? FS_COLOR_SPACE_SRGB : FS_COLOR_SPACE_LINEAR_SRGB;
}

int main(void) {
    FS_Error error = FS_ERROR_INIT;
    if (fs_app_register_glfw_backend(&error) != FS_RESULT_OK) return 1;
    FS_AppDesc app_desc = FS_APP_DESC_INIT;
    app_desc.backend_name = "glfw";
    app_desc.create_default_window = true;
    app_desc.default_window.title = "WCN generic GLFW backend";
    app_desc.default_window.width = 960;
    app_desc.default_window.height = 600;
    FS_App* app = NULL;
    if (fs_app_create(&app_desc, &app, &error) != FS_RESULT_OK) {
        fprintf(stderr, "%s\n", error.message);
        return 1;
    }
    FS_AppWindow* window = fs_app_window_at(app, 0);
    FS_RenderContext* render = NULL;
    bool running = true;
    float phase = 0.0f;
    uint32_t frame_count = 0;
    const char* max_frames_text = getenv("FS_DEMO_MAX_FRAMES");
    uint32_t max_frames = max_frames_text
        ? (uint32_t)strtoul(max_frames_text, NULL, 10) : 0;
    while (running) {
        if (fs_app_begin_frame(app, &error) != FS_RESULT_OK) break;
        if (fs_app_pump_events(app, FS_APP_PUMP_POLL, 0, &error) != FS_RESULT_OK)
            break;
        FS_AppEvent event = FS_APP_EVENT_INIT;
        while (fs_app_poll_event(app, &event, &error) == FS_RESULT_OK) {
            if (event.type == FS_APP_EVENT_QUIT_REQUESTED) running = false;
            fs_app_event_release(&event);
        }
        FS_AppFrame app_frame = FS_APP_FRAME_INIT;
        FS_Result acquire = fs_app_window_acquire_frame(window, &app_frame, &error);
        if (acquire == FS_RESULT_OK) {
            if (!render) {
                FS_RenderContextDesc desc = FS_RENDER_CONTEXT_DESC_INIT;
                desc.gpu = fs_app_gpu(app);
                desc.width = app_frame.width;
                desc.height = app_frame.height;
                desc.output_format = app_frame.format;
                desc.output_color_space = output_color_space(app_frame.format);
                if (fs_render_context_create(&desc, &render, &error) != FS_RESULT_OK)
                    running = false;
            }
            if (render) {
                FS_Core* core = fs_render_context_core(render);
                fs_core_begin_commands(core);
                float cx = app_frame.width * 0.5f;
                float cy = app_frame.height * 0.5f;
                float radius = 60.0f + 24.0f * sinf(phase);
                (void)fs_cmd_rect(core, 0, 0, (float)app_frame.width,
                                  (float)app_frame.height, 0, 0x111318ffu);
                (void)fs_cmd_circle(core, cx, cy, radius, 0x8ab4f8ffu);
                FS_RenderTargetDesc target_desc = FS_RENDER_TARGET_DESC_INIT;
                target_desc.texture = app_frame.texture;
                target_desc.view = app_frame.view;
                target_desc.format = app_frame.format;
                target_desc.width = app_frame.width;
                target_desc.height = app_frame.height;
                target_desc.color_space = output_color_space(app_frame.format);
                target_desc.generation = app_frame.surface_generation;
                FS_RenderTarget* target = NULL;
                FS_RenderFrame* frame = NULL;
                FS_CommandBatch batch = {0};
                if (fs_render_target_import(&target_desc, &target, &error) ==
                        FS_RESULT_OK &&
                    fs_render_context_begin_frame(render, target, &frame, &error) ==
                        FS_RESULT_OK &&
                    fs_render_frame_encode(frame, 0, 0, 0, 1, &batch, &error) ==
                        FS_RESULT_OK &&
                    fs_command_batch_submit(render, &batch, &error) == FS_RESULT_OK &&
                    fs_app_frame_mark_submitted(window, &app_frame,
                                                batch.submission, &error) ==
                        FS_RESULT_OK) {
                    (void)fs_app_frame_present(window, &app_frame, &error);
                } else {
                    (void)fs_app_frame_cancel(window, &app_frame, NULL);
                }
                fs_command_batch_release(&batch);
                fs_render_target_destroy(target);
            } else {
                (void)fs_app_frame_cancel(window, &app_frame, NULL);
            }
        }
        if (fs_app_end_frame(app, &error) != FS_RESULT_OK) break;
        phase += 0.04f;
        frame_count++;
        if (max_frames && frame_count >= max_frames) running = false;
    }
    fs_render_context_destroy(render);
    (void)fs_app_begin_destroy(app, NULL);
    fs_app_release(app);
    (void)fs_app_backend_unregister("glfw", NULL);
    return 0;
}
