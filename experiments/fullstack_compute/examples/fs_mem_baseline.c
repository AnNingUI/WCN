// fs_mem_baseline.c — 不加载字体，只渲染一个矩形，测量 FS_Core 基线 PrivateWS

#include "../impl/fullstack_glfw_backend.h"
#include "../impl/fullstack_stb_image_backend.h"
#include "../impl/fullstack_freetype2_font_backend.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <windows.h>
#include <psapi.h>

static void print_ws(void) {
    PROCESS_MEMORY_COUNTERS_EX pmc = {0};
    pmc.cb = sizeof(pmc);
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    SIZE_T priv_ws = 0;
    DWORD n = (DWORD)(pmc.WorkingSetSize / 4096) * 2;
    DWORD buf_sz = sizeof(PSAPI_WORKING_SET_INFORMATION) + n * sizeof(PSAPI_WORKING_SET_BLOCK);
    PSAPI_WORKING_SET_INFORMATION* info = (PSAPI_WORKING_SET_INFORMATION*)calloc(1, buf_sz);
    if (info && QueryWorkingSet(GetCurrentProcess(), info, buf_sz)) {
        for (ULONG_PTR i = 0; i < info->NumberOfEntries; i++)
            if (info->WorkingSetInfo[i].Shared == FALSE) priv_ws += 4096;
    }
    free(info);
    printf("[WS] PrivateWS=%.1f MB  WorkingSet=%.1f MB\n",
           priv_ws / (1024.0*1024.0), pmc.WorkingSetSize / (1024.0*1024.0));
}

int main(void) {
    FS_GlfwBackend backend;
    if (!fs_glfw_backend_init(&backend, 1280, 720, "FS Mem Baseline")) {
        fprintf(stderr, "backend init failed\n");
        return 1;
    }
    FS_Core* core = fs_glfw_backend_core(&backend);
    fs_core_set_image_backend(core, fs_get_stb_image_backend());
    // 不加载字体，不设置 font backend
    print_ws();

    int frame = 0;
    while (!glfwWindowShouldClose(backend.window)) {
        glfwPollEvents();
        fs_core_begin_commands(core);
        fs_cmd_rect(core, 100.0f, 100.0f, 400.0f, 300.0f, 0.0f, 0xFF4080FFu);
        fs_cmd_rect_stroke(core, 50.0f, 50.0f, 500.0f, 400.0f, 0.0f, 3.0f, 0xFFFFFFFFu);
        if (!fs_glfw_backend_present(&backend, 0.08f, 0.10f, 0.14f, 1.0f))
            break;
        if (frame == 0) {
            print_ws();
        }
        frame++;
    }
    fs_glfw_backend_shutdown(&backend);
    return 0;
}
