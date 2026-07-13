#ifndef FS_BACKEND_CONTRACT_SUITE_H
#define FS_BACKEND_CONTRACT_SUITE_H

#include "fullstack_app.h"
#include <assert.h>

static void fs_backend_contract_basic(const char* backend_name, bool visible) {
    FS_Error error = FS_ERROR_INIT;
    FS_AppDesc desc = FS_APP_DESC_INIT;
    desc.backend_name = backend_name;
    desc.create_default_window = true;
    desc.default_window.width = 160;
    desc.default_window.height = 120;
    desc.default_window.visible = visible;
    FS_App* app = NULL;
    assert(fs_app_create(&desc, &app, &error) == FS_RESULT_OK);
    assert(fs_app_window_count(app) == 1);
    FS_AppWindowDesc second_desc = FS_APP_WINDOW_DESC_INIT;
    second_desc.width = 96;
    second_desc.height = 64;
    second_desc.visible = visible;
    FS_AppWindow* second = NULL;
    assert(fs_app_create_window(app, &second_desc, &second, &error) ==
           FS_RESULT_OK);
    assert(fs_app_window_count(app) == 2);
    assert(fs_app_begin_frame(app, &error) == FS_RESULT_OK);
    assert(fs_app_pump_events(app, FS_APP_PUMP_POLL, 0, &error) == FS_RESULT_OK);
    assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);
    assert(fs_app_destroy_window(app, second, &error) == FS_RESULT_OK);
    assert(fs_app_window_count(app) == 1);
    assert(fs_app_begin_destroy(app, &error) == FS_RESULT_OK);
    fs_app_release(app);
}

#endif
