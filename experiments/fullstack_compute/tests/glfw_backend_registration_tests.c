#include "fullstack_glfw_app_backend.h"

#include <assert.h>

int main(void) {
    FS_Error error = FS_ERROR_INIT;
    assert(fs_app_register_glfw_backend(&error) == FS_RESULT_OK);
    const FS_AppBackendFactory* factory = fs_app_backend_find("glfw");
    assert(factory && factory->ops);
    assert(factory->abi_version == FS_APP_BACKEND_ABI_VERSION);
    assert(factory->ops->acquire_frame && factory->ops->present_frame &&
           factory->ops->cancel_frame);
    assert(fs_app_backend_unregister("glfw", &error) == FS_RESULT_OK);
    return 0;
}
