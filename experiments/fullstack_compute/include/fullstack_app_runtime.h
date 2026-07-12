#ifndef FULLSTACK_APP_RUNTIME_H
#define FULLSTACK_APP_RUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

#include "fullstack_result.h"

typedef void (FS_CALL *FS_AppTaskFn)(void* user_data);
typedef void (FS_CALL *FS_AppTaskDestroyFn)(void* user_data);
typedef struct FS_AppTaskQueue FS_AppTaskQueue;
typedef struct FS_Lifetime FS_Lifetime;

typedef void (FS_CALL *FS_AppWakeFn)(void* user_data);

typedef struct FS_AppTaskQueueDesc {
    uint32_t struct_size;
    const FS_Allocator* allocator;
    FS_AppWakeFn wake;
    void* wake_user_data;
} FS_AppTaskQueueDesc;
#define FS_APP_TASK_QUEUE_DESC_INIT { sizeof(FS_AppTaskQueueDesc), NULL, NULL, NULL }

FS_API FS_Result FS_CALL fs_app_task_queue_create(const FS_AppTaskQueueDesc* desc,
                                                   FS_AppTaskQueue** out_queue,
                                                   FS_Error* error);
FS_API void FS_CALL fs_app_task_queue_begin_close(FS_AppTaskQueue* queue);
FS_API FS_Result FS_CALL fs_app_task_queue_post(FS_AppTaskQueue* queue,
                                                 FS_AppTaskFn task,
                                                 void* user_data,
                                                 FS_AppTaskDestroyFn destroy,
                                                 FS_Error* error);
FS_API uint32_t FS_CALL fs_app_task_queue_drain(FS_AppTaskQueue* queue,
                                                 uint32_t max_tasks);
FS_API void FS_CALL fs_app_task_queue_destroy(FS_AppTaskQueue* queue);

FS_API FS_Result FS_CALL fs_lifetime_create(const FS_Allocator* allocator,
                                             FS_Lifetime** out_lifetime,
                                             FS_Error* error);
FS_API bool FS_CALL fs_lifetime_try_retain(FS_Lifetime* lifetime);
FS_API void FS_CALL fs_lifetime_release(FS_Lifetime* lifetime);
FS_API void FS_CALL fs_lifetime_begin_close(FS_Lifetime* lifetime);
FS_API bool FS_CALL fs_lifetime_is_closing(const FS_Lifetime* lifetime);
FS_API uint64_t FS_CALL fs_lifetime_generation(const FS_Lifetime* lifetime);

#ifdef __cplusplus
}
#endif

#endif
