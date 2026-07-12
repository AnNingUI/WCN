#include "fullstack_app_runtime.h"
#include <assert.h>
static int tasks,wakes,destroys;
static void FS_CALL wake(void*d){(void)d;wakes++;}
static void FS_CALL task(void*d){tasks+=*(int*)d;}
static void FS_CALL cleanup(void*d){(void)d;destroys++;}
int main(void){FS_Error e=FS_ERROR_INIT;FS_AppTaskQueueDesc d=FS_APP_TASK_QUEUE_DESC_INIT;d.wake=wake;FS_AppTaskQueue*q=NULL;assert(fs_app_task_queue_create(&d,&q,&e)==FS_RESULT_OK);int value=3;assert(fs_app_task_queue_post(q,task,&value,cleanup,&e)==FS_RESULT_OK);assert(wakes==1&&fs_app_task_queue_drain(q,0)==1&&tasks==3&&destroys==1);fs_app_task_queue_begin_close(q);assert(fs_app_task_queue_post(q,task,&value,cleanup,&e)==FS_RESULT_INVALID_STATE);assert(destroys==2);fs_app_task_queue_destroy(q);FS_Lifetime*l=NULL;assert(fs_lifetime_create(NULL,&l,&e)==FS_RESULT_OK);uint64_t g=fs_lifetime_generation(l);assert(fs_lifetime_try_retain(l));fs_lifetime_begin_close(l);assert(fs_lifetime_is_closing(l)&&fs_lifetime_generation(l)==g+1&&!fs_lifetime_try_retain(l));fs_lifetime_release(l);fs_lifetime_release(l);return 0;}
