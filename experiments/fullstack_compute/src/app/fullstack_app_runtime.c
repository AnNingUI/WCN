#include "fullstack_app_runtime.h"
#include "fullstack_sync_private.h"
#include <string.h>

typedef struct FS_AppTaskNode {
    FS_AppTaskFn task;
    FS_AppTaskDestroyFn destroy;
    void* user_data;
    struct FS_AppTaskNode* next;
} FS_AppTaskNode;
struct FS_AppTaskQueue {
    const FS_Allocator* allocator;
    FS_Mutex mutex;
    FS_AppTaskNode* head;
    FS_AppTaskNode* tail;
    FS_AppWakeFn wake;
    void* wake_user_data;
    bool closing;
};
struct FS_Lifetime {
    const FS_Allocator* allocator;
    FS_Mutex mutex;
    uint32_t references;
    uint64_t generation;
    bool closing;
};
FS_Result FS_CALL fs_app_task_queue_create(const FS_AppTaskQueueDesc*d,FS_AppTaskQueue**out,FS_Error*error){
    if(out)*out=NULL;if(!d||!out||d->struct_size<sizeof(*d))return FS_RESULT_INVALID_ARGUMENT;const FS_Allocator*a=d->allocator?d->allocator:fs_default_allocator();FS_AppTaskQueue*q=(FS_AppTaskQueue*)fs_allocator_allocate(a,sizeof(*q),sizeof(void*));if(!q)return FS_RESULT_OUT_OF_MEMORY;memset(q,0,sizeof(*q));q->allocator=a;q->wake=d->wake;q->wake_user_data=d->wake_user_data;if(!fs_mutex_init(&q->mutex)){fs_allocator_deallocate(a,q,sizeof(*q),sizeof(void*));FS_ERROR_SET(error,FS_RESULT_INTERNAL_ERROR,FS_ERROR_DOMAIN_APP,0,"fs_app_task_queue_create","mutex initialization failed");return FS_RESULT_INTERNAL_ERROR;}*out=q;return FS_RESULT_OK;
}
void FS_CALL fs_app_task_queue_begin_close(FS_AppTaskQueue*q){if(!q)return;fs_mutex_lock(&q->mutex);q->closing=true;fs_mutex_unlock(&q->mutex);}
FS_Result FS_CALL fs_app_task_queue_post(FS_AppTaskQueue*q,FS_AppTaskFn task,void*data,FS_AppTaskDestroyFn destroy,FS_Error*error){
    if(!q||!task)return FS_RESULT_INVALID_ARGUMENT;FS_AppTaskNode*n=(FS_AppTaskNode*)fs_allocator_allocate(q->allocator,sizeof(*n),sizeof(void*));if(!n)return FS_RESULT_OUT_OF_MEMORY;*n=(FS_AppTaskNode){task,destroy,data,NULL};fs_mutex_lock(&q->mutex);if(q->closing){fs_mutex_unlock(&q->mutex);fs_allocator_deallocate(q->allocator,n,sizeof(*n),sizeof(void*));if(destroy)destroy(data);FS_ERROR_SET(error,FS_RESULT_INVALID_STATE,FS_ERROR_DOMAIN_APP,0,"fs_app_task_queue_post","task queue is closing");return FS_RESULT_INVALID_STATE;}if(q->tail)q->tail->next=n;else q->head=n;q->tail=n;FS_AppWakeFn wake=q->wake;void*wake_data=q->wake_user_data;fs_mutex_unlock(&q->mutex);if(wake)wake(wake_data);return FS_RESULT_OK;
}
uint32_t FS_CALL fs_app_task_queue_drain(FS_AppTaskQueue*q,uint32_t max){if(!q)return 0;uint32_t done=0;while(!max||done<max){fs_mutex_lock(&q->mutex);FS_AppTaskNode*n=q->head;if(n){q->head=n->next;if(!q->head)q->tail=NULL;}fs_mutex_unlock(&q->mutex);if(!n)break;n->task(n->user_data);if(n->destroy)n->destroy(n->user_data);fs_allocator_deallocate(q->allocator,n,sizeof(*n),sizeof(void*));done++;}return done;}
void FS_CALL fs_app_task_queue_destroy(FS_AppTaskQueue*q){if(!q)return;fs_app_task_queue_begin_close(q);fs_mutex_lock(&q->mutex);FS_AppTaskNode*n=q->head;q->head=q->tail=NULL;fs_mutex_unlock(&q->mutex);while(n){FS_AppTaskNode*next=n->next;if(n->destroy)n->destroy(n->user_data);fs_allocator_deallocate(q->allocator,n,sizeof(*n),sizeof(void*));n=next;}fs_mutex_destroy(&q->mutex);fs_allocator_deallocate(q->allocator,q,sizeof(*q),sizeof(void*));}
FS_Result FS_CALL fs_lifetime_create(const FS_Allocator*a,FS_Lifetime**out,FS_Error*error){if(out)*out=NULL;if(!out)return FS_RESULT_INVALID_ARGUMENT;if(!a)a=fs_default_allocator();FS_Lifetime*l=(FS_Lifetime*)fs_allocator_allocate(a,sizeof(*l),sizeof(void*));if(!l)return FS_RESULT_OUT_OF_MEMORY;memset(l,0,sizeof(*l));l->allocator=a;l->references=1;l->generation=1;if(!fs_mutex_init(&l->mutex)){fs_allocator_deallocate(a,l,sizeof(*l),sizeof(void*));FS_ERROR_SET(error,FS_RESULT_INTERNAL_ERROR,FS_ERROR_DOMAIN_APP,0,"fs_lifetime_create","mutex initialization failed");return FS_RESULT_INTERNAL_ERROR;}*out=l;return FS_RESULT_OK;}
bool FS_CALL fs_lifetime_try_retain(FS_Lifetime*l){if(!l)return false;fs_mutex_lock(&l->mutex);bool ok=!l->closing&&l->references>0;if(ok)l->references++;fs_mutex_unlock(&l->mutex);return ok;}
void FS_CALL fs_lifetime_release(FS_Lifetime*l){if(!l)return;bool free_now=false;fs_mutex_lock(&l->mutex);if(l->references)l->references--;free_now=l->references==0;fs_mutex_unlock(&l->mutex);if(free_now){const FS_Allocator*a=l->allocator;fs_mutex_destroy(&l->mutex);fs_allocator_deallocate(a,l,sizeof(*l),sizeof(void*));}}
void FS_CALL fs_lifetime_begin_close(FS_Lifetime*l){if(!l)return;fs_mutex_lock(&l->mutex);if(!l->closing){l->closing=true;l->generation++;}fs_mutex_unlock(&l->mutex);}
bool FS_CALL fs_lifetime_is_closing(const FS_Lifetime*l){if(!l)return true;FS_Lifetime*m=(FS_Lifetime*)l;fs_mutex_lock(&m->mutex);bool value=m->closing;fs_mutex_unlock(&m->mutex);return value;}
uint64_t FS_CALL fs_lifetime_generation(const FS_Lifetime*l){if(!l)return 0;FS_Lifetime*m=(FS_Lifetime*)l;fs_mutex_lock(&m->mutex);uint64_t value=m->generation;fs_mutex_unlock(&m->mutex);return value;}
