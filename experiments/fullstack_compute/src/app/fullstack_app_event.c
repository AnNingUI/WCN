#include "fullstack_app_event.h"
#include "fullstack_sync_private.h"
#include <string.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

struct FS_EventQueue {
    const FS_Allocator* allocator;
    FS_Mutex mutex;
    FS_AppEvent* events;
    uint32_t capacity, hard_capacity, head, count;
    uint64_t payload_bytes, hard_payload_bytes, next_sequence;
    bool fatal_pending;
    FS_AppEvent fatal_event;
};

static uint64_t fs_now_ns(void) {
#if defined(_WIN32)
    LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (uint64_t)((c.QuadPart * 1000000000ull) / (uint64_t)f.QuadPart);
#else
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
#endif
}
static bool fs_event_is_coalescible(FS_AppEventType t) {
    return t==FS_APP_EVENT_POINTER_MOVE || t==FS_APP_EVENT_WINDOW_RESIZED ||
           t==FS_APP_EVENT_FRAMEBUFFER_RESIZED || t==FS_APP_EVENT_SCALE_CHANGED;
}
static bool fs_event_is_critical(FS_AppEventType t) {
    return t==FS_APP_EVENT_KEY || t==FS_APP_EVENT_POINTER_BUTTON ||
           t==FS_APP_EVENT_TEXT_INPUT || t==FS_APP_EVENT_TEXT_EDITING ||
           t==FS_APP_EVENT_SUSPEND || t==FS_APP_EVENT_RESUME ||
           t==FS_APP_EVENT_SURFACE_AVAILABLE || t==FS_APP_EVENT_SURFACE_UNAVAILABLE ||
           t==FS_APP_EVENT_DEVICE_LOST || t==FS_APP_EVENT_BACKEND_ERROR ||
           t==FS_APP_EVENT_QUIT_REQUESTED;
}
static void fs_event_rebase_inline(FS_AppEvent* e) {
    if (!e) return;
    if ((e->type == FS_APP_EVENT_TEXT_INPUT || e->type == FS_APP_EVENT_TEXT_EDITING) &&
        !e->data.text.utf8.owner) {
        e->data.text.utf8.data = e->data.text.utf8.inline_data;
    }
}
static uint64_t fs_event_heap_bytes(const FS_AppEvent* e) {
    if ((e->type==FS_APP_EVENT_TEXT_INPUT || e->type==FS_APP_EVENT_TEXT_EDITING) &&
        e->data.text.utf8.owner) return e->data.text.utf8.size;
    return 0;
}
FS_Result FS_CALL fs_app_event_set_utf8(FS_AppEvent* e,const char* text,uint32_t n,
                                         const FS_Allocator* a,FS_Error* error) {
    if(!e || (!text&&n)) return FS_RESULT_INVALID_ARGUMENT;
    FS_AppOwnedBytes* b=&e->data.text.utf8; memset(b,0,sizeof(*b)); b->size=n;
    if(n<FS_APP_INLINE_PAYLOAD_CAPACITY){ if(n)memcpy(b->inline_data,text,n); b->inline_data[n]=0; b->data=b->inline_data; return FS_RESULT_OK; }
    if(!a)a=fs_default_allocator(); uint8_t* p=(uint8_t*)fs_allocator_allocate(a,(size_t)n+1u,1);
    if(!p){FS_ERROR_SET(error,FS_RESULT_OUT_OF_MEMORY,FS_ERROR_DOMAIN_APP,0,"fs_app_event_set_utf8","payload allocation failed");return FS_RESULT_OUT_OF_MEMORY;}
    memcpy(p,text,n);p[n]=0;b->data=p;b->owner=(void*)a;return FS_RESULT_OK;
}
void FS_CALL fs_app_event_release(FS_AppEvent* e){
    if(!e)return; FS_AppOwnedBytes* b=&e->data.text.utf8;
    if((e->type==FS_APP_EVENT_TEXT_INPUT||e->type==FS_APP_EVENT_TEXT_EDITING)&&b->owner&&b->data){
        fs_allocator_deallocate((const FS_Allocator*)b->owner,(void*)b->data,(size_t)b->size+1u,1);
    }
    memset(e,0,sizeof(*e)); e->struct_size=sizeof(*e);
}
static FS_Result fs_event_copy(FS_EventQueue*q,FS_AppEvent*dst,const FS_AppEvent*src,FS_Error*error){
    *dst=*src; dst->struct_size=sizeof(*dst);
    if(src->type==FS_APP_EVENT_TEXT_INPUT||src->type==FS_APP_EVENT_TEXT_EDITING){
        uint32_t n=src->data.text.utf8.size; const char* p=(const char*)src->data.text.utf8.data;
        memset(&dst->data.text.utf8,0,sizeof(dst->data.text.utf8));
        return fs_app_event_set_utf8(dst,p,n,q->allocator,error);
    }
    return FS_RESULT_OK;
}
FS_Result FS_CALL fs_event_queue_create(const FS_EventQueueDesc*d,FS_EventQueue**out,FS_Error*error){
    if(out)*out=NULL;if(!d||!out||d->struct_size<sizeof(*d)||!d->initial_capacity||d->hard_event_capacity<d->initial_capacity)return FS_RESULT_INVALID_ARGUMENT;
    const FS_Allocator*a=d->allocator?d->allocator:fs_default_allocator();
    FS_EventQueue*q=(FS_EventQueue*)fs_allocator_allocate(a,sizeof(*q),sizeof(void*));
    if(!q)return FS_RESULT_OUT_OF_MEMORY;memset(q,0,sizeof(*q));q->allocator=a;q->capacity=d->initial_capacity;q->hard_capacity=d->hard_event_capacity;q->hard_payload_bytes=d->hard_payload_bytes;q->next_sequence=1;
    q->events=(FS_AppEvent*)fs_allocator_allocate(a,q->capacity*sizeof(*q->events),sizeof(void*));
    if(!q->events||!fs_mutex_init(&q->mutex)){if(q->events)fs_allocator_deallocate(a,q->events,q->capacity*sizeof(*q->events),sizeof(void*));fs_allocator_deallocate(a,q,sizeof(*q),sizeof(void*));FS_ERROR_SET(error,FS_RESULT_OUT_OF_MEMORY,FS_ERROR_DOMAIN_APP,0,"fs_event_queue_create","queue allocation failed");return FS_RESULT_OUT_OF_MEMORY;}
    memset(q->events,0,q->capacity*sizeof(*q->events));*out=q;return FS_RESULT_OK;
}
void FS_CALL fs_event_queue_destroy(FS_EventQueue*q){if(!q)return;fs_mutex_lock(&q->mutex);for(uint32_t i=0;i<q->count;i++)fs_app_event_release(&q->events[(q->head+i)%q->capacity]);fs_mutex_unlock(&q->mutex);fs_mutex_destroy(&q->mutex);fs_allocator_deallocate(q->allocator,q->events,q->capacity*sizeof(*q->events),sizeof(void*));fs_allocator_deallocate(q->allocator,q,sizeof(*q),sizeof(void*));}
static bool fs_queue_grow(FS_EventQueue*q){if(q->capacity>=q->hard_capacity)return false;uint32_t n=q->capacity*2u;if(n>q->hard_capacity)n=q->hard_capacity;FS_AppEvent*p=(FS_AppEvent*)fs_allocator_allocate(q->allocator,n*sizeof(*p),sizeof(void*));if(!p)return false;memset(p,0,n*sizeof(*p));for(uint32_t i=0;i<q->count;i++){p[i]=q->events[(q->head+i)%q->capacity];fs_event_rebase_inline(&p[i]);}fs_allocator_deallocate(q->allocator,q->events,q->capacity*sizeof(*q->events),sizeof(void*));q->events=p;q->capacity=n;q->head=0;return true;}
static void fs_queue_set_fatal(FS_EventQueue*q,const FS_AppEvent*source){if(q->fatal_pending)return;memset(&q->fatal_event,0,sizeof(q->fatal_event));q->fatal_event.struct_size=sizeof(q->fatal_event);q->fatal_event.type=FS_APP_EVENT_FATAL_QUEUE_PRESSURE;q->fatal_event.sequence=q->next_sequence++;q->fatal_event.timestamp_ns=fs_now_ns();q->fatal_event.window_id=source?source->window_id:0;q->fatal_event.data.failure.code=source?source->type:0;q->fatal_event.data.failure.detail=q->count;q->fatal_pending=true;}
FS_Result FS_CALL fs_event_queue_push(FS_EventQueue*q,const FS_AppEvent*e,FS_Error*error){
    if(!q||!e)return FS_RESULT_INVALID_ARGUMENT;fs_mutex_lock(&q->mutex);
    if(q->fatal_pending){fs_mutex_unlock(&q->mutex);return FS_RESULT_INVALID_STATE;}
    if(fs_event_is_coalescible(e->type)){for(uint32_t i=q->count;i>0;i--){FS_AppEvent*x=&q->events[(q->head+i-1u)%q->capacity];if(x->type==e->type&&x->window_id==e->window_id){uint64_t seq=x->sequence;*x=*e;x->struct_size=sizeof(*x);x->sequence=seq;x->timestamp_ns=fs_now_ns();fs_mutex_unlock(&q->mutex);return FS_RESULT_OK;}}}
    uint64_t payload=(e->type==FS_APP_EVENT_TEXT_INPUT||e->type==FS_APP_EVENT_TEXT_EDITING)?e->data.text.utf8.size:0;
    if(q->payload_bytes+payload>q->hard_payload_bytes|| (q->count==q->capacity&&!fs_queue_grow(q))){if(fs_event_is_critical(e->type))fs_queue_set_fatal(q,e);fs_mutex_unlock(&q->mutex);FS_ERROR_SET(error,FS_RESULT_QUEUE_PRESSURE,FS_ERROR_DOMAIN_APP,0,"fs_event_queue_push","event queue hard limit reached");return FS_RESULT_QUEUE_PRESSURE;}
    uint32_t index=(q->head+q->count)%q->capacity;FS_Result r=fs_event_copy(q,&q->events[index],e,error);if(r!=FS_RESULT_OK){if(fs_event_is_critical(e->type))fs_queue_set_fatal(q,e);fs_mutex_unlock(&q->mutex);return r;}
    q->events[index].sequence=q->next_sequence++;q->events[index].timestamp_ns=fs_now_ns();q->payload_bytes+=fs_event_heap_bytes(&q->events[index]);q->count++;fs_mutex_unlock(&q->mutex);return FS_RESULT_OK;
}
FS_Result FS_CALL fs_event_queue_poll(FS_EventQueue*q,FS_AppEvent*out,FS_Error*error){(void)error;if(!q||!out)return FS_RESULT_INVALID_ARGUMENT;fs_mutex_lock(&q->mutex);if(q->count){*out=q->events[q->head];fs_event_rebase_inline(out);memset(&q->events[q->head],0,sizeof(*out));q->head=(q->head+1u)%q->capacity;q->count--;uint64_t n=fs_event_heap_bytes(out);q->payload_bytes=q->payload_bytes>=n?q->payload_bytes-n:0;fs_mutex_unlock(&q->mutex);return FS_RESULT_OK;}if(q->fatal_pending){*out=q->fatal_event;q->fatal_pending=false;fs_mutex_unlock(&q->mutex);return FS_RESULT_OK;}fs_mutex_unlock(&q->mutex);return FS_RESULT_SKIP;}
uint32_t FS_CALL fs_event_queue_drain(FS_EventQueue*q,FS_AppEvent*events,uint32_t capacity){uint32_t n=0;while(n<capacity&&fs_event_queue_poll(q,&events[n],NULL)==FS_RESULT_OK)n++;return n;}
uint64_t FS_CALL fs_event_queue_count(const FS_EventQueue*q){if(!q)return 0;FS_EventQueue*m=(FS_EventQueue*)q;fs_mutex_lock(&m->mutex);uint64_t n=m->count+(m->fatal_pending?1u:0u);fs_mutex_unlock(&m->mutex);return n;}
void FS_CALL fs_input_state_init(FS_InputState*s){if(!s)return;memset(s,0,sizeof(*s));s->struct_size=sizeof(*s);}
void FS_CALL fs_input_state_begin_frame(FS_InputState*s){if(!s)return;memset(s->keys_pressed,0,sizeof(s->keys_pressed));memset(s->keys_released,0,sizeof(s->keys_released));memset(s->buttons_pressed,0,sizeof(s->buttons_pressed));memset(s->buttons_released,0,sizeof(s->buttons_released));s->scroll_x=s->scroll_y=0;}
void FS_CALL fs_input_state_apply(FS_InputState*s,const FS_AppEvent*e){if(!s||!e)return;if(e->type==FS_APP_EVENT_KEY&&e->data.key.key<FS_APP_MAX_KEYS){uint32_t k=e->data.key.key;s->modifiers=e->data.key.modifiers;if(e->data.key.action==FS_APP_KEY_PRESS){if(!s->keys_down[k])s->keys_pressed[k]=true;s->keys_down[k]=true;}else if(e->data.key.action==FS_APP_KEY_RELEASE){if(s->keys_down[k])s->keys_released[k]=true;s->keys_down[k]=false;}}else if(e->type==FS_APP_EVENT_POINTER_BUTTON&&e->data.pointer.button<FS_APP_MAX_POINTER_BUTTONS){uint32_t b=e->data.pointer.button;if(e->data.pointer.action){if(!s->buttons_down[b])s->buttons_pressed[b]=true;s->buttons_down[b]=true;}else{if(s->buttons_down[b])s->buttons_released[b]=true;s->buttons_down[b]=false;}}else if(e->type==FS_APP_EVENT_POINTER_MOVE){s->pointer_logical_x=e->data.pointer.logical_x;s->pointer_logical_y=e->data.pointer.logical_y;s->pointer_framebuffer_x=e->data.pointer.framebuffer_x;s->pointer_framebuffer_y=e->data.pointer.framebuffer_y;}else if(e->type==FS_APP_EVENT_POINTER_SCROLL){s->scroll_x+=e->data.scroll.x;s->scroll_y+=e->data.scroll.y;}}
