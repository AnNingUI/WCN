#ifndef FULLSTACK_APP_EVENT_H
#define FULLSTACK_APP_EVENT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "fullstack_result.h"

#define FS_APP_EVENT_ABI_VERSION FS_ABI_VERSION(1, 0)
#define FS_APP_INLINE_PAYLOAD_CAPACITY 64u
#define FS_APP_MAX_KEYS 256u
#define FS_APP_MAX_POINTER_BUTTONS 8u

typedef uint64_t FS_AppWindowId;
typedef uint64_t FS_AppDeviceId;
typedef uint64_t FS_AppPointerId;
typedef uint32_t FS_AppEventType;
#define FS_APP_EVENT_NONE                    ((FS_AppEventType)0u)
#define FS_APP_EVENT_QUIT_REQUESTED          ((FS_AppEventType)1u)
#define FS_APP_EVENT_WINDOW_RESIZED          ((FS_AppEventType)2u)
#define FS_APP_EVENT_FRAMEBUFFER_RESIZED     ((FS_AppEventType)3u)
#define FS_APP_EVENT_SCALE_CHANGED           ((FS_AppEventType)4u)
#define FS_APP_EVENT_POINTER_MOVE            ((FS_AppEventType)5u)
#define FS_APP_EVENT_POINTER_BUTTON          ((FS_AppEventType)6u)
#define FS_APP_EVENT_POINTER_SCROLL          ((FS_AppEventType)7u)
#define FS_APP_EVENT_KEY                     ((FS_AppEventType)8u)
#define FS_APP_EVENT_TEXT_INPUT              ((FS_AppEventType)9u)
#define FS_APP_EVENT_TEXT_EDITING            ((FS_AppEventType)10u)
#define FS_APP_EVENT_SUSPEND                 ((FS_AppEventType)11u)
#define FS_APP_EVENT_RESUME                  ((FS_AppEventType)12u)
#define FS_APP_EVENT_SURFACE_AVAILABLE       ((FS_AppEventType)13u)
#define FS_APP_EVENT_SURFACE_UNAVAILABLE     ((FS_AppEventType)14u)
#define FS_APP_EVENT_LOW_MEMORY              ((FS_AppEventType)15u)
#define FS_APP_EVENT_DEVICE_LOST             ((FS_AppEventType)16u)
#define FS_APP_EVENT_BACKEND_ERROR           ((FS_AppEventType)17u)
#define FS_APP_EVENT_USER                    ((FS_AppEventType)18u)
#define FS_APP_EVENT_FATAL_QUEUE_PRESSURE    ((FS_AppEventType)19u)

typedef uint32_t FS_AppKeyAction;
#define FS_APP_KEY_RELEASE ((FS_AppKeyAction)0u)
#define FS_APP_KEY_PRESS   ((FS_AppKeyAction)1u)
#define FS_APP_KEY_REPEAT  ((FS_AppKeyAction)2u)

typedef struct FS_AppOwnedBytes {
    uint32_t size;
    const uint8_t* data;
    uint8_t inline_data[FS_APP_INLINE_PAYLOAD_CAPACITY];
    void* owner;
} FS_AppOwnedBytes;

typedef struct FS_AppPointerEvent {
    FS_AppPointerId pointer_id;
    float logical_x, logical_y;
    float framebuffer_x, framebuffer_y;
    float delta_logical_x, delta_logical_y;
    uint32_t button;
    uint32_t action;
    uint32_t modifiers;
} FS_AppPointerEvent;

typedef struct FS_AppWindowEvent {
    uint32_t logical_width, logical_height;
    uint32_t framebuffer_width, framebuffer_height;
    float scale_x, scale_y;
} FS_AppWindowEvent;

typedef struct FS_AppKeyEvent {
    uint32_t key;
    uint32_t physical_key;
    FS_AppKeyAction action;
    uint32_t modifiers;
} FS_AppKeyEvent;

typedef struct FS_AppTextEvent {
    FS_AppOwnedBytes utf8;
    uint32_t selection_start;
    uint32_t selection_length;
} FS_AppTextEvent;

typedef struct FS_AppEvent {
    uint32_t struct_size;
    FS_AppEventType type;
    uint64_t sequence;
    uint64_t timestamp_ns;
    FS_AppWindowId window_id;
    FS_AppDeviceId device_id;
    union {
        FS_AppPointerEvent pointer;
        FS_AppWindowEvent window;
        FS_AppKeyEvent key;
        FS_AppTextEvent text;
        struct { float x, y; } scroll;
        struct { uint64_t code, detail; } failure;
        struct { uint64_t kind, value; } user;
    } data;
} FS_AppEvent;
#define FS_APP_EVENT_INIT { sizeof(FS_AppEvent), FS_APP_EVENT_NONE, 0, 0, 0, 0, {{0}} }

typedef struct FS_EventQueueDesc {
    uint32_t struct_size;
    const FS_Allocator* allocator;
    uint32_t initial_capacity;
    uint32_t hard_event_capacity;
    uint64_t hard_payload_bytes;
} FS_EventQueueDesc;
#define FS_EVENT_QUEUE_DESC_INIT { sizeof(FS_EventQueueDesc), NULL, 256u, 4096u, 4u*1024u*1024u }

typedef struct FS_EventQueue FS_EventQueue;

FS_API FS_Result FS_CALL fs_event_queue_create(const FS_EventQueueDesc* desc,
                                                FS_EventQueue** out_queue,
                                                FS_Error* error);
FS_API void FS_CALL fs_event_queue_destroy(FS_EventQueue* queue);
FS_API FS_Result FS_CALL fs_event_queue_push(FS_EventQueue* queue,
                                              const FS_AppEvent* event,
                                              FS_Error* error);
FS_API FS_Result FS_CALL fs_event_queue_poll(FS_EventQueue* queue,
                                              FS_AppEvent* out_event,
                                              FS_Error* error);
FS_API uint32_t FS_CALL fs_event_queue_drain(FS_EventQueue* queue,
                                              FS_AppEvent* events,
                                              uint32_t capacity);
FS_API void FS_CALL fs_app_event_release(FS_AppEvent* event);
FS_API void FS_CALL fs_app_owned_bytes_release(FS_AppOwnedBytes* bytes);
FS_API FS_Result FS_CALL fs_app_event_set_utf8(FS_AppEvent* event,
                                                const char* utf8,
                                                uint32_t byte_count,
                                                const FS_Allocator* allocator,
                                                FS_Error* error);
FS_API uint64_t FS_CALL fs_event_queue_count(const FS_EventQueue* queue);

typedef struct FS_InputState {
    uint32_t struct_size;
    bool keys_down[FS_APP_MAX_KEYS];
    bool keys_pressed[FS_APP_MAX_KEYS];
    bool keys_released[FS_APP_MAX_KEYS];
    bool buttons_down[FS_APP_MAX_POINTER_BUTTONS];
    bool buttons_pressed[FS_APP_MAX_POINTER_BUTTONS];
    bool buttons_released[FS_APP_MAX_POINTER_BUTTONS];
    float pointer_logical_x, pointer_logical_y;
    float pointer_framebuffer_x, pointer_framebuffer_y;
    float scroll_x, scroll_y;
    uint32_t modifiers;
} FS_InputState;

FS_API void FS_CALL fs_input_state_init(FS_InputState* state);
FS_API void FS_CALL fs_input_state_begin_frame(FS_InputState* state);
FS_API void FS_CALL fs_input_state_apply(FS_InputState* state,
                                          const FS_AppEvent* event);

#ifdef __cplusplus
}
#endif

#endif
