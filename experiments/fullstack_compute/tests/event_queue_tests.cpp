#include "fullstack_app_event.h"
#include <cassert>
#include <cstring>
#include <thread>
#include <vector>

int main(){
    FS_EventQueueDesc d=FS_EVENT_QUEUE_DESC_INIT;d.initial_capacity=4;d.hard_event_capacity=64;d.hard_payload_bytes=4096;
    FS_EventQueue*q=nullptr;FS_Error e=FS_ERROR_INIT;assert(fs_event_queue_create(&d,&q,&e)==FS_RESULT_OK);
    FS_AppEvent text=FS_APP_EVENT_INIT;text.type=FS_APP_EVENT_TEXT_INPUT;const char*zh=u8"异步莫奈取色";assert(fs_app_event_set_utf8(&text,zh,(uint32_t)strlen(zh),nullptr,&e)==FS_RESULT_OK);assert(fs_event_queue_push(q,&text,&e)==FS_RESULT_OK);fs_app_event_release(&text);
    FS_AppEvent move=FS_APP_EVENT_INIT;move.type=FS_APP_EVENT_POINTER_MOVE;move.window_id=7;move.data.pointer.logical_x=1;assert(fs_event_queue_push(q,&move,&e)==FS_RESULT_OK);move.data.pointer.logical_x=9;assert(fs_event_queue_push(q,&move,&e)==FS_RESULT_OK);assert(fs_event_queue_count(q)==2);
    FS_AppEvent out=FS_APP_EVENT_INIT;assert(fs_event_queue_poll(q,&out,&e)==FS_RESULT_OK);assert(out.type==FS_APP_EVENT_TEXT_INPUT&&std::strcmp((const char*)out.data.text.utf8.data,zh)==0);fs_app_event_release(&out);assert(fs_event_queue_poll(q,&out,&e)==FS_RESULT_OK&&out.data.pointer.logical_x==9);
    std::vector<std::thread> workers;for(int t=0;t<4;t++)workers.emplace_back([q,t](){for(int i=0;i<10;i++){FS_AppEvent x=FS_APP_EVENT_INIT;x.type=FS_APP_EVENT_USER;x.data.user.kind=(uint64_t)t;x.data.user.value=(uint64_t)i;assert(fs_event_queue_push(q,&x,nullptr)==FS_RESULT_OK);}});for(auto&x:workers)x.join();assert(fs_event_queue_count(q)==40);while(fs_event_queue_poll(q,&out,nullptr)==FS_RESULT_OK)fs_app_event_release(&out);fs_event_queue_destroy(q);
    d.initial_capacity=1;d.hard_event_capacity=1;d.hard_payload_bytes=1;assert(fs_event_queue_create(&d,&q,&e)==FS_RESULT_OK);FS_AppEvent key=FS_APP_EVENT_INIT;key.type=FS_APP_EVENT_KEY;assert(fs_event_queue_push(q,&key,&e)==FS_RESULT_OK);assert(fs_event_queue_push(q,&key,&e)==FS_RESULT_QUEUE_PRESSURE);assert(fs_event_queue_poll(q,&out,&e)==FS_RESULT_OK&&out.type==FS_APP_EVENT_KEY);assert(fs_event_queue_poll(q,&out,&e)==FS_RESULT_OK&&out.type==FS_APP_EVENT_FATAL_QUEUE_PRESSURE);fs_event_queue_destroy(q);
    FS_InputState input;fs_input_state_init(&input);key.data.key.key=4;key.data.key.action=FS_APP_KEY_PRESS;fs_input_state_apply(&input,&key);assert(input.keys_down[4]&&input.keys_pressed[4]);fs_input_state_begin_frame(&input);assert(input.keys_down[4]&&!input.keys_pressed[4]);key.data.key.action=FS_APP_KEY_RELEASE;fs_input_state_apply(&input,&key);assert(!input.keys_down[4]&&input.keys_released[4]);return 0;
}
