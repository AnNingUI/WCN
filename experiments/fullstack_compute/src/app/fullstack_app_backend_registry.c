#include "fullstack_app_private.h"
#include <string.h>
#if !defined(_WIN32)
#include <pthread.h>
#endif

typedef struct FS_RegistryEntry { const FS_AppBackendFactory* factory; uint32_t instances; } FS_RegistryEntry;
static FS_RegistryEntry fs_registry[16]; static uint32_t fs_registry_count; static FS_Mutex fs_registry_mutex;
#if defined(_WIN32)
static INIT_ONCE fs_registry_once=INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK fs_registry_init_win(PINIT_ONCE once,PVOID param,PVOID*ctx){(void)once;(void)param;(void)ctx;return fs_mutex_init(&fs_registry_mutex);}
static void fs_registry_init(void){InitOnceExecuteOnce(&fs_registry_once,fs_registry_init_win,NULL,NULL);}
#else
static pthread_once_t fs_registry_once=PTHREAD_ONCE_INIT;
static void fs_registry_init_posix(void){(void)fs_mutex_init(&fs_registry_mutex);}
static void fs_registry_init(void){pthread_once(&fs_registry_once,fs_registry_init_posix);}
#endif
static int fs_find_index(const char*name){for(uint32_t i=0;i<fs_registry_count;i++)if(strcmp(fs_registry[i].factory->name,name)==0)return(int)i;return-1;}
FS_Result FS_CALL fs_app_backend_register(const FS_AppBackendFactory*f,FS_Error*e){if(!f||f->struct_size<sizeof(*f)||!f->name||!f->ops)return FS_RESULT_INVALID_ARGUMENT;FS_Result v=fs_abi_validate_header((const FS_AbiHeader*)f,sizeof(FS_AppBackendFactory),FS_APP_BACKEND_ABI_VERSION);if(v!=FS_RESULT_OK)return v;v=fs_abi_validate_header((const FS_AbiHeader*)f->ops,sizeof(FS_AppBackendOps),FS_APP_BACKEND_ABI_VERSION);if(v!=FS_RESULT_OK)return v;if(!f->ops->create||!f->ops->destroy||!f->ops->create_window||!f->ops->destroy_window||!f->ops->pump_events||!f->ops->request_wake||!f->ops->get_window_metrics||!f->ops->acquire_frame||!f->ops->present_frame||!f->ops->cancel_frame)return FS_RESULT_INVALID_ARGUMENT;fs_registry_init();fs_mutex_lock(&fs_registry_mutex);if(fs_find_index(f->name)>=0){fs_mutex_unlock(&fs_registry_mutex);return FS_RESULT_INVALID_STATE;}if(fs_registry_count>=16){fs_mutex_unlock(&fs_registry_mutex);FS_ERROR_SET(e,FS_RESULT_OUT_OF_MEMORY,FS_ERROR_DOMAIN_BACKEND,0,"fs_app_backend_register","registry capacity reached");return FS_RESULT_OUT_OF_MEMORY;}fs_registry[fs_registry_count++]=(FS_RegistryEntry){f,0};fs_mutex_unlock(&fs_registry_mutex);return FS_RESULT_OK;}
FS_Result FS_CALL fs_app_backend_unregister(const char*name,FS_Error*e){(void)e;if(!name)return FS_RESULT_INVALID_ARGUMENT;fs_registry_init();fs_mutex_lock(&fs_registry_mutex);int idx=fs_find_index(name);if(idx<0){fs_mutex_unlock(&fs_registry_mutex);return FS_RESULT_SKIP;}if(fs_registry[idx].instances){fs_mutex_unlock(&fs_registry_mutex);return FS_RESULT_INVALID_STATE;}for(uint32_t i=(uint32_t)idx+1;i<fs_registry_count;i++)fs_registry[i-1]=fs_registry[i];fs_registry_count--;fs_mutex_unlock(&fs_registry_mutex);return FS_RESULT_OK;}
const FS_AppBackendFactory* FS_CALL fs_app_backend_find(const char*name){if(!name)return NULL;fs_registry_init();fs_mutex_lock(&fs_registry_mutex);int idx=fs_find_index(name);const FS_AppBackendFactory*f=idx>=0?fs_registry[idx].factory:NULL;fs_mutex_unlock(&fs_registry_mutex);return f;}
uint32_t FS_CALL fs_app_backend_count(void){fs_registry_init();fs_mutex_lock(&fs_registry_mutex);uint32_t n=fs_registry_count;fs_mutex_unlock(&fs_registry_mutex);return n;}
FS_Result fs_app_backend_factory_acquire(const char*name,const FS_AppBackendFactory**out,FS_Error*e){if(out)*out=NULL;if(!name||!out)return FS_RESULT_INVALID_ARGUMENT;fs_registry_init();fs_mutex_lock(&fs_registry_mutex);int idx=fs_find_index(name);if(idx<0){fs_mutex_unlock(&fs_registry_mutex);FS_ERROR_SET(e,FS_RESULT_UNSUPPORTED,FS_ERROR_DOMAIN_BACKEND,0,"fs_app_backend_factory_acquire","backend not registered: %s",name);return FS_RESULT_UNSUPPORTED;}fs_registry[idx].instances++;*out=fs_registry[idx].factory;fs_mutex_unlock(&fs_registry_mutex);return FS_RESULT_OK;}
void fs_app_backend_factory_release(const FS_AppBackendFactory*f){if(!f)return;fs_registry_init();fs_mutex_lock(&fs_registry_mutex);for(uint32_t i=0;i<fs_registry_count;i++)if(fs_registry[i].factory==f&&fs_registry[i].instances){fs_registry[i].instances--;break;}fs_mutex_unlock(&fs_registry_mutex);}
