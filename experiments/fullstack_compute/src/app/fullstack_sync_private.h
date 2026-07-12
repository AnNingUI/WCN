#ifndef FULLSTACK_SYNC_PRIVATE_H
#define FULLSTACK_SYNC_PRIVATE_H
#include <stdbool.h>
#if defined(_WIN32)
#include <windows.h>
typedef CRITICAL_SECTION FS_Mutex;
static inline bool fs_mutex_init(FS_Mutex* m){ InitializeCriticalSection(m); return true; }
static inline void fs_mutex_destroy(FS_Mutex* m){ DeleteCriticalSection(m); }
static inline void fs_mutex_lock(FS_Mutex* m){ EnterCriticalSection(m); }
static inline void fs_mutex_unlock(FS_Mutex* m){ LeaveCriticalSection(m); }
#else
#include <pthread.h>
typedef pthread_mutex_t FS_Mutex;
static inline bool fs_mutex_init(FS_Mutex* m){ return pthread_mutex_init(m,NULL)==0; }
static inline void fs_mutex_destroy(FS_Mutex* m){ pthread_mutex_destroy(m); }
static inline void fs_mutex_lock(FS_Mutex* m){ pthread_mutex_lock(m); }
static inline void fs_mutex_unlock(FS_Mutex* m){ pthread_mutex_unlock(m); }
#endif
#endif
