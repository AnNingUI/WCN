#ifndef FULLSTACK_ABI_H
#define FULLSTACK_ABI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(FS_BUILD_SHARED)
#    if defined(FS_BUILDING_LIBRARY)
#      define FS_API __declspec(dllexport)
#    else
#      define FS_API __declspec(dllimport)
#    endif
#  else
#    define FS_API
#  endif
#  define FS_CALL __cdecl
#else
#  if defined(__GNUC__) && defined(FS_BUILD_SHARED)
#    define FS_API __attribute__((visibility("default")))
#  else
#    define FS_API
#  endif
#  define FS_CALL
#endif

#define FS_ABI_VERSION(major, minor) \
    ((((uint32_t)(major)) << 16u) | ((uint32_t)(minor) & 0xffffu))
#define FS_ABI_VERSION_MAJOR(version) ((uint32_t)(version) >> 16u)
#define FS_ABI_VERSION_MINOR(version) ((uint32_t)(version) & 0xffffu)

typedef uint32_t FS_Result;

typedef void* (FS_CALL *FS_AllocateFn)(void* user_data, size_t size, size_t alignment);
typedef void* (FS_CALL *FS_ReallocateFn)(void* user_data, void* memory,
                                         size_t old_size, size_t new_size,
                                         size_t alignment);
typedef void (FS_CALL *FS_DeallocateFn)(void* user_data, void* memory,
                                        size_t size, size_t alignment);

typedef struct FS_Allocator {
    uint32_t struct_size;
    void* user_data;
    FS_AllocateFn allocate;
    FS_ReallocateFn reallocate;
    FS_DeallocateFn deallocate;
} FS_Allocator;

typedef struct FS_AbiHeader {
    uint32_t struct_size;
    uint32_t abi_version;
} FS_AbiHeader;

typedef struct FS_CapabilityHeader {
    uint32_t struct_size;
    uint32_t capability_id;
    uint32_t version_major;
    uint32_t version_minor;
    uint64_t flags;
} FS_CapabilityHeader;

FS_API const FS_Allocator* FS_CALL fs_default_allocator(void);
FS_API void* FS_CALL fs_allocator_allocate(const FS_Allocator* allocator,
                                            size_t size, size_t alignment);
FS_API void* FS_CALL fs_allocator_reallocate(const FS_Allocator* allocator,
                                              void* memory, size_t old_size,
                                              size_t new_size, size_t alignment);
FS_API void FS_CALL fs_allocator_deallocate(const FS_Allocator* allocator,
                                             void* memory, size_t size,
                                             size_t alignment);
FS_API FS_Result FS_CALL fs_abi_validate_header(const FS_AbiHeader* header,
                                                uint32_t minimum_size,
                                                uint32_t supported_version);
FS_API bool FS_CALL fs_abi_has_field(uint32_t struct_size,
                                     size_t field_offset,
                                     size_t field_size);

#endif
