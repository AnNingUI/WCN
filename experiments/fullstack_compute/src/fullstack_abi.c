#include "fullstack_abi.h"
#include "fullstack_result.h"

#include <stdlib.h>

static void* FS_CALL fs_std_allocate(void* user_data, size_t size, size_t alignment) {
    (void)user_data; (void)alignment;
    return size ? malloc(size) : NULL;
}
static void* FS_CALL fs_std_reallocate(void* user_data, void* memory,
                                       size_t old_size, size_t new_size,
                                       size_t alignment) {
    (void)user_data; (void)old_size; (void)alignment;
    if (!new_size) { free(memory); return NULL; }
    return realloc(memory, new_size);
}
static void FS_CALL fs_std_deallocate(void* user_data, void* memory,
                                      size_t size, size_t alignment) {
    (void)user_data; (void)size; (void)alignment; free(memory);
}
static const FS_Allocator fs_std_allocator = {
    sizeof(FS_Allocator), NULL,
    fs_std_allocate, fs_std_reallocate, fs_std_deallocate
};

const FS_Allocator* FS_CALL fs_default_allocator(void) { return &fs_std_allocator; }
void* FS_CALL fs_allocator_allocate(const FS_Allocator* a, size_t n, size_t align) {
    if (!a) a = fs_default_allocator();
    return a->allocate ? a->allocate(a->user_data, n, align) : NULL;
}
void* FS_CALL fs_allocator_reallocate(const FS_Allocator* a, void* p,
                                      size_t old_n, size_t new_n, size_t align) {
    if (!a) a = fs_default_allocator();
    return a->reallocate ? a->reallocate(a->user_data, p, old_n, new_n, align) : NULL;
}
void FS_CALL fs_allocator_deallocate(const FS_Allocator* a, void* p,
                                     size_t n, size_t align) {
    if (!p) return;
    if (!a) a = fs_default_allocator();
    if (a->deallocate) a->deallocate(a->user_data, p, n, align);
}
bool FS_CALL fs_abi_has_field(uint32_t size, size_t offset, size_t field_size) {
    return (size_t)size >= offset && field_size <= (size_t)size - offset;
}
FS_Result FS_CALL fs_abi_validate_header(const FS_AbiHeader* h,
                                         uint32_t minimum_size,
                                         uint32_t supported_version) {
    if (!h) return FS_RESULT_INVALID_ARGUMENT;
    if (h->struct_size < minimum_size) return FS_RESULT_STRUCT_TOO_SMALL;
    if (FS_ABI_VERSION_MAJOR(h->abi_version) !=
        FS_ABI_VERSION_MAJOR(supported_version)) return FS_RESULT_ABI_MISMATCH;
    if (FS_ABI_VERSION_MINOR(h->abi_version) >
        FS_ABI_VERSION_MINOR(supported_version)) return FS_RESULT_ABI_MISMATCH;
    return FS_RESULT_OK;
}
