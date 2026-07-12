#include "fullstack_abi.h"
#include "fullstack_result.h"
#include <assert.h>
#include <stddef.h>
#include <string.h>

static int diagnostic_called = 0;
static void FS_CALL on_diagnostic(const FS_Diagnostic* d, void* user) {
    (void)user; assert(d && d->code == FS_RESULT_INVALID_STATE); diagnostic_called++;
}
int main(void) {
    FS_AbiHeader good = { sizeof(good), FS_ABI_VERSION(1, 2) };
    assert(fs_abi_validate_header(&good, sizeof(good), FS_ABI_VERSION(1, 3)) == FS_RESULT_OK);
    good.struct_size = 4;
    assert(fs_abi_validate_header(&good, sizeof(FS_AbiHeader), FS_ABI_VERSION(1, 3)) == FS_RESULT_STRUCT_TOO_SMALL);
    good.struct_size = sizeof(good); good.abi_version = FS_ABI_VERSION(2, 0);
    assert(fs_abi_validate_header(&good, sizeof(good), FS_ABI_VERSION(1, 3)) == FS_RESULT_ABI_MISMATCH);
    assert(fs_abi_has_field(sizeof(FS_CapabilityHeader), offsetof(FS_CapabilityHeader, flags), sizeof(uint64_t)));
    FS_Error e = FS_ERROR_INIT;
    FS_ERROR_SET(&e, FS_RESULT_INVALID_ARGUMENT, FS_ERROR_DOMAIN_ABI, 7, "test", "bad %s", "value");
    assert(e.code == FS_RESULT_INVALID_ARGUMENT && strstr(e.message, "bad value"));
    void* p = fs_allocator_allocate(NULL, 32, 8); assert(p); fs_allocator_deallocate(NULL, p, 32, 8);
    FS_DiagnosticSink sink = { sizeof(sink), on_diagnostic, NULL };
    FS_Diagnostic d = { sizeof(d), FS_LOG_ERROR, FS_ERROR_DOMAIN_APP, FS_RESULT_INVALID_STATE, 1, 2, "test", "state" };
    fs_diagnostic_emit(&sink, &d); assert(diagnostic_called == 1);
    return 0;
}
