#include "fullstack_abi.h"
#include "fullstack_result.h"
#include <type_traits>
static_assert(std::is_same<FS_Result, uint32_t>::value, "FS_Result must be fixed-width");
static void FS_CALL callback(const FS_Diagnostic*, void*) {}
int main() { FS_DiagnosticCallback fn = callback; return fn ? 0 : 1; }
