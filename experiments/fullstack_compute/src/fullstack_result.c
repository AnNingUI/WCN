#include "fullstack_result.h"
#include <stdio.h>
#include <string.h>

void FS_CALL fs_error_clear(FS_Error* error) {
    if (!error) return;
    uint32_t size = error->struct_size;
    memset(error, 0, sizeof(*error));
    error->struct_size = size ? size : (uint32_t)sizeof(*error);
}
void FS_CALL fs_error_set_v(FS_Error* error, FS_Result code,
                            FS_ErrorDomain domain, int64_t native_code,
                            const char* operation, const char* source_file,
                            uint32_t source_line, const char* format,
                            va_list args) {
    if (!error) return;
    fs_error_clear(error);
    error->code = code; error->domain = domain; error->native_code = native_code;
    error->operation = operation; error->source_file = source_file;
    error->source_line = source_line;
    if (format) vsnprintf(error->message, sizeof(error->message), format, args);
}
void FS_CALL fs_error_set(FS_Error* error, FS_Result code,
                          FS_ErrorDomain domain, int64_t native_code,
                          const char* operation, const char* source_file,
                          uint32_t source_line, const char* format, ...) {
    va_list args; va_start(args, format);
    fs_error_set_v(error, code, domain, native_code, operation, source_file,
                   source_line, format, args);
    va_end(args);
}
void FS_CALL fs_diagnostic_emit(const FS_DiagnosticSink* sink,
                                const FS_Diagnostic* diagnostic) {
    if (sink && sink->callback && diagnostic) sink->callback(diagnostic, sink->user_data);
}
