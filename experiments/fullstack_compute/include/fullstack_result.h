#ifndef FULLSTACK_RESULT_H
#define FULLSTACK_RESULT_H

#include "fullstack_abi.h"
#include <stdarg.h>

#define FS_RESULT_OK                    ((FS_Result)0u)
#define FS_RESULT_SKIP                  ((FS_Result)1u)
#define FS_RESULT_PENDING               ((FS_Result)2u)
#define FS_RESULT_CANCELLED             ((FS_Result)3u)
#define FS_RESULT_UNSUPPORTED           ((FS_Result)4u)
#define FS_RESULT_INVALID_ARGUMENT      ((FS_Result)0x1001u)
#define FS_RESULT_INVALID_STATE         ((FS_Result)0x1002u)
#define FS_RESULT_WRONG_THREAD          ((FS_Result)0x1003u)
#define FS_RESULT_OUT_OF_MEMORY         ((FS_Result)0x1004u)
#define FS_RESULT_ABI_MISMATCH          ((FS_Result)0x1005u)
#define FS_RESULT_STRUCT_TOO_SMALL      ((FS_Result)0x1006u)
#define FS_RESULT_QUEUE_PRESSURE        ((FS_Result)0x1007u)
#define FS_RESULT_INTERNAL_ERROR        ((FS_Result)0x1fffu)

typedef uint32_t FS_ErrorDomain;
#define FS_ERROR_DOMAIN_NONE             ((FS_ErrorDomain)0u)
#define FS_ERROR_DOMAIN_ABI              ((FS_ErrorDomain)1u)
#define FS_ERROR_DOMAIN_APP              ((FS_ErrorDomain)2u)
#define FS_ERROR_DOMAIN_BACKEND          ((FS_ErrorDomain)3u)
#define FS_ERROR_DOMAIN_GPU              ((FS_ErrorDomain)4u)
#define FS_ERROR_DOMAIN_RENDER           ((FS_ErrorDomain)5u)
#define FS_ERROR_DOMAIN_PLATFORM_SERVICE ((FS_ErrorDomain)6u)

typedef struct FS_Error {
    uint32_t struct_size;
    FS_Result code;
    FS_ErrorDomain domain;
    int64_t native_code;
    const char* operation;
    char message[256];
    const char* source_file;
    uint32_t source_line;
    uint64_t sequence;
} FS_Error;

#define FS_ERROR_INIT { sizeof(FS_Error), FS_RESULT_OK, FS_ERROR_DOMAIN_NONE, \
                        0, NULL, {0}, NULL, 0, 0 }

typedef uint32_t FS_LogLevel;
#define FS_LOG_TRACE   ((FS_LogLevel)0u)
#define FS_LOG_DEBUG   ((FS_LogLevel)1u)
#define FS_LOG_INFO    ((FS_LogLevel)2u)
#define FS_LOG_WARNING ((FS_LogLevel)3u)
#define FS_LOG_ERROR   ((FS_LogLevel)4u)
#define FS_LOG_FATAL   ((FS_LogLevel)5u)

typedef struct FS_Diagnostic {
    uint32_t struct_size;
    FS_LogLevel level;
    FS_ErrorDomain domain;
    FS_Result code;
    uint64_t sequence;
    uint64_t object_id;
    const char* operation;
    const char* message;
} FS_Diagnostic;

typedef void (FS_CALL *FS_DiagnosticCallback)(const FS_Diagnostic* diagnostic,
                                               void* user_data);
typedef struct FS_DiagnosticSink {
    uint32_t struct_size;
    FS_DiagnosticCallback callback;
    void* user_data;
} FS_DiagnosticSink;

FS_API void FS_CALL fs_error_clear(FS_Error* error);
FS_API void FS_CALL fs_error_set(FS_Error* error, FS_Result code,
                                 FS_ErrorDomain domain, int64_t native_code,
                                 const char* operation, const char* source_file,
                                 uint32_t source_line, const char* format, ...);
FS_API void FS_CALL fs_error_set_v(FS_Error* error, FS_Result code,
                                   FS_ErrorDomain domain, int64_t native_code,
                                   const char* operation, const char* source_file,
                                   uint32_t source_line, const char* format,
                                   va_list args);
FS_API void FS_CALL fs_diagnostic_emit(const FS_DiagnosticSink* sink,
                                       const FS_Diagnostic* diagnostic);

#define FS_ERROR_SET(error, code, domain, native_code, operation, format, ...) \
    fs_error_set((error), (code), (domain), (native_code), (operation), \
                 __FILE__, (uint32_t)__LINE__, (format), ##__VA_ARGS__)

#endif
