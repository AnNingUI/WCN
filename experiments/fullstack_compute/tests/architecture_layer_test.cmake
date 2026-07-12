if(NOT DEFINED FS_SOURCE_ROOT)
    message(FATAL_ERROR "FS_SOURCE_ROOT is required")
endif()
execute_process(
    COMMAND powershell -NoProfile -ExecutionPolicy Bypass
        -File "${FS_SOURCE_ROOT}/tools/architecture/check_layers.ps1"
        -Root "${FS_SOURCE_ROOT}"
    RESULT_VARIABLE audit_result
    OUTPUT_VARIABLE audit_stdout
    ERROR_VARIABLE audit_stderr
)
if(NOT audit_result EQUAL 0)
    message(FATAL_ERROR "Architecture layer audit failed.\n${audit_stdout}${audit_stderr}")
endif()
message(STATUS "${audit_stdout}")
