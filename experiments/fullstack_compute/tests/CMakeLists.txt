add_test(
    NAME fs_architecture_layers
    COMMAND ${CMAKE_COMMAND}
        -DFS_SOURCE_ROOT=${FS_ROOT}
        -P ${FS_ROOT}/tests/architecture_layer_test.cmake
)
