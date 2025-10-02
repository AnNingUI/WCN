# 准确性测试
add_executable(wcn_math_accuracy R.c)
target_link_libraries(wcn_math_accuracy wcn)
target_include_directories(wcn_math_accuracy PRIVATE ${CMAKE_SOURCE_DIR}/include)

# 性能测试
add_executable(wcn_math_benchmark B.c)
target_link_libraries(wcn_math_benchmark wcn)
target_include_directories(wcn_math_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/include)

# FP Test
add_executable(wcn_math_fp Fp_test.c)
target_link_libraries(wcn_math_fp wcn)
target_include_directories(wcn_math_fp PRIVATE ${CMAKE_SOURCE_DIR}/include)