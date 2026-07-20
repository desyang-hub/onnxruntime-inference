add_executable(buffer_pool_test
    buffer_pool_test.cpp
)

target_link_libraries(
    buffer_pool_test
    ${INFER_LIB}
)