add_executable(buffer_pool_test
    buffer_pool_test.cpp
)

target_link_libraries(
    buffer_pool_test
    ${INFER_LIB}
)

if(ENABLE_CUDA)
    add_executable(batch_scheduler_test
        batch_scheduler_test.cpp
    )

    target_link_libraries(batch_scheduler_test
        ${INFER_LIB}
    )
endif()


add_executable(concurrent_queue_test
    concurrent_queue_test.cpp
)

target_link_libraries(concurrent_queue_test
    ${INFER_LIB}
)

add_executable(kernel_test
    kernel_test.cpp
)


target_link_libraries(kernel_test
    ${INFER_LIB}
)