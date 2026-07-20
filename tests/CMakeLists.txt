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