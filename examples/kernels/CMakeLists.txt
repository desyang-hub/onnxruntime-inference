add_executable(BenKernelBatchPiplineDemo
    BenKernelBatchPiplineDemo.cpp
)

target_link_libraries(BenKernelBatchPiplineDemo
    PRIVATE ${INFER_LIB}
)