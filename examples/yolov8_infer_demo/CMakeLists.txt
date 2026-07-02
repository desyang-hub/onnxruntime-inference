# 查找自己的相对位置
# file(RELATIVE_PATH _rel_dir ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# # 设置输出相对目录
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${_rel_dir})

# 添加目标文件
add_executable(yolov8_infer 
    yolov8_infer_demo.cpp
)

if(USE_CUDA)
    target_compile_definitions(yolov8_infer PRIVATE ENABLE_CUDA)
    # GPU 版和 CPU 版的库名相同，但安装路径/内容不同
    # 这里不需要额外 link 什么，因为符号在主库里声明、在 provider so 里实现
endif()

# 包含头文件
target_include_directories(yolov8_infer
    PRIVATE # 目标为可执行文件，直接用PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 链接库
target_link_libraries(yolov8_infer
    PRIVATE # 目标为可执行文件，直接用PRIVATE
    ${OpenCV_LIBS}
    ${INFER_LIB}
)

copy_runtime_dlls(yolov8_infer)