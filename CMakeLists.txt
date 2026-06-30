cmake_minimum_required(VERSION 3.10)

project(ONNXRuntime_infer)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 设置ONNXRUNTIME_ROOT路径
set(ONNXRUNTIME_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime)

option(USE_CUDA "Link against onnxruntime-gpu and enable CUDA provider" OFF)

find_path(ONNXRUNTIME_INCLUDE_DIR 
    NAMES onnxruntime_c_api.h
    PATHS ${ONNXRUNTIME_ROOT}/include
    NO_DEFAULT_PATH REQUIRED
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS ${ONNXRUNTIME_ROOT}/lib
    NO_DEFAULT_PATH REQUIRED
)


find_package(OpenCV REQUIRED HINTS ${OpenCV_DIR})

add_executable(main 
    src/main.cpp
)

if(USE_CUDA)
    target_compile_definitions(main PRIVATE ENABLE_CUDA)
    # GPU 版和 CPU 版的库名相同，但安装路径/内容不同
    # 这里不需要额外 link 什么，因为符号在主库里声明、在 provider so 里实现
endif()

target_include_directories(main
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(main 
    ${ONNXRUNTIME_LIBRARY}
    ${OpenCV_LIBS}
)



add_executable(yolov8_infer 
    src/yolov8_infer_demo.cpp
)

if(USE_CUDA)
    target_compile_definitions(yolov8_infer PRIVATE ENABLE_CUDA)
    # GPU 版和 CPU 版的库名相同，但安装路径/内容不同
    # 这里不需要额外 link 什么，因为符号在主库里声明、在 provider so 里实现
endif()

target_include_directories(yolov8_infer
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(yolov8_infer 
    ${ONNXRUNTIME_LIBRARY}
    ${OpenCV_LIBS}
)