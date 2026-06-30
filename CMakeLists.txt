cmake_minimum_required(VERSION 3.10)

project(ONNXRuntime_infer)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 设置ONNXRUNTIME_ROOT路径
set(ONNXRUNTIME_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime)

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

include_directories(${ONNXRUNTIME_INCLUDE_DIR})

add_executable(main 
    src/main.cpp
)

target_link_libraries(main ${ONNXRUNTIME_LIBRARY}
)