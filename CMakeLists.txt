cmake_minimum_required(VERSION 3.10)

project(ONNXRuntime_infer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

option(BUILD_EXAMPLES "Build examples" ON)
message(STATUS "Build examples: ${BUILD_EXAMPLES}")
option(USE_CUDA "Link against onnxruntime-gpu and enable CUDA provider" OFF)



# 配置yaml-cpp
include(FetchContent)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG yaml-cpp-0.9.0 # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
)
FetchContent_MakeAvailable(yaml-cpp)

# 使用CPM
include(cmake/CPM.cmake)
# ONNX Runtime 1.18.0 Linux x64 CPU 预编译包
set(ONNXRUNTIME_VERSION 1.18.0)

# sha256sum build/_deps/onnxruntime_prebuilt-subbuild/onnxruntime_prebuilt-populate-prefix/src/onnxruntime-linux-x64-1.18.0.tgz => URL_HASH SHA256=...
if(WIN32)

    CPMAddPackage(
        NAME onnxruntime_prebuilt
        # URL https://gh-proxy.com/https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz
        https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip
        DOWNLOAD_ONLY YES
    )
elseif(UNIX AND NOT APPLE)
    CPMAddPackage(
        NAME onnxruntime_prebuilt
        URL https://gh-proxy.com/https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz
        # URL https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
        URL_HASH SHA256=fa4d11b3fa1b2bf1c3b2efa8f958634bc34edc95e351ac2a0408c6ad5c5504f0
        DOWNLOAD_ONLY YES
    )
else()
    message(FATAL_ERROR "Unsupported platform for ONNX Runtime prebuilt")
endif()




# 创建可直接 link 的导入目标, 从外部导入共享库 IMPORTED(外部导入) GLOBAL 项目全局可见
add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION             "${onnxruntime_prebuilt_SOURCE_DIR}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_prebuilt_SOURCE_DIR}/include"
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
)

target_link_libraries(main 
    ${OpenCV_LIBS}
    onnxruntime
    yaml-cpp::yaml-cpp
)


# 编译链接库
set(INFER_LIB infer_lib)
add_library(${INFER_LIB} ""
    src/backend/InferenceBackend.cpp
    src/runner/detect/YoloDetector.cpp
    src/runner/ModelRunner.cpp
)

target_include_directories(${INFER_LIB}
    PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${INFER_LIB} 
    PUBLIC
    ${OpenCV_LIBS}
    onnxruntime
    yaml-cpp::yaml-cpp
)


if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()