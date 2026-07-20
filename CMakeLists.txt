cmake_minimum_required(VERSION 3.21)

project(ONNXRuntime_infer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

option(BUILD_EXAMPLES "Build examples" ON)
message(STATUS "Build examples: ${BUILD_EXAMPLES}")
option(ENABLE_CUDA "Link against onnxruntime-gpu and enable CUDA provider" OFF)
message(STATUS "Enable CUDA: ${ENABLE_CUDA}")
option(BUILD_TEST "Build tests" ON)
message(STATUS "Build tests: ${BUILD_TEST}")



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
if(ENABLE_CUDA)
    set(ONNXRUNTIME_VERSION 1.25.0)
    enable_language(CUDA)
else()
    set(ONNXRUNTIME_VERSION 1.18.0)
endif()


# sha256sum build/_deps/onnxruntime_prebuilt-subbuild/onnxruntime_prebuilt-populate-prefix/src/onnxruntime-linux-x64-1.18.0.tgz => URL_HASH SHA256=...

# mirror: https://gh-proxy.com/<github-url>
if(WIN32)
    # 在 CMakeLists.txt 中添加
    if(MSVC)
        add_compile_options(/utf-8)
    endif()

    add_compile_definitions(NOMINMAX)

    if(ENABLE_CUDA)
        CPMAddPackage(
            NAME onnxruntime_prebuilt
            URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-gpu-${ONNXRUNTIME_VERSION}.zip
            URL_HASH SHA256=125c9fe408f41b9ae1ad7138dac5ebb19a85e65438d1e368d21b50e6abb32f4e
            DOWNLOAD_ONLY YES
        )
    else()
        CPMAddPackage(
            NAME onnxruntime_prebuilt
            URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip
            URL_HASH SHA256=a91af21ca8f9bdfa5a1aac3fdd0591384b4e2866d41612925f1758d5522829e7
            DOWNLOAD_ONLY YES
        )
    endif()
elseif(UNIX AND NOT APPLE)
    if(ENABLE_CUDA)
        CPMAddPackage(
            NAME onnxruntime_prebuilt
            URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz
            URL_HASH SHA256=630d1c292a8e530425bc4047598ce30acece2c4b5b5f4f46b676212302c38d93
            DOWNLOAD_ONLY YES
        )
    else()
        CPMAddPackage(
            NAME onnxruntime_prebuilt
            URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz
            URL_HASH SHA256=fa4d11b3fa1b2bf1c3b2efa8f958634bc34edc95e351ac2a0408c6ad5c5504f0
            DOWNLOAD_ONLY YES
        )
    endif()
else()
    message(FATAL_ERROR "Unsupported platform for ONNX Runtime prebuilt")
endif()


# 创建可直接 link 的导入目标, 从外部导入共享库 IMPORTED(外部导入) GLOBAL 项目全局可见
add_library(onnxruntime SHARED IMPORTED GLOBAL)
if(WIN32)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION             "${onnxruntime_prebuilt_SOURCE_DIR}/lib/onnxruntime.dll"
        IMPORTED_IMPLIB               "${onnxruntime_prebuilt_SOURCE_DIR}/lib/onnxruntime.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_prebuilt_SOURCE_DIR}/include"
    )
elseif(UNIX AND NOT APPLE)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION             "${onnxruntime_prebuilt_SOURCE_DIR}/lib/libonnxruntime.so"
        INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_prebuilt_SOURCE_DIR}/include"
    )
endif()

# 1. 查找 OpenMP 包
find_package(OpenMP REQUIRED)
# 2. 查找 OpenCV 包
find_package(OpenCV REQUIRED HINTS ${OpenCV_DIR})
# 3. 添加logger
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/setup_spdlog.cmake)

set(project_sources
    src/BufferPool.cpp
    src/backend/InferenceBackend.cpp
    src/backend/OrtSessionWrapper.cpp
    src/preprocess/utils.cpp
    src/runner/detect/YoloDetector.cpp
    src/runner/ModelRunner.cpp
    src/runner/detect/Detector.cpp
    src/runner/image_restoration/Restorer.cpp
    src/runner/image_restoration/NAFNet.cpp
    src/scheduler/stage/Stage.cpp
)

if(ENABLE_CUDA)
    set(project_device_sources 
        src/device/gpu_preprocess.cu
        src/device/gpu_filter.cu
    )
else()
    set(project_device_sources 
        ""
    )
endif()

# 编译链接库
set(INFER_LIB onnx_infer)
add_library(${INFER_LIB} ""
    ${project_sources}
    ${project_device_sources}
)

if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)

    target_compile_definitions(${INFER_LIB}
        PUBLIC ENABLE_CUDA
    )
    
    target_link_libraries(${INFER_LIB} 
        PUBLIC
        CUDA::cudart
    )

    # ⭐ 找到定义 YoloDetector 的库目标（根据你的项目结构，名字可能是 onnx_infer）
    #    把 .cu 加到这个库上，而不是加到 yolo_infer 可执行文件上
    target_sources(${INFER_LIB} PRIVATE 
        ${project_device_sources}
    )

    # ⭐ 路径必须与 target_sources 中完全一致！
    set_source_files_properties(
        ${project_device_sources}
        PROPERTIES LANGUAGE CUDA
    )
endif()

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
    OpenMP::OpenMP_CXX
    logger
)

add_executable(main 
    src/main.cpp
)

target_include_directories(main 
    PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(main 
    PUBLIC
    ${OpenCV_LIBS}
    onnxruntime
    logger
)

# 用于自动将可执行文件依赖的 dll 复制到输出目录
include(cmake/copy_runtime_dlls.cmake)
copy_runtime_dlls(main)

# 构建 exmples 子目录程序
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()


if(BUILD_TEST)
    enable_testing()
    add_subdirectory(tests)
endif()