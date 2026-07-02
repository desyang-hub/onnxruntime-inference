# 自动计算当前源码目录相对于项目根目录的路径
file(RELATIVE_PATH _rel_dir ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# 将输出目录设置为 bin/<相对路径>/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${_rel_dir})

add_executable(yolo_infer
    yolo_infer.cpp
)

target_link_libraries(yolo_infer
    ${INFER_LIB}
)

# ⭐ 关键修改：使用 TARGET_FILE_DIR 确保与 exe 同目录
configure_file(
    ${CMAKE_SOURCE_DIR}/config/model_config.yaml
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config/model_config.yaml
    COPYONLY
)