# 自动计算当前源码目录相对于项目根目录的路径
# file(RELATIVE_PATH _rel_dir ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# # 将输出目录设置为 bin/<相对路径>/
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${_rel_dir})

add_executable(yaml_usage
    yaml_use.cpp
)

target_link_libraries(yaml_usage
    yaml-cpp::yaml-cpp
)


add_custom_command(TARGET yaml_usage POST_BUILD
    # 1. 确保目标目录存在（防止因文件夹不存在导致 copy 失败）
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:yaml_usage>/config"

    # 2. 智能复制文件
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/config/config.yaml"
        "$<TARGET_FILE_DIR:yaml_usage>/config/config.yaml"
    COMMENT "[Config] Copying config.yaml for yaml_usage..."
)