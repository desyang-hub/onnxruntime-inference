# 自动计算当前源码目录相对于项目根目录的路径
file(RELATIVE_PATH _rel_dir ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# 将输出目录设置为 bin/<相对路径>/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${_rel_dir})

add_executable(yaml_usage
    yaml_use.cpp
)

target_link_libraries(yaml_usage
    yaml-cpp::yaml-cpp
)



# ⭐ 关键修改：使用 TARGET_FILE_DIR 确保与 exe 同目录
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config/config.yaml
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config/config.yaml
    COPYONLY
)