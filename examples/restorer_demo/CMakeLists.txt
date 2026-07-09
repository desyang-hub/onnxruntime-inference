# 自动计算当前源码目录相对于项目根目录的路径
# file(RELATIVE_PATH _rel_dir ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# # 将输出目录设置为 bin/<相对路径>/
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${_rel_dir})

add_executable(nafnet
    NAFNet_demo.cpp
)

target_link_libraries(nafnet
    ${INFER_LIB}
)

include(${PROJECT_SOURCE_DIR}/cmake/copy_dir_to_target.cmkae)
copy_config_dir_to_target(nafnet "${PROJECT_SOURCE_DIR}/config" "config")