# 自动计算当前源码目录相对于项目根目录的路径
# file(RELATIVE_PATH _rel_dir ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# # 将输出目录设置为 bin/<相对路径>/
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${_rel_dir})

add_executable(yolo_infer
    yolo_infer.cpp
)

target_link_libraries(yolo_infer
    ${INFER_LIB}
)

# ⭐ 正确做法：构建时动态复制，完美适配 Debug/Release
add_custom_command(TARGET yolo_infer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:yolo_infer>/config"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/config/model_config.yaml"
        "$<TARGET_FILE_DIR:yolo_infer>/config/model_config.yaml"
    COMMENT "[Config] Copying model_config.yaml for yolo_infer..."
)

add_executable(video_infer
    video_infer.cpp
)

target_link_libraries(video_infer
    ${INFER_LIB}
)

add_executable(batch_yolo_infer
    batch_yolo_infer.cpp
)

target_link_libraries(batch_yolo_infer
    ${INFER_LIB}
)

add_executable(logger_test
    logger_test.cpp
)

target_link_libraries(logger_test
    PUBLIC
    logger
)