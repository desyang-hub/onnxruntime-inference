add_executable(pipline
    pipline.cpp
)

target_link_libraries(pipline
    PRIVATE
    ${INFER_LIB}
)

add_executable(pipline_multiple
    pipline_multiple.cpp
)

target_link_libraries(pipline_multiple
    PRIVATE
    ${INFER_LIB}
)