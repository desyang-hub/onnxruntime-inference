# Benchmark tool target

add_executable(benchmark
    main.cpp
    benchmark_runner.cpp
)

target_include_directories(benchmark
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(benchmark
    PRIVATE
    ${INFER_LIB}
)

# Copy config directory next to the benchmark binary
include(${PROJECT_SOURCE_DIR}/cmake/copy_dir_to_target.cmkae)
copy_config_dir_to_target(benchmark "${PROJECT_SOURCE_DIR}/config" "config")

# Copy benchmark YAML config
add_custom_command(TARGET benchmark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_config.yaml"
        "$<TARGET_FILE_DIR:benchmark>/benchmark_config.yaml"
    COMMENT "[Benchmark] Copying benchmark_config.yaml..."
)
