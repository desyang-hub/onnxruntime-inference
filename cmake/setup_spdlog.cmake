include_guard(GLOBAL)  # ← 防止被多次 include 时重复执行

# 1. 拉取 spdlog
include(FetchContent)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.15.0
)
set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(spdlog)

# 2. 编译 logger_core
add_library(logger_core STATIC
    ${CMAKE_CURRENT_LIST_DIR}/../include/logger/logger.cpp
)
target_include_directories(logger_core PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../include/
)
target_link_libraries(logger_core PUBLIC spdlog::spdlog)
target_compile_definitions(logger_core PUBLIC
    # SPDLOG_FMT_EXTERNAL=0
    # LOG_LEVEL_DEFAULT=2
    $<$<CONFIG:Debug>:LOG_LEVEL_DEFAULT=1>    # Debug 保留 debug
    $<$<CONFIG:Release>:LOG_LEVEL_DEFAULT=2>  # Release 从 info 开始
)

# 3. INTERFACE 包装层
add_library(logger INTERFACE)
target_link_libraries(logger INTERFACE logger_core)