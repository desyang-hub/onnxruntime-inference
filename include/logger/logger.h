/**
 * @file Logger.hpp
 * @brief 日志封装（基于 spdlog）
 */
#pragma once

#include <string>
#include <spdlog/spdlog.h> 

namespace logger {

enum LoggerLevel {
    LOGLEVEL_DEBUG,
    LOGLEVEL_WARN,
    LOGLEVEL_ERROR,
    LOGLEVEL_CRITICAL,
    LOGLEVEL_TRACE,
    LOGLEVEL_INFO
};

/**
 * @brief 初始化日志系统
 * @note  首次调用时自动创建，后续调用仅更新配置
 * @param log_level     日志级别: debug/info/warn/error/critical
 * @param log_file      日志文件路径（空字符串则仅输出到控制台）
 * @param max_file_size 单文件最大字节数（默认 10MB）
 * @param max_files     轮转保留文件数（默认 3）
 */
void Init(LoggerLevel log_level = LOGLEVEL_INFO,
          const std::string& log_file = "",
          size_t max_file_size = 10 * 1024 * 1024,
          int max_files = 3);

/**
 * @brief 运行时动态修改日志级别
 */
void SetLevel(LoggerLevel log_level);

std::shared_ptr<spdlog::logger>& GetInstance();

} // namespace logger

// ============================================================
// 便捷日志宏
// 使用方式: LOG_INFO("Model loaded, fps={}", fps);
// ============================================================
#define LOG_TRACE(...)    SPDLOG_TRACE(__VA_ARGS__)
#define LOG_DEBUG(...)    SPDLOG_DEBUG(__VA_ARGS__)
#define LOG_INFO(...)     SPDLOG_INFO(__VA_ARGS__)
#define LOG_WARN(...)     SPDLOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...)    SPDLOG_ERROR(__VA_ARGS__)
#define LOG_CRITICAL(...) SPDLOG_CRITICAL(__VA_ARGS__)

// 带源码位置信息的日志宏（调试用）
#define LOG_DEBUG_LOC(...) SPDLOG_DEBUG("{}:{} [{}] " __VA_ARGS__, __FILE__, __LINE__, __FUNCTION__)
#define LOG_INFO_LOC(...)  SPDLOG_INFO("{}:{} [{}] " __VA_ARGS__, __FILE__, __LINE__, __FUNCTION__)
#define LOG_WARN_LOC(...)  SPDLOG_WARN("{}:{} [{}] " __VA_ARGS__, __FILE__, __LINE__, __FUNCTION__)
#define LOG_ERROR_LOC(...) SPDLOG_ERROR("{}:{} [{}] " __VA_ARGS__, __FILE__, __LINE__, __FUNCTION__)