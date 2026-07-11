/**
 * @FilePath     : /onnxruntime-inference/include/logger/logger.cpp
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-07-09 19:34:42
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-09 20:40:30
**/
#include "logger.h"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/common.h>

#include <memory>
#include <vector>
#include <cstdio>
#include <iostream>
#include <stdlib.h>

namespace logger {

namespace {
spdlog::level::level_enum ParseLevel(LoggerLevel level) {
    if (level == LOGLEVEL_DEBUG)    return spdlog::level::debug;
    if (level == LOGLEVEL_WARN)     return spdlog::level::warn;
    if (level == LOGLEVEL_ERROR)    return spdlog::level::err;
    if (level == LOGLEVEL_CRITICAL) return spdlog::level::critical;
    if (level == LOGLEVEL_TRACE)    return spdlog::level::trace;
    return static_cast<spdlog::level::level_enum>(SPDLOG_ACTIVE_LEVEL);
}

} // anonymous namespace


/**
 * @brief 获取内部 logger 实例（Meyers' Singleton）
 * @note  C++11 保证线程安全的懒初始化；
 *        程序退出时 shared_ptr 自动析构 → spdlog 自动 flush + shutdown。
 */
std::shared_ptr<spdlog::logger>& GetInstance() {
    static std::shared_ptr<spdlog::logger> instance = [] {
        try {
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_level(spdlog::level::trace);

            std::vector<spdlog::sink_ptr> sinks{console_sink};

            auto lgr = std::make_shared<spdlog::logger>("app", sinks.begin(), sinks.end());
            lgr->set_pattern("[%H:%M:%S.%e] [%^%L%$] [thread %t] %v");
            lgr->set_level(spdlog::level::info);
            lgr->flush_on(spdlog::level::info); // warn 及以上立即刷新
            return lgr;
        } catch (const spdlog::spdlog_ex& ex) {
            fprintf(stderr, "[Logger] Fallback: %s\n", ex.what());
            return spdlog::stderr_color_mt("fallback");
        }
    }();
    return instance;
}

void Init(LoggerLevel log_level,
    const std::string& log_file,
    size_t max_file_size,
    int max_files) {
      
    auto& lgr = GetInstance();
    auto spd_log_level = ParseLevel(log_level);

    std::vector<spdlog::sink_ptr> sinks;

    // ✅ 关键修复：必须为每个 sink 单独设置级别
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spd_log_level); 
    sinks.push_back(console_sink);

    if (!log_file.empty()) {
    try {
        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            log_file, max_file_size, max_files);
        file_sink->set_level(spd_log_level); // ✅ 文件 sink 同理
        sinks.push_back(file_sink);
    } catch (const spdlog::spdlog_ex& ex) {
        fprintf(stderr, "[Logger] File sink failed: %s\n", ex.what());
    }
    }

    lgr->sinks() = std::move(sinks);
    lgr->set_level(spd_log_level);

    // ⚠️ 再次强调：不要对 debug 级别使用 flush_on
    lgr->flush_on(spdlog::level::warn); 

    // ✅ 关键修复：将自定义 logger 设为全局默认
    spdlog::set_default_logger(lgr);
}

void SetLevel(LoggerLevel level) {
    GetInstance()->set_level(ParseLevel(level));
}

} // namespace logger