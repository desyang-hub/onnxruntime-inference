#include <chrono>
#include <string>
#include <functional>
#include "logger/logger.h"

// 如果你的项目有自定义日志宏，替换下面的 LOG_TRACE 即可
// #ifndef LOG_TRACE
// #include <cstdio>
// #define LOG_TRACE(fmt, ...) std::printf("[TRACE] " fmt "\n", ##__VA_ARGS__)
// #endif

class ScopedTimer {
public:
    // 构造时记录起始时间 + 标签名
    explicit ScopedTimer(std::string name)
        : m_name(std::move(name))
        , m_start(std::chrono::steady_clock::now())
    {}

    // 禁止拷贝和移动，防止重复打印或悬垂引用
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;

    // 析构时自动计算并打印耗时
    ~ScopedTimer() {
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - m_start).count();
        LOG_TRACE("{} | elapsed={} ms", m_name.c_str(), ms);
    }

    // 可选：手动获取当前已消耗的时间（用于中间检查点）
    double elapsed_ms() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(now - m_start).count();
    }

private:
    std::string m_name;
    std::chrono::steady_clock::time_point m_start;
};