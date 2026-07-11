#pragma once

#include <functional>

/// @brief 抽象执行器
class Executor
{
public:
    using Task = std::function<void()>;
public:
    Executor() = default;
    virtual ~Executor() = default;

    // 异步任务
    virtual void submit(Task&& f) = 0;
};