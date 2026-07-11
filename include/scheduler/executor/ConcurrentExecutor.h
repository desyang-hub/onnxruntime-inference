#pragma once

#include "Executor.h"
#include "thread/ThreadPool.h"

/// @brief 并发执行器
class ConcurrentExecutor : public Executor
{
private:
    ThreadPool thread_pool_;
public:
    ConcurrentExecutor() : thread_pool_() {

    }

    void submit(Task f) override {
        thread_pool_.enqueue(std::move(f));
    }
};