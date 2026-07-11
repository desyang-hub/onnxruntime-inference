#pragma once

#include <thread>
#include <atomic>
#include <queue>
#include <future>
#include <stdexcept>

#include "Executor.h"
#include "thread/ThreadSafeQueue.h"
#include "logger/logger.h"

/// @brief 同步执行器 `内部所有的任务都将在一个线程中先后执行`
class SyncExecutor : public Executor
{
private:
    std::queue<Task> tasks_;
    std::mutex que_mutex_;
    std::condition_variable condition_;
    bool is_close;
    std::thread worker_;
    int max_task_len_;
    
    
public:
    SyncExecutor() : is_close(false) {
        // 启动单线程用于处理任务
        worker_ = std::thread([this]{
            while (true) {
                Task f;
                {
                    std::unique_lock<std::mutex> lock(que_mutex_);
                    condition_.wait(lock, [this]{
                        return !tasks_.empty() || is_close;
                    });

                    if (is_close && tasks_.empty()) break;
                    f = std::move(tasks_.front());
                    tasks_.pop();
                }

                try
                {
                    f();
                }
                catch(const std::exception& e)
                {
                    LOG_ERROR_LOC("throw exception {}", e.what());
                }
            }
        });
    }

    ~SyncExecutor() {
        {
            std::lock_guard<std::mutex> lock(que_mutex_);
            is_close = true;
        }
        condition_.notify_all();

        if (worker_.joinable()) {
            worker_.join();
        }
    }

    // 阻塞进行
    void submit(Task&& f) override {
        {
            std::lock_guard<std::mutex> lock(que_mutex_);
            if (is_close) {
                throw std::runtime_error("submit in exector stoped.");
            }

            tasks_.push(std::move(f));
        }
        condition_.notify_one();
    }
};
