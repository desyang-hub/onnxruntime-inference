#pragma once

#include <queue>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <thread>
#include <future>
#include <utility>
#include <functional>

using JobHandler = std::function<void()>;

class ThreadPool
{
private:
    std::vector<std::thread> workers_;
    std::queue<JobHandler> que_;
    std::mutex que_mutex_;
    std::condition_variable condition_;
    bool is_running;

public:
    explicit ThreadPool(int pool_size = 4);
    ~ThreadPool();

public:
    template<class F, typename ...Args>
    auto enqueue(F&& f, Args&& ...args) -> std::future<typename std::result_of<F(Args...)>::type>;
};


inline ThreadPool::ThreadPool(int pool_size) : is_running(true) {
    workers_.reserve(pool_size);

    for (int i = 0; i < pool_size; ++i) {
        workers_.emplace_back([this]{

            JobHandler job;

            while (true) {
                {
                    std::unique_lock<std::mutex> lock(que_mutex_);

                    condition_.wait(lock, [this]{
                        return !is_running || !que_.empty();
                    });

                    if (!que_.empty()) {
                        job = std::move(que_.front());
                        que_.pop();
                    }
                    else {
                        break;
                    }
                }

                job(); // 执行，任务需要自行处理异常
            }
        });
    }
}

inline ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(que_mutex_);
        is_running = false;
    }

    condition_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

template<class F, typename ...Args>
auto ThreadPool::enqueue(F&& f, Args&& ...args) -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    // 构造一个异步任务
    auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();

    {
        std::lock_guard<std::mutex> lock(que_mutex_);

        if (!is_running) {
            throw std::runtime_error("enqueue in thread pool stoped");
        }

        que_.push([task]{
            (*task)();
        });
    }

    condition_.notify_one();

    return res;
}