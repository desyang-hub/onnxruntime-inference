#pragma once

#include <memory>
#include <mutex>
#include <thread>
#include <queue>
#include <functional>
#include <stdexcept>
#include <condition_variable>

#include "InferenceScheduler.h"
#include "thread/ThreadSafeQueue.h"
#include "type_trait/runner_type_trait.h"
#include "stage/Stage.h"
#include "logger/logger.h"


/// @brief 同步调度器，所有的任务同步进行，所有任务过程，必须要前面的任务完成了才可以开始下一个任务 
/// @tparam Runner 推理器类型
template<class Runner>
class SyncScheduler : public InferenceScheduler<Runner>
{
private:
    using InputType = typename runner_type_trait<Runner>::InputType;
    using OutputType = typename runner_type_trait<Runner>::OutputType;
    using Job = std::function<void()>;

    // 阶段处理器
    PreStage<Runner> pre_stage_;
    InferStage<Runner> infer_stage_;
    PostStage<Runner> post_stage_;

    bool is_close_;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::queue<Job> tasks_;

    void run() {
        while (true) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(mutex_);

                condition_.wait(lock, [this]{
                    return !tasks_.empty() || is_close_;
                });

                if (!tasks_.empty()) {
                    job = std::move(tasks_.front());
                    tasks_.pop();
                }
                else {
                    break;
                }
            }

            // 使用packaged_task 包装，无需捕获，调用get后重新抛出
            job();
        }
    }

public:
    SyncScheduler(std::shared_ptr<Runner> backend) : 
        pre_stage_(backend), infer_stage_(backend), post_stage_(backend), is_close_(false) 
    {
        worker_ = std::thread(&SyncScheduler::run, this); // 启动后台处理线程
    }

    ~SyncScheduler() {
        shutdown();
    }

    // 提交任务，返回Future，阻塞调用get, 同步直接用功能方法即可，没必要搞那么复杂
    std::future<OutputType> submit(const InputType& input) override {

        std::lock_guard<std::mutex> lock(mutex_);
        if (is_close_) {
            LOG_DEBUG("Scheduler submit in stoped!");
            throw std::runtime_error("Scheduler submit in stoped!");
        }

        auto task = std::make_shared<std::packaged_task<OutputType()>>(
            [this, input]() -> OutputType {
                auto pre_out = pre_stage_.execute(input);
                auto infer_out = infer_stage_.execute(pre_out);
                return post_stage_.execute(infer_out)[0]; // 这里是同步任务，每次只后处理只有一个结果
            }
        );

        std::future<OutputType> fut = task->get_future();

        tasks_.push([task]{
            (*task)();
        });

        condition_.notify_one();

        return fut;
    }

    // 优雅停止，任务退出
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            is_close_ = true;
        }
        condition_.notify_all();

        // 回收线程
        if (worker_.joinable()) {
            worker_.join();
        }
    }
};