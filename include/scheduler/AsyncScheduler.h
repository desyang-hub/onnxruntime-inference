#pragma once

#include <memory>

#include "InferenceScheduler.h"
#include "thread/ThreadSafeQueue.h"


#include "backend/InferenceBackend.h"
#include "stage/Stage.h"
#include "executor/SyncExecutor.h"
#include "executor/ConcurrentExecutor.h"
#include "logger/logger.h"

/// @brief 异步推理调度器
/// @tparam Runner 推理器类型
template<class Runner>
class AsyncScheduler : public InferenceScheduler<Runner>
{
public:
    using InputType = typename runner_type_trait<Runner>::InputType;
    using OutputType = typename runner_type_trait<Runner>::OutputType;
private:
    // 三个阶段的独立线程池/执行器
    std::unique_ptr<Executor> pre_executor_;   // CPU 线程池
    std::unique_ptr<Executor> infer_executor_; // GPU 串行/有限并发
    std::unique_ptr<Executor> post_executor_;  // CPU 线程池

    // 阶段实例
    std::shared_ptr<Stage<InputType, TensorBuffer>>     pre_stage_;
    std::shared_ptr<Stage<TensorBuffer, ModelOutput>>  infer_stage_;
    std::shared_ptr<Stage<ModelOutput, OutputType>>    post_stage_;

public:
    AsyncScheduler(std::shared_ptr<Runner> backend) : 
        pre_executor_(std::make_unique<ConcurrentExecutor>()),
        infer_executor_(std::make_unique<SyncExecutor>()),
        post_executor_(std::make_unique<ConcurrentExecutor>()),
        pre_stage_(std::make_shared<PreStage<Runner>>(backend)),
        infer_stage_(std::make_shared<InferStage<Runner>>(backend)),
        post_stage_(std::make_shared<PostStage<Runner>>(backend)) {}

    /// @brief 提交异步任务
    /// @param input 
    /// @return 
    std::future<OutputType> submit(const InputType& input) override {
        auto p = std::make_shared<std::promise<OutputType>>();
        std::future<OutputType> fut = p.get_future();

        std::packaged_task<void()> task([this,  p, input]() mutable {
            TensorBuffer tenbuf = pre_stage_->execute(input);
            // 异步提交直接结束
            std::packaged_task<void()> task([this, p, tenbuf = std::move(tenbuf)]() mutable {
                // 最后一个阶段任务
                ModelOutput model_out = infer_stage_->execute(tenbuf);
                std::packaged_task<void()> task([this, p, model_out = std::move(model_out)]() mutable {
                    OutputType out = post_stage_->execute(model_out);
                    p->set_value(std::move(out));
                });
                post_executor_->submit(std::move(task));
            });
            infer_executor_->submit(std::move(task));
        });

        // 提交任务
        pre_executor_->submit(std::move(task));

        return fut;
    }
};