#pragma once

#include <memory>

#include "stage/Stage.h"
#include "logger/logger.h"
#include "thread/ThreadPool.h"
#include "InferenceScheduler.h"
#include "backend/InferenceBackend.h"

/// @brief 异步推理调度器
/// @tparam Runner 推理器类型
template <class Runner>
class AsyncScheduler : public InferenceScheduler<Runner>
{
public:
    using InputType = typename runner_type_trait<Runner>::InputType;
    using OutputType = typename runner_type_trait<Runner>::OutputType;

private:
    int buffer_size_;
    ThreadPool pre_executor_;
    // 这里的并发量可以根据缓冲池大小来定义
    ThreadPool infer_executor_;
    ThreadPool post_executor_;

    // 阶段实例
    std::shared_ptr<Stage<InputType, TensorBuffer>> pre_stage_;
    std::shared_ptr<Stage<TensorBuffer, ModelOutput>> infer_stage_;
    std::shared_ptr<Stage<ModelOutput, std::vector<OutputType>>> post_stage_;

public:
    AsyncScheduler(std::shared_ptr<Runner> backend) : 
        buffer_size_(backend->getBufferSize()),
        pre_executor_(buffer_size_),
        infer_executor_(buffer_size_),
        post_executor_(buffer_size_),
        pre_stage_(std::make_shared<PreStage<Runner>>(backend)),
        infer_stage_(std::make_shared<InferStage<Runner>>(backend)),
        post_stage_(std::make_shared<PostStage<Runner>>(backend)) {}

    /// @brief 提交异步任务
    /// @param input
    /// @return
    std::future<OutputType> submit(const InputType &input) override
    {
        auto p = std::make_shared<std::promise<OutputType>>();
        std::future<OutputType> fut = p->get_future();

        pre_executor_.enqueue([this, p, input]() mutable
        {
            TensorBuffer tenbuf = pre_stage_->execute(input);
            // 异步提交直接结束

            infer_executor_.enqueue([this, p, tenbuf = std::move(tenbuf)]() mutable 
            {
                // 最后一个阶段任务
                ModelOutput model_out = infer_stage_->execute(tenbuf);

                post_executor_.enqueue([this, p, model_out = std::move(model_out)]() mutable 
                {
                    p->set_value(post_stage_->execute(model_out)[0]);
                });
            }); 
        });

        return fut;
    }
};