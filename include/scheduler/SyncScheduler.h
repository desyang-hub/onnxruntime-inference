#pragma once

#include <memory>

#include "InferenceScheduler.h"
#include "thread/ThreadSafeQueue.h"
#include "type_trait/runner_type_trait.h"
#include "stage/Stage.h"


/// @brief 同步调度器，所有的任务同步进行，所有任务过程，必须要前面的任务完成了才可以开始下一个任务 
/// @tparam Runner 推理器类型
template<class Runner>
class SyncScheduler : public InferenceScheduler<Runner>
{
private:
    using InputType = typename runner_type_trait<Runner>::InputType;
    using OutputType = typename runner_type_trait<Runner>::OutputType;

    // ThreadSafeQueue<InputType> jobs_; // 任务的入口参数队列

    // 阶段处理器
    PreStage<Runner> pre_stage_;
    InferStage<Runner> infer_stage_;
    PostStage<Runner> post_stage_;


public:
    SyncScheduler(std::shared_ptr<Runner> backend) : pre_stage_(backend), infer_stage_(backend), post_stage_(backend) {}
    ~SyncScheduler() {
        shutdown();
    }

    // 提交任务，返回Future，阻塞调用get, 同步直接用功能方法即可，没必要搞那么复杂
    std::future<OutputType> submit(const InputType& input) override {
        std::promise<OutputType> p;
        auto fut = p.get_future();

        TensorBuffer tenbuf = pre_stage_.execute(input);
        ModelOutput model_out = infer_stage_.execute(tenbuf);
        
        p.set_value(post_stage_.execute(model_out));
        return fut;
    }

    // 优雅停止，任务退出
    void shutdown() {}
};