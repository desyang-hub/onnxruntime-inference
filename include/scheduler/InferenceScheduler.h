#pragma once

#include <future>

#include "type_trait/runner_type_trait.h"

/**
 * @brief 推理调度器
 * 
 * @note 推理调度器是一个对于推理任务的抽象，推理过程中可以使用异步预处理、后处理，或者是等待批量推理等等，这些都是调度器所需要做的事情，提供接口，根据自己的使用需求自行实现
 * 
 * 2. 推荐的架构设计模式
    建议采用 接口 + 策略模式 + Future/Promise 模型：
    核心接口定义
 * 
 */

/// @brief 推理调度器
/// @tparam Runner 推理器
/// @note 推理调度器是一个对于推理任务的抽象，推理过程中可以使用异步预处理、后处理，或者是等待批量推理等等，这些都是调度器所需要做的事情，提供接口，根据自己的使用需求自行实现
template<class Runner>
class InferenceScheduler
{
protected:
    using InputType     = typename runner_type_trait<Runner>::InputType;
    using OutputType    = typename runner_type_trait<Runner>::OutputType;
public:
    virtual ~InferenceScheduler() = default;

    // 提交任务，返回Future，阻塞调用get
    virtual std::future<OutputType> submit(const InputType& input) = 0;
    // 优雅停止，任务退出
    virtual void shutdown() {}
};
