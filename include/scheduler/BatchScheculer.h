#pragma once

#include "InferenceScheduler.h"

template<class Runner>
class BatchScheculer : InferenceScheduler<Runner>
{
private:
    std::shared_ptr<Runner> backend_;
public:
    BatchScheculer(std::shared_ptr<Runner> backend) {

    }

    virtual std::future<OutputType> submit(const InputType& input) override {
        // 提交数据进入处理

        input
    }
};