#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

#include "TensorBuffer.h"
#include "scheduler/type_trait/runner_type_trait.h"

/// @brief 阶段处理抽象
/// @tparam Input 输入类型
/// @tparam Output 输出类型
template<class Input, class Output>
class Stage {
protected:
    using InputType = Input;
    using OutputType = Output;
public:
    virtual ~Stage() = default;
    virtual OutputType execute(const InputType& input) = 0;
};

/// @brief 预处理阶段实现
/// @tparam Runner 推理器类型
template<class Runner>
class PreStage : public Stage<typename runner_type_trait<Runner>::InputType, TensorBuffer> {
private:
    std::shared_ptr<Runner> backend_;
public:
    PreStage(std::shared_ptr<Runner> backend) : backend_(backend) {}
    TensorBuffer execute(const typename runner_type_trait<Runner>::InputType& inp) override {
        return backend_->preprocess(inp); // CPU 密集
    }

#ifdef ENABLE_CUDA
    void execute(const typename runner_type_trait<Runner>::InputType& inp, TensorBuffer& tenbuf, int offset) {
        backend_->preprocess(inp, tenbuf, offset); // GPU批量预处理
    }
#endif
};

/// @brief 推理阶段实现
/// @tparam Runner 推理器类型
template<class Runner>
class InferStage : public Stage<TensorBuffer, ModelOutput> {
private:
    std::shared_ptr<Runner> backend_;
public:
    InferStage(std::shared_ptr<Runner> backend) : backend_(backend) {}
    ModelOutput execute(const TensorBuffer& inp) override {
        return backend_->infer(inp); // GPU
    }
};

/// @brief 后处理阶段实现
/// @tparam Runner 推理器类型
template<class Runner>
class PostStage : public Stage<ModelOutput, std::vector<typename runner_type_trait<Runner>::OutputType>> {
private:
    std::shared_ptr<Runner> backend_;
public:
    PostStage(std::shared_ptr<Runner> backend) : backend_(backend) {}
    std::vector<typename runner_type_trait<Runner>::OutputType>
    execute(const ModelOutput& inp) override {
        return backend_->postprocess(inp); // CPU 密集
    }
};