#pragma once

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <stdexcept>

#include "pipline/TaskContext.h"
#include "pipline/GPUBuffer.h"
#include "pipline/CPUBuffer.h"
#include "device/cuda_utils.h"
#include "runner/detect/Detector.h"
#include "logger/logger.h"
#include "preprocess/utils.h"
#include "ScopedTimer.h"
#include "exceptions/utils.h"
#include "IBench.h"

// kernel下
class Bench : public IBench
{
protected:
    // 最大支持图像的大小
    static constexpr size_t kMaxImageTotalElements{1024 * 1024 * 3};

    Ort::Env env_;
    TaskContext context_;
    GPUBuffer d_buffer_;
    CPUBuffer h_input_;
    CPUBuffer h_output_;

    CudaMallocGuard<uint8_t> gpu_store{kMaxImageTotalElements}; // 最大支持3M大小的图像
public:
    Bench(const YAML::Node& config) : 
        env_(ORT_LOGGING_LEVEL_ERROR, "KernelBase"),
        context_(config, env_),
        d_buffer_(context_),
        h_input_(context_.num_input_bytes_size),
        h_output_(context_.num_output_bytes_size) {

        context_.warm_up(d_buffer_);
    }

    Bench(TaskContext&& context) : 
        context_(std::move(context)),         
        d_buffer_(context_),
        h_input_(context_.num_input_bytes_size),
        h_output_(context_.num_output_bytes_size) {
        context_.warm_up(d_buffer_);
    }
};