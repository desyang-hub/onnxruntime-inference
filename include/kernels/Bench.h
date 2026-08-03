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

// kernel下
class Bench
{
protected:
    // 最大支持图像的大小
    const size_t kMaxImageTotalElements{1024 * 1024 * 3};

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

    }

    virtual LetterboxParams preprocess(const cv::Mat& img) = 0;

    virtual void infer() = 0;

    // yolo后处理
    virtual std::vector<std::vector<Detection>> postprocess(const std::vector<LetterboxParams>& params) = 0;

    std::vector<Detection> detect(const cv::Mat& img) {
        std::vector<LetterboxParams> params;
        // ScopedTimer st("pre");
        auto param = preprocess(img);
        // auto param = preprocess(img, h_input_.data());
        // LOG_INFO("pre time: {}", st.elapsed_ms());

        params.push_back(std::move(param));

        // ScopedTimer st1("infer");
        infer();
        // LOG_INFO("kernel time: {}", st1.elapsed_ms());
        cudaStreamSynchronize(0);
        
        // ScopedTimer st2("infer");
        auto res = postprocess(params);
        // LOG_INFO("post time: {}", st2.elapsed_ms());
        // LOG_INFO("res size: {}", res.size());

        return res[0];
    }

    virtual std::vector<std::vector<Detection>> batch_detect(const std::vector<cv::Mat>& img) {
        throw std::runtime_error(MESSAGE_WITH_LOC("Method detect(imgs) UnImplement!"));
    }
};