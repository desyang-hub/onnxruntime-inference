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
class IBench
{
public:
    IBench() = default;
    virtual ~IBench() = default;

    virtual LetterboxParams preprocess(const cv::Mat& img) {
        throw std::runtime_error(MESSAGE_WITH_LOC("Method preprocess(imgs) UnImplement!"));
    }

    virtual void infer() {
        throw std::runtime_error(MESSAGE_WITH_LOC("Method infer() UnImplement!"));
    }

    // yolo后处理
    virtual std::vector<std::vector<Detection>> postprocess(const std::vector<LetterboxParams>& params) {
        throw std::runtime_error(MESSAGE_WITH_LOC("Method postprocess(params) UnImplement!"));
    }

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