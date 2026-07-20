/**
 * @FilePath     : /onnxruntime-infer/include/runner/detect/Detector.h
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-07-01 15:31:29
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-01 20:01:03
**/
#pragma once

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <vector>
#include <string>
#include <iostream>

#include "runner/ModelRunner.h"


struct Detection {
    cv::Rect2f box;
    float score;
    int class_id;
};


class Detector : public ModelRunner
{
public:
    using InputType = cv::Mat;
    using OutputType = std::vector<Detection>; // output 指代的一个结果，batch的推理结果是一个vector<OutputType>
private:
    std::vector<std::string> labels_;
public:
    explicit Detector(const YAML::Node& config);

    template<class T>
    static std::unique_ptr<Detector> Load(const std::string& cfg);

    virtual TensorBuffer preprocess(const InputType&) = 0;
#ifdef ENABLE_CUDA
    virtual void preprocess(const InputType&, TensorBuffer&, int offset) = 0;
#endif
    virtual std::vector<OutputType> postprocess(const ModelOutput&) = 0;

    OutputType detect(const cv::Mat& img);

    const std::string& class_label(size_t id) const;
};

template<class T>
std::unique_ptr<Detector> Detector::Load(const std::string& cfg) {
    static_assert(std::is_base_of_v<Detector, T> && "type must inherit Detector.");
    return std::make_unique<T>(YAML::LoadFile(cfg));
}