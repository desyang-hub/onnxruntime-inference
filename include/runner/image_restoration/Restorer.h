#pragma once

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <memory>
#include <string>

#include "runner/ModelRunner.h"

class Restorer : public ModelRunner
{
public:
    using InputType     = cv::Mat;
    using OutputType    = std::vector<cv::Mat>;
public:
    Restorer(const YAML::Node& config);

    template<class T>
    static std::unique_ptr<Restorer> Load(const std::string& config_path);

    virtual TensorBuffer preprocess(const cv::Mat&) = 0;
    virtual std::vector<cv::Mat> postprocess(const ModelOutput&) = 0;

#ifdef ENABLE_CUDA
    virtual void preprocess(const InputType&, TensorBuffer&, int offset) = 0;
#endif

    cv::Mat restoration(const cv::Mat& img);
    // std::vector<cv::Mat> restoration(const std::vector<cv::Mat>& imgs);
};

template<class T>
std::unique_ptr<Restorer> Restorer::Load(const std::string& config_path) {
    return std::make_unique<T>(YAML::LoadFile(config_path));
}