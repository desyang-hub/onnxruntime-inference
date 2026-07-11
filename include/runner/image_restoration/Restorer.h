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
    using OutputType    = cv::Mat;
public:
    Restorer(const YAML::Node& config);

    template<class T>
    static std::unique_ptr<Restorer> Load(const std::string& config_path);

    virtual std::vector<cv::Mat> restoration(const std::vector<cv::Mat>& imgs) = 0;
    virtual cv::Mat restoration(const cv::Mat& img) = 0;
};

template<class T>
std::unique_ptr<Restorer> Restorer::Load(const std::string& config_path) {
    return std::make_unique<T>(YAML::LoadFile(config_path));
}