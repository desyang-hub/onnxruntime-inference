#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "TensorBuffer.h"
#include "runner/ModelRunner.h"
#include "backend/InferenceBackend.h"

class Classifier : public ModelRunner
{
private:
    using InputType = cv::Mat;
    using OutputType = int;

    std::vector<std::string> labels_;
public:
    Classifier(const YAML::Node& config);
    ~Classifier() = default;

    template<class T>
    static std::shared_ptr<Classifier> Load(const std::string& cfg);

    virtual TensorBuffer preprocess(const InputType&) = 0;
#ifdef ENABLE_CUDA
    virtual void preprocess(const InputType&, TensorBuffer&, int offset) = 0;
#endif
    virtual std::vector<OutputType> postprocess(const ModelOutput&) = 0;

    OutputType classify(const cv::Mat& img);

    const std::string& class_label(size_t id) const;
};
