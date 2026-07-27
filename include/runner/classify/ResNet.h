#pragma once

#include <opencv2/opencv.hpp>

#include "Classifier.h"
#include "device/cuda_utils.h"

class ResNet : public Classifier
{
private:
    CudaStream stream_;
    
public:
    ResNet(const YAML::Node& config);
    ~ResNet() = default;

    cv::Mat resize_256(const cv::Mat&) const;
    cv::Mat center_crop(const cv::Mat&, const cv::Size& size = cv::Size(224, 224)) const;


    TensorBuffer preprocess(const cv::Mat&) override;
#ifdef ENABLE_CUDA
    void preprocess(const cv::Mat&, TensorBuffer&, int offset) override;
#endif
    std::vector<int> postprocess(const ModelOutput&) override;
};