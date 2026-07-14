/**
 * @FilePath     : /onnxruntime-inference/include/runner/detect/YoloDetector.h
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-07-01 15:34:52
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-07 17:26:12
**/
#pragma once

#include <yaml-cpp/yaml.h>

#include "Detector.h"
#include "device/cuda_utils.h"
#include "TensorBuffer.h"

class YoloDetector : public Detector
{
public:
    using ModelRunner::infer;
private:
    YAML::Node config_;
    // model
    size_t warm_up_;

    // preprocess
    bool auto_aspect_ratio_;
    cv::Scalar pad_color_;
    float norm_scale_;
    bool bgr2rgb_;

    // postprocess
    float conf_threshold_;
    float nms_threshold_;
    size_t max_detections_;
    // classes:

    std::vector<std::string> labels_;

#ifdef ENABLE_CUDA
    // uint8_t* d_input_bgr_ = nullptr;
    std::shared_ptr<uint8_t> d_input_bgr_{};
    CudaStreamPtr cuStream_;
#endif
    
public:
    explicit YoloDetector(const YAML::Node& config);

    virtual TensorBuffer preprocess(const cv::Mat&) override;
    virtual std::vector<std::vector<Detection>> postprocess(const ModelOutput&) override;

    // TensorBuffer preprocess(const cv::Mat& img);
    // TensorBuffer preprocess(const std::vector<cv::Mat>& imgs);
};