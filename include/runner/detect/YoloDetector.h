/**
 * @FilePath     : /onnxruntime-infer/include/runner/detect/YoloDetector.h
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-07-01 15:34:52
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-02 11:07:03
**/
#pragma once

#include <yaml-cpp/yaml.h>

#include "Detector.h"
#include "TensorBuffer.h"

class ModelOutput;

class YoloDetector : public Detector
{
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

public:
    explicit YoloDetector(const std::string& config_path);

    TensorBuffer preprocess(const cv::Mat& img);
    std::vector<Detection> postprocess(const TensorBuffer&);

    const std::string& class_label(size_t id) const;


    virtual std::vector<Detection> detect(const cv::Mat& img) override;
};