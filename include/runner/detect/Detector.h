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

#include "runner/ModelRunner.h"


struct Detection {
    cv::Rect2f box;
    float score;
    int class_id;
};


class Detector : public ModelRunner
{
public:
    explicit Detector(std::unique_ptr<InferenceBackend> backend) : ModelRunner(std::move(backend)) {}
    virtual std::vector<Detection> detect(const cv::Mat& img) = 0;
};