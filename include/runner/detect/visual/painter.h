/**
 * @FilePath     : /onnxruntime-infer/include/runner/detect/visual/painter.h
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-07-02 10:10:50
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-02 10:45:35
**/
#pragma once

#include <opencv2/opencv.hpp>

#include "runner/detect/Detector.h"

static const cv::Scalar GREEN   = {0, 255, 0};
static const cv::Scalar BLUE    = {255, 0, 0};
static const cv::Scalar RED     = {0, 0, 255};

inline cv::Mat& draw_box(cv::Mat& img, const Detection& det, const cv::Scalar &color = GREEN, int thickness = 1, int lineType = 8, int shift = 0) {
    cv::rectangle(img, det.box, color, thickness, lineType, shift);

    cv::putText(img, std::to_string(det.class_id),  {static_cast<int>(det.box.x), static_cast<int>(det.box.y)}, cv::FONT_HERSHEY_SIMPLEX, 1.0, color, thickness, cv::LINE_AA);

    return img;
}

inline cv::Mat& draw_boxs(cv::Mat& img, const std::vector<Detection>& dets, const cv::Scalar &color = GREEN, int thickness = 1, int lineType = 8, int shift = 0) {

    for (const auto& det : dets) {
        draw_box(img, det, color, thickness, lineType, shift);
    }

    return img;
}