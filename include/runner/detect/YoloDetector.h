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
#include "BufferPool.h"

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
    int max_detections_;
    // classes:

    int num_attributes_;   // 84
    int num_predictions_;  // 8400

    std::vector<std::string> labels_;

#ifdef ENABLE_CUDA
    // uint8_t* d_input_bgr_ = nullptr;
    std::shared_ptr<uint8_t> d_input_bgr_{};
    CudaStreamPtr cuStream_;
    
    std::unordered_map<uint8_t*, CudaStreamPtr> cuStreams_;
    // 现在要的是缓冲池
    InferTensorBufferPoolPtr pool_; // 每次取出空闲指针来用即可

    std::unordered_map<float*, CudaStreamPtr> filter_streams_;
    InferTensorBufferPoolPtr d_filtered_buffers_;
    InferTensorBufferPoolPtr d_count_ptrs_;

    BufferPoolPtr cpu_buffer_pool_;

#endif
    
public:
    explicit YoloDetector(const YAML::Node& config);

    virtual TensorBuffer preprocess(const cv::Mat&) override;
#ifdef ENABLE_CUDA
    virtual void preprocess(const cv::Mat&, TensorBuffer&, int offset) override;
#endif
    virtual std::vector<std::vector<Detection>> postprocess(const ModelOutput&) override;

    // TensorBuffer preprocess(const cv::Mat& img);
    // TensorBuffer preprocess(const std::vector<cv::Mat>& imgs);
};