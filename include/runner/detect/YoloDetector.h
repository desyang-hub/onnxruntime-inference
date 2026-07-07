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
#include <driver_types.h>

#include "Detector.h"
#include "device/cuda_utils.h"
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

#ifdef ENABLE_CUDA
    struct cuda_stream_deleter {
        void operator()(cudaStream_t stream) {
            if (stream) {
                cudaStreamSynchronize(stream); 
                cudaStreamDestroy(stream);
            }
        }
    };
    using CudaStreamPtr = std::unique_ptr<CUstream_st, cuda_stream_deleter>;

    std::vector<CudaStreamPtr> streams_;
#endif

public:
    explicit YoloDetector(const std::string& config_path);
    ~YoloDetector() = default;

    TensorBuffer preprocess(const cv::Mat& img);
    void preprocess(const std::vector<cv::Mat>& imgs);
    
    std::vector<Detection> postprocess(const TensorBuffer&);
    std::vector<std::vector<Detection>> postprocess(const TensorBuffer&, size_t batch);

    const std::string& class_label(size_t id) const;


    std::vector<Detection> detect(const cv::Mat& img) override;
    std::vector<std::vector<Detection>> detect(const std::vector<cv::Mat>& imgs);
};