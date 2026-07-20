#pragma once

#include <memory>

#include "Restorer.h"
#include "BufferPool.h"
#include "device/cuda_utils.h"

class NAFNet : public Restorer
{
private:
    size_t padder_size_;
    cv::Scalar pad_color_;
    float norm_scale_;
    bool bgr2rgb_;

#ifdef ENABLE_CUDA
    std::vector<CudaStreamPtr> streams_;
    std::unique_ptr<BufferPool> cpu_buffer_pool_;
#endif

public:
    NAFNet(const YAML::Node& config);

    TensorBuffer preprocess(const cv::Mat&) override;
    std::vector<cv::Mat> postprocess(const ModelOutput&) override;

#ifdef ENABLE_CUDA
    void preprocess(const cv::Mat&, TensorBuffer&, int offset) override;
#endif

    TensorBuffer preprocess(const std::vector<cv::Mat>& imgs);

    std::vector<cv::Mat> postprocess(const TensorBuffer& tensor_buf, size_t batch);
};
