#pragma once

#include "Restorer.h"
#include "device/cuda_utils.h"

class NAFNet : public Restorer
{
private:
    size_t padder_size_;
    cv::Scalar pad_color_;
    float norm_scale_;
    bool bgr2rgb_;

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
    NAFNet(const YAML::Node& config);

    TensorBuffer preprocess(const cv::Mat&) override;
    std::vector<cv::Mat> postprocess(const ModelOutput&) override;

    TensorBuffer preprocess(const std::vector<cv::Mat>& imgs);

    std::vector<cv::Mat> postprocess(const TensorBuffer& tensor_buf, size_t batch);
};
