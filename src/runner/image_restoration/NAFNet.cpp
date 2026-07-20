#include <omp.h>
#include <memory>

#include "device/cuda_utils.h"
#include "preprocess/utils.h"
#include "runner/image_restoration/NAFNet.h"
#include "logger/logger.h"

NAFNet::NAFNet(const YAML::Node& config) : Restorer(config["model"]) {
    padder_size_ = config["model"]["padder_size"].as<size_t>(16);
    std::vector<int> pc = 
        config["preprocess"]["pad_color"]
        .as<std::vector<int>>(std::vector<int>{114, 114, 114});
    pad_color_          = cv::Scalar(pc[0], pc[1], pc[2]);
    norm_scale_         = 1.0f / 255.0f;

    bgr2rgb_            = config["preprocess"]["bgr_to_rgb"].as<bool>(true);

#ifdef ENABLE_CUDA
    int batch_size = getBatchSize();
    streams_.resize(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        cudaStream_t stream{};
        cudaStreamCreate(&stream);
        streams_[i].reset(stream);
    }

    int buffer_size = getBufferSize();
    auto& output_shapes = getOutputShapes();
    size_t num_output_elements = std::accumulate(output_shapes.begin(), output_shapes.end(), size_t{1}, std::multiplies<size_t>());

    cpu_buffer_pool_ = std::make_unique<BufferPool>(buffer_size, num_output_elements * sizeof(float));
    cpu_rgb_buffer_pool_ = 
    std::make_unique<BufferPool>(buffer_size, num_output_elements / batch_size  * sizeof(float));

    single_shapes_ = backend_->shapes();
    single_shapes_[0] = 1;
#endif 
}


TensorBuffer NAFNet::preprocess(const cv::Mat& img) {
    // ====== 1. 预处理 ======
    // BGR → RGB, HWC → CHW, uint8 [0,255] → float32 [0,1]
    auto& shape = backend_->shapes();

    // cpu空间也可以预分配，避免重复建立空间
    TensorBuffer buf = backend_->tensorBuffer();
    const cv::Size size(buf.shape[3], buf.shape[2]);
    size_t plane_size = buf.plane_size();

    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    buf.letterbox_params[0] = pad_to_size(rgb_img, rgb_img, size);

    LOG_TRACE("ptr: {}", fmt::ptr(buf.data));

    convert_and_normalize(rgb_img, buf.data, size, norm_scale_, bgr2rgb_);

    
#ifdef ENABLE_CUDA
    // 将数据拷贝到GPU
    float* data = backend_->GetTensorBuffer().data;
    cudaMemcpyAsync(data, buf.data, buf.num_elements * sizeof(float), cudaMemcpyHostToDevice, streams_[0].get());
    cudaStreamSynchronize(streams_[0].get());
    buf.data = data;
#endif


    return buf;
}


#ifdef ENABLE_CUDA
void NAFNet::preprocess(const cv::Mat& img, TensorBuffer& tenbuf, int offset) {
        // ====== 1. 预处理 ======
    // BGR → RGB, HWC → CHW, uint8 [0,255] → float32 [0,1]

    if (!tenbuf.valid()) {
        LOG_DEBUG("benbuf is invalid!");
        throw std::runtime_error("benbuf is invalid!");
    }

    LOG_TRACE("tenbuf data ptr: {}", fmt::ptr(tenbuf.data));

    // cpu空间也可以预分配，避免重复建立空间
    auto cpu_buffer = cpu_rgb_buffer_pool_->Acquire();
    
    TensorBuffer buf = TensorBuffer::wrap(cpu_buffer.get(), single_shapes_);
    const cv::Size size(buf.shape[3], buf.shape[2]);
    size_t plane_size = buf.plane_size();

    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    buf.letterbox_params[0] = pad_to_size(rgb_img, rgb_img, size);

    LOG_TRACE("ptr: {}", fmt::ptr(buf.data));

    convert_and_normalize(rgb_img, buf.data, size, norm_scale_, bgr2rgb_);

    cudaMemcpyAsync(tenbuf.data + offset * tenbuf.plane_size(), 
    buf.data, buf.num_elements * sizeof(float), cudaMemcpyHostToDevice, streams_[offset].get());
    cudaStreamSynchronize(streams_[offset].get());
}
#endif


std::vector<cv::Mat> NAFNet::postprocess(const ModelOutput& model_out) {
    // 先将数据拷贝到cpu
    auto tensor_buf = std::move(model_out.primary());

#ifdef ENABLE_CUDA
    auto cpu_filtered = cpu_buffer_pool_->Acquire();
    // std::unique_ptr<float[]>(new float[tensor_buf.num_elements]);
    cudaMemcpy(cpu_filtered.get(), tensor_buf.data, tensor_buf.byte_size(), cudaMemcpyDeviceToHost);
    tensor_buf.data = cpu_filtered.get();
#endif

    // ====== 4. 后处理 ======
    int64_t plane_size = tensor_buf.plane_size();
    auto& out_shape = tensor_buf.shape;
    size_t batch = tensor_buf.letterbox_params.size();

    // ⭐ 优化：将 clamp lambda 移出循环，避免每次迭代重新构造闭包对象
    auto clamp = [](float v) -> uint8_t {
        return static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, v)) * 255.0f + 0.5f);
    };

    int out_h = static_cast<int>(out_shape[2]);
    int out_w = static_cast<int>(out_shape[3]);

    std::vector<cv::Mat> out_imgs(batch);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < batch; ++i) {
        float* pdata = tensor_buf.data + i * plane_size;

        const int channel_size = out_h * out_w;
        const float* r_plane = pdata;
        const float* g_plane = pdata + channel_size;
        const float* b_plane = pdata + 2 * channel_size;

        const int& ori_h = tensor_buf.letterbox_params[i].orig_h;
        const int& ori_w = tensor_buf.letterbox_params[i].orig_w;
        const int& pad_top = tensor_buf.letterbox_params[i].pad_top;
        const int& pad_left = tensor_buf.letterbox_params[i].pad_left;

        cv::Mat result(ori_h, ori_w, CV_8UC3);

        for (int h = 0; h < ori_h; ++h) {
            auto* row_ptr = result.ptr<cv::Vec3b>(h);
            const int src_row_base = (h + pad_top) * out_w + pad_left;
        
            for (int w = 0; w < ori_w; ++w) {
                const int src_idx = src_row_base + w;
                // 连续写入 BGR，预计算通道基址避免重复乘法
                row_ptr[w] = cv::Vec3b(
                    clamp(b_plane[src_idx]),
                    clamp(g_plane[src_idx]),
                    clamp(r_plane[src_idx])
                );
            }
        }

        out_imgs[i] = std::move(result);
    }

    return out_imgs;
}


/// @brief 图像预处理
/// @param imgs 图像集合
// TensorBuffer NAFNet::preprocess(const std::vector<cv::Mat>& imgs) {
//     // ====== 1. 预处理 ======
//     // BGR → RGB, HWC → CHW, uint8 [0,255] → float32 [0,1]
//     int batch = imgs.size();
//     std::vector<cv::Mat> rgb_imgs(batch);

//     TensorBuffer& buf = backend_->tensorBuffer();
//     const cv::Size size(buf.shape[3], buf.shape[2]);
//     size_t plane_size = buf.plane_size();

// #pragma omp parallel for schedule(dynamic)
//     for (int i = 0; i < batch; ++i) {
//         cv::cvtColor(imgs[i], rgb_imgs[i], cv::COLOR_BGR2RGB);
//         buf.letterbox_params[i] = pad_to_size(rgb_imgs[i], rgb_imgs[i], size);

//         convert_and_normalize(rgb_imgs[i], buf.data + i * plane_size, size, norm_scale_, bgr2rgb_);
//     }

//     // 1. 同步搬运数据
//     // cudaMemcpy(backend_->data(), buf.data, buf.byte_size(), cudaMemcpyHostToDevice);
//     // 2. H2D 全部走异步多流（唯一的数据搬运路径）
// #ifdef ENABLE_CUDA
//     if (backend_->isGPUActivate()) {
//         for (int i = 0; i < batch; ++i) {
//             cudaMemcpyAsync(
//                 backend_->data() + i * plane_size,
//                 buf.data + i * plane_size,
//                 plane_size * sizeof(float),
//                 cudaMemcpyHostToDevice,
//                 streams_[i].get()
//             );
//         }
//     }
// #endif

//     return buf;
// }

/// @brief 对模型推理结果进行后处理
/// @param batch 实际的批量大小
/// @return 
// std::vector<cv::Mat> NAFNet::postprocess(const TensorBuffer& tensor_buf, size_t batch) {
//     // ====== 4. 后处理 ======

//     int64_t plane_size = tensor_buf.plane_size();
//     auto& out_shape = tensor_buf.shape;

//     // ⭐ 优化：将 clamp lambda 移出循环，避免每次迭代重新构造闭包对象
//     auto clamp = [](float v) -> uint8_t {
//         return static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, v)) * 255.0f + 0.5f);
//     };

//     int out_h = static_cast<int>(out_shape[2]);
//     int out_w = static_cast<int>(out_shape[3]);

//     std::vector<cv::Mat> out_imgs(batch);

// #pragma omp parallel for schedule(dynamic)
//     for (int i = 0; i < batch; ++i) {
//         float* pdata = tensor_buf.data + i * plane_size;

//         const int channel_size = out_h * out_w;
//         const float* r_plane = pdata;
//         const float* g_plane = pdata + channel_size;
//         const float* b_plane = pdata + 2 * channel_size;

//         const int& ori_h = tensor_buf.letterbox_params[i].orig_h;
//         const int& ori_w = tensor_buf.letterbox_params[i].orig_w;
//         const int& pad_top = tensor_buf.letterbox_params[i].pad_top;
//         const int& pad_left = tensor_buf.letterbox_params[i].pad_left;

//         cv::Mat result(ori_h, ori_w, CV_8UC3);

//         for (int h = 0; h < ori_h; ++h) {
//             auto* row_ptr = result.ptr<cv::Vec3b>(h);
//             const int src_row_base = (h + pad_top) * out_w + pad_left;
        
//             for (int w = 0; w < ori_w; ++w) {
//                 const int src_idx = src_row_base + w;
//                 // 连续写入 BGR，预计算通道基址避免重复乘法
//                 row_ptr[w] = cv::Vec3b(
//                     clamp(b_plane[src_idx]),
//                     clamp(g_plane[src_idx]),
//                     clamp(r_plane[src_idx])
//                 );
//             }
//         }

//         out_imgs[i] = std::move(result);
//     }

//     return out_imgs;
// }