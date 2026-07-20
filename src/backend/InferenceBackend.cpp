/**
 * @FilePath     : /onnxruntime-infer/src/backend/InferenceBackend.cpp
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-07-02 11:14:23
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-02 11:24:39
**/
#include "backend/InferenceBackend.h"

#include "logger/logger.h"

void InferenceBackend::init() {
    if (is_init_) return;
    tensorBuffer_ = TensorBuffer::create(shapes());

#ifdef ENABLE_CUDA
    pool_ = std::make_unique<InferTensorBufferPool>(buffer_size_, tensorBuffer_.byte_size());
#endif

    is_init_ = true;
}

/// @brief 将CPU的数据移动到目标设备中
void InferenceBackend::memCpyHostToDevice(const TensorBuffer& tenbuf) {
    // 将预处理参数传递到全局TensorBuffer中，便于后续后处理通过参数还原
    tensorBuffer().letterbox_params = tenbuf.letterbox_params;
    size_t batch = tenbuf.letterbox_params.size();
    
    if (data() != tenbuf.data) {
        if (isGPUActivate()) {
#ifdef ENABLE_CUDA
            // 初始化cuda流
            if (!is_custream_init_) {
                is_custream_init_ = true;
                int64_t batch = shapes()[0];
                streams_.reserve(batch);
                for (int i = 0; i < batch; ++i) {
                    cudaStream_t stream{};
                    cudaStreamCreate(&stream);
                    streams_.emplace_back(stream);
                }
            }
            
            // 将数据从CPU异步移动到GPU
            int64_t plane_size = tenbuf.plane_size();
            for (int i = 0; i < batch; ++i) {
                cudaMemcpyAsync(
                    data() + i * plane_size,
                    tenbuf.data + i * plane_size,
                    plane_size * sizeof(float),
                    cudaMemcpyHostToDevice,
                    streams_[i].get()
                );
            }
#endif
        } 
        else {
            std::memcpy(data(), tenbuf.data, tenbuf.byte_size());
        }
    }
}


ModelOutput InferenceBackend::run(const TensorBuffer& tenbuf) {
    memCpyHostToDevice(tenbuf);
    return infer();
}


void InferenceBackend::warm_up(size_t cnt) {
    ModelOutput model_output;
#ifdef ENABLE_CUDA
    for (int i = 0; i < cnt; ++i) {
        float* data = pool_->Acquire<float>();
        auto tenbuf = TensorBuffer::wrap(data, shapes());
        if (i == 0) {
            model_output = infer(tenbuf);
        } else {
            infer(tenbuf);
        }
    }

    
#else
    auto tenbuf = GetTensorBuffer();
    for (int i = 0; i < cnt; ++i) {
        if (i == 0) {
            model_output = infer(tenbuf);
        } else {
            infer(tensorBuffer());
        }
    }
#endif

    output_shapes_ = model_output.primary().shape;
    // std::cout << "热身已完成" << std::endl;
    LOG_INFO("Backend warm up successfully");
}