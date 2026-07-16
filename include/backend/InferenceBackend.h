/**
 * @FilePath     : /onnxruntime-infer/include/backend/InferenceBackend.h
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-07-01 13:25:05
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-02 11:24:49
**/
#pragma once

#include <cstddef>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <onnxruntime_cxx_api.h>
#include <stdexcept>

#include "TensorBuffer.h"
#include "device/cuda_utils.h"

/// @brief 后端无关的模型输出容器
/// 无论底层是 ORT/TRT/RKNN，对外都呈现为命名张量集合
struct ModelOutput {
    /// 按名称索引的输出张量（支持多输出模型）
    std::unordered_map<std::string, TensorBuffer> tensors;
    
    /// 便捷访问：获取第一个/唯一输出
    const TensorBuffer& primary() const {
        assert(!tensors.empty());
        return tensors.begin()->second;
    }

    TensorBuffer& primary() {
        assert(!tensors.empty());
        return tensors.begin()->second;
    }
    
    /// 检查某个输出是否存在
    bool has(const std::string& name) const {
        return tensors.count(name) > 0;
    }
};

class InferenceBackend
{
private:
    TensorBuffer tensorBuffer_;
    bool is_init_ = false;

    bool is_gpu_active_ = false;
    int active_gpu_id_ = -1;

    void memCpyHostToDevice(const TensorBuffer&);

#ifdef ENABLE_CUDA
    std::vector<CudaStreamPtr> streams_;
    bool is_custream_init_ = false;
#endif

protected:
    void init();
public:

#ifdef ENABLE_CUDA
    InferTensorBufferPoolPtr pool_ = nullptr;
#endif

public:
    InferenceBackend() = default;
    virtual ~InferenceBackend() = default;
    virtual const std::vector<int64_t>& shapes() const = 0;
    ModelOutput run(const TensorBuffer&);

    virtual ModelOutput infer() = 0;

    virtual ModelOutput infer(const TensorBuffer&) = 0;

    // 获取张量
    TensorBuffer GetTensorBuffer() {
#ifdef ENABLE_CUDA
        assert(pool_.get());
        float* p = pool_->Acquire();
#else
        float* p = data();
#endif
        return TensorBuffer::wrap(p, shapes());
    }

#ifdef ENABLE_CUDA
    void release(float* data) {
        assert(pool_.get());
        pool_->Release(data);
    }
#endif    


    TensorBuffer& tensorBuffer() {
        if (!is_init_)
            init();
        
        return tensorBuffer_;
    }

    const TensorBuffer& tensorBuffer() const {
        if (!is_init_) throw std::runtime_error("backend not init.");
        return tensorBuffer_;
    }

    bool isGPUActivate() const {
        return is_gpu_active_;
    }

    void enableGPUActivate() {
        is_gpu_active_ = true;
    }

    int activateGPUId() const {
        return active_gpu_id_;
    }

    void setActivateGPUId(int gpu_id) {
        active_gpu_id_ = gpu_id;
#ifdef ENABLE_CUDA
        cudaSetDevice(gpu_id);
#endif
    }

    // warm_up 中包含了初始化，不论cnt是否为0，都必须调用
    void warm_up(size_t cnt);

    /// @brief 返回推理设备的缓存指针，比如GPU内存数据指针
    /// @return 
    virtual float* data() = 0;
};