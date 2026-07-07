/**
 * @FilePath     : /onnxruntime-infer/include/TensorBuffer.h
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-07-01 17:09:32
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-01 21:47:59
**/
#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include <cassert>
#include <numeric>

#include "preprocess/utils.h"

/// @brief 硬件无关的张量缓冲区，作为 Preprocessor 与 InferenceBackend 之间的标准接口
struct TensorBuffer {
    // ==================== 1. 核心数据 ====================
    std::shared_ptr<float[]> storage;  // ⭐ 用 shared_ptr 支持零拷贝传递
    float* data = nullptr;                        // 指向实际数据的裸指针（方便传给 C API）
    
    // ==================== 2. 形状与布局元数据 ====================
    std::vector<int64_t> shape;                   // NCHW: [1, 3, 640, 640]
    size_t num_elements = 0;                      // shape 各维度乘积
    
    // ==================== 3. Letterbox 后处理还原参数 ====================
    // float scale_factor = 1.0f;                    // letterbox 缩放比
    // int pad_left = 0;                             // 左填充像素
    // int pad_top = 0;                              // 上填充像素
    std::vector<LetterboxParams> letterbox_params;
    
    // ==================== 构造与工厂方法 ====================
    TensorBuffer() = default;
    
    /// @brief 分配指定形状的缓冲区
    static TensorBuffer create(const std::vector<int64_t>& shape_) {
        TensorBuffer buf;
        buf.shape = shape_;
        buf.num_elements = std::accumulate(
            shape_.begin(), shape_.end(), 
            size_t{1}, std::multiplies<size_t>()
        );
        buf.letterbox_params.resize(shape_[0]);
        // buf.storage = std::make_shared<std::vector<float>>(buf.num_elements);
        buf.storage = std::shared_ptr<float[]>(new float[buf.num_elements]);
        buf.data = buf.storage.get();
        return buf;
    }
    
    /// @brief 从外部内存零拷贝包装（如 TensorRT/RKNN 已分配的 buffer）
    static TensorBuffer wrap(float* external_data, 
                             const std::vector<int64_t>& shape_,
                             std::shared_ptr<void> lifetime_guard = nullptr) 
    {
        TensorBuffer buf;
        buf.shape = shape_;
        buf.num_elements = std::accumulate(
            shape_.begin(), shape_.end(),
            size_t{1}, std::multiplies<size_t>()
        );
        buf.data = external_data;
        buf.letterbox_params.resize(shape_[0]);
        // ⭐ lifetime_guard 持有外部内存的所有权，防止悬空指针
        // 如果不需要管理生命周期，传 nullptr 即可（调用方自行保证）
        if (lifetime_guard) {
            buf.storage = std::reinterpret_pointer_cast<float[]>(lifetime_guard);
        }
        return buf;
    }
    
    // ==================== 工具方法 ====================
    bool valid() const { return data != nullptr && num_elements > 0; }

    size_t ele_size() const {
        return num_elements;
    }
    
    size_t byte_size() const { return num_elements * sizeof(float); }

    size_t plane_size() const {
        return num_elements / shape[0];
    }
};