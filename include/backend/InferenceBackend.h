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

#include "TensorBuffer.h"

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
public:
    InferenceBackend() = default;
    virtual ~InferenceBackend() = default;
    virtual const std::vector<int64_t>& shapes() const = 0;
    virtual ModelOutput run(const TensorBuffer& buffer) = 0;

    void warm_up(size_t cnt);
};