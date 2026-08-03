#pragma once

#include <onnxruntime_cxx_api.h>

#include "device/cuda_utils.h"
#include "BufferState.h"
#include "logger/logger.h"

class TaskContext;


struct GPUBuffer {
    // 配对管理
    CudaMallocGuard<float> g_input;
    CudaMallocGuard<float> g_output;

    // GPU Tensor
    Ort::Value input_tensor{nullptr};
    Ort::Value output_tensor{nullptr};

    /// @brief 还需要自己手动初始化tensor
    /// @param input_elements_size 
    /// @param output_elements_size 
    GPUBuffer(size_t input_elements_size, 
        size_t output_elements_size) : 
        g_input(input_elements_size),
        g_output(output_elements_size) {
    }

    /// @brief 通过上下文自动初始化对象，并初始化Tensor
    /// @param context 
    GPUBuffer(const TaskContext& context);
};