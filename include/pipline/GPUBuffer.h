#pragma once

#include <onnxruntime_cxx_api.h>

#include "device/cuda_utils.h"
#include "BufferState.h"
#include "logger/logger.h"

struct GPUBuffer {
    // 配对管理
    CudaMallocGuard g_input;
    CudaMallocGuard g_output;

    // GPU Tensor
    Ort::Value input_tensor{nullptr};
    Ort::Value output_tensor{nullptr};

    GPUBuffer(size_t input_bytes_size, size_t output_bytes_size) : 
        g_input(input_bytes_size),
        g_output(output_bytes_size) {
    }
    
    std::atomic<BufferState> state{BufferState::IDLE};
    
    // ✅ 整个buffer作为一个整体管理
    bool isAvailable() {
        return state.load() == BufferState::IDLE;
    }
};