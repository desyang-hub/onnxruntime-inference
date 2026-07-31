#pragma once

#include <cuda_runtime.h>
#include "macor.h"

/// @brief 使用RAII封装cudaStream_t
class CudaStream
{
private:
    cudaStream_t stream_;
public:
    CudaStream() : stream_(nullptr)
    {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~CudaStream()
    {
        if (stream_) {
            CUDA_CHECK(cudaStreamDestroy(stream_));
        }
    }

    cudaStream_t get() {
        return stream_;
    }
};