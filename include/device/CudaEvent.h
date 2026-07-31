#pragma once

#include <cuda_runtime.h>
#include "macor.h"

/// @brief 使用RAII封装cudaEvent_t
class CudaEvent
{
private:
    cudaEvent_t event_;
public:
    CudaEvent(/* args */) : event_(nullptr)
    {
        CUDA_CHECK(cudaEventCreate(&event_));
    }
    
    ~CudaEvent()
    {
        if (event_) {
            CUDA_CHECK(cudaEventDestroy(event_));
        }
    }


    cudaEvent_t get() {
        return event_;
    }
};