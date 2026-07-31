#pragma once

#include <cuda_runtime.h>
#include "macor.h"
#include "exceptions/utils.h"

class CudaMallocGuard {
private:
    void* gpu_ptr_;
public:
    explicit CudaMallocGuard(size_t bytes_size) : gpu_ptr_{} {
        cuMalloc(bytes_size);
    }

    CudaMallocGuard() : gpu_ptr_{} {
    }

    void cuMalloc(size_t bytes_size) {
        if (gpu_ptr_) throw std::runtime_error(MESSAGE_WITH_LOC("Repeat cudaMalloc"));
        CUDA_CHECK(cudaMalloc(&gpu_ptr_, bytes_size));
    }

    ~CudaMallocGuard() {
        CUDA_CHECK(cudaFree(gpu_ptr_));
    }

    template<class T = float>
    T* get() {
        return reinterpret_cast<T*>(gpu_ptr_);
    }
};