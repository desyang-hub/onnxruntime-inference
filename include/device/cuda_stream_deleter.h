#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include "exceptions/utils.h"

struct cuda_stream_deleter {
    void operator()(cudaStream_t stream) {
        if (stream) {
            cudaStreamSynchronize(stream); 
            cudaStreamDestroy(stream);
        }
    }
};

using CudaStreamPtr = std::unique_ptr<CUstream_st, cuda_stream_deleter>;

// 使用RAII将cudaStream_t进行封装
class CudaStream {
private:
    std::shared_ptr<CUstream_st> cudaStream_;
public:
    CudaStream() {
        cudaStream_t stream{};
        cudaError_t err = cudaStreamCreate(&stream);

        if (err != cudaSuccess) {
            throw std::runtime_error(MESSAGE_WITH_LOC("cudaStreamCreate error"));
        }

        cudaStream_.reset(stream, [](cudaStream_t p) {
            if (p) {
                cudaStreamDestroy(p);
            }
        });
    }

    cudaStream_t get() const {
        return cudaStream_.get();
    }
};