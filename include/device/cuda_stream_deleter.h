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