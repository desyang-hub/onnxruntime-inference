/* 用于定义一些不同设备下可能冲突的代码或头文件声明 */
#pragma once

#include <memory>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>


struct cuda_stream_deleter {
    void operator()(cudaStream_t stream) {
        if (stream) {
            cudaStreamSynchronize(stream); 
            cudaStreamDestroy(stream);
        }
    }
};
using CudaStreamPtr = std::unique_ptr<CUstream_st, cuda_stream_deleter>;
#endif