/* 用于定义一些不同设备下可能冲突的代码或头文件声明 */
#pragma once

#include <memory>

#ifdef ENABLE_CUDA

#include <cuda_runtime.h>
#include "cuda_stream_deleter.h"
#include "InferTensorBufferPool.h"
#include "gpu_preprocess.h"
#include "gpu_filter.h"


#endif