// gpu_preprocess.h
#pragma once

#include <cstdint>
#include "cuda_utils.h"

#ifdef ENABLE_CUDA

extern "C" void gpu_yolo_preprocess(
    const uint8_t* d_src_bgr, float* d_dst_nchw,
    int src_w, int src_h, int src_step,
    int dst_w, int dst_h,
    float scale, int pad_left, int pad_top,
    float norm_scale, uint8_t pad_value,
    cudaStream_t stream);

#endif