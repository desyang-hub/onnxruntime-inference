#pragma once

#include <cstdint>
#include <cuda_runtime.h>

/// @brief yolo预处理启动函数
/// @param src_bgr img指针数据
/// @param dest_nchw gpu指针
/// @param src_h src img height
/// @param src_w src img width
/// @param src_step per row bytes size
/// @param new_h new img height
/// @param new_w new img widht
/// @param dst_h dest img heigh
/// @param dst_w dest img width
/// @param scale scale 
/// @param pad_left padding left size
/// @param pad_top padding top size
/// @param norm_scale normalize scale
/// @param pad_value padding value
/// @param stream cudaStream default `nullptr`
void lunchPreprocess(
    void* src_bgr,
    float* dest_nchw,
    int src_h, int src_w, int src_step,
    int new_h, int new_w,
    int dst_h, int dst_w,
    float scale, int pad_left, int pad_top,
    float norm_scale, int pad_value,
    cudaStream_t stream = 0
);