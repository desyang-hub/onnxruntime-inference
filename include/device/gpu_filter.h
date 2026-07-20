// gpu_filter.h
#pragma once
#include <cuda_runtime.h>

void launch_filter_and_compact(
    const float* d_input,
    float* d_output,
    int* d_count,
    int batch_size,
    int num_predictions,
    int num_classes,
    float conf_thresh,
    int max_det,
    cudaStream_t stream
);