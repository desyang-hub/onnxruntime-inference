#pragma once

void lunchPostprocessFilter(
    const float* d_input,
    float* d_output,
    int* d_count, // gpu计数指针
    size_t plane_size,
    int batch_size,
    int num_predictions,
    int num_classes,
    float conf_thresh,
    int max_det,
    cudaStream_t stream=0
);