#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ✅ 支持 Multi-Batch 的 Filter & Compact Kernel
__global__ void filter_and_compact_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int* __restrict__ count,              // ← 改为 global atomic
    const int num_predictions,
    const int num_classes,
    const float conf_thresh,
    const int max_det)
{
    const int bid = blockIdx.x;
    if (bid >= gridDim.x) return;

    const int stride = num_predictions;
    const float* batch_input = input + bid * (4 + num_classes) * stride;
    float* batch_output = output + bid * max_det * 6;

    // ✅ 初始化全局计数器（仅每个 batch 的第一个线程）
    if (blockIdx.y == 0 && threadIdx.x == 0) {
        count[bid] = 0;
    }
    // ⚠️ 必须确保所有 block 看到 count 已被重置
    __threadfence_system();
    __syncthreads();  // 注意：这只同步本 block，但配合 threadfence 足够

    // grid-stride loop
    for (int i = threadIdx.x + blockIdx.y * blockDim.x;
         i < num_predictions;
         i += blockDim.x * gridDim.y)
    {
        float max_score = -1.0f;
        int max_cls = -1;
        const float* cls_ptr = batch_input + 4 * stride + i;
        #pragma unroll
        for (int c = 0; c < num_classes; ++c) {
            float s = cls_ptr[c * stride];
            if (s > max_score) {
                max_score = s;
                max_cls = c;
            }
        }

        if (max_score >= conf_thresh) {
            // ✅ 全局原子递增，所有 block.y 共享同一个计数器
            int idx = atomicAdd(&count[bid], 1);
            if (idx < max_det) {
                float* out = batch_output + idx * 6;
                out[0] = batch_input[0 * stride + i];
                out[1] = batch_input[1 * stride + i];
                out[2] = batch_input[2 * stride + i];
                out[3] = batch_input[3 * stride + i];
                out[4] = max_score;
                out[5] = static_cast<float>(max_cls);
            }
        }
    }
    // ✅ 不再需要最后写 count[bid]，它已经在 atomicAdd 中实时更新
}

// ✅ Host 端启动函数
void launch_filter_and_compact(
    const float* d_input,
    float* d_output,
    int* d_count,
    int batch_size,
    int num_predictions,
    int num_classes,
    float conf_thresh,
    int max_det,
    cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid(batch_size, (num_predictions + block.x - 1) / block.x);

    filter_and_compact_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, d_count,
        num_predictions, num_classes, conf_thresh, max_det);
}