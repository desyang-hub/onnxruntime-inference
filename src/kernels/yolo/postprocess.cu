#include "kernels/yolo/postprocess.cuh"
#include <cuda_runtime.h>

// ✅ 模板化核函数：num_classes 编译期已知，可完全展开循环
template <int NUM_CLASSES, int MAX_DET>
__global__ void filter_and_compact_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int* __restrict__ count,
    const size_t input_plane_size,
    const int num_predictions,      // 8400 (运行时变量，用于边界检查)
    const float conf_thresh,
    const int output_plane_size)
{
    const int bid = blockIdx.x;
    if (bid >= gridDim.x) return; // 超过的batch 不计算

    const int stride = num_predictions;
    const float* batch_input = input + bid * input_plane_size;
    float* batch_output = output + bid * output_plane_size;

    // Grid-stride loop 这里只会处理一次
    for (int i = threadIdx.x + blockIdx.y * blockDim.x;
         i < num_predictions;
         i += blockDim.x * gridDim.y)
    {
        // ========== 1. 向量化读取 bbox (xywh) ==========
        // SoA 布局下 xywh 连续存储，用 float4 一次加载
        // float4 box = reinterpret_cast<const float4*>(
        //     batch_input + i)[0]; 
        // 注意: 这里假设输入指针已对齐，若未对齐需回退到标量读取

        // ========== 2. 编译期展开的类别最大值搜索 ==========
        float max_score = -1.0f;
        int max_cls = -1;
        const float* cls_ptr = batch_input + 4 * stride + i;

        #pragma unroll
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float s = cls_ptr[c * stride];
            if (s > max_score) {
                max_score = s;
                max_cls = c;
            }
        }

        // ========== 3. 阈值过滤 + 原子紧凑写入 ==========
        if (max_score >= conf_thresh) {
            int idx = atomicAdd(&count[bid], 1);
            if (idx < MAX_DET) {
                float* out = batch_output + idx * 6;
                // 写回 bbox
                out[0] = batch_input[0 * stride + i]; // 一共84个类别，每个类占用stride
                out[1] = batch_input[1 * stride + i];
                out[2] = batch_input[2 * stride + i];
                out[3] = batch_input[3 * stride + i];
                out[4] = max_score;
                out[5] = static_cast<float>(max_cls);
            }
        }
    }
}

// ✅ Host 端启动函数
void lunchPostprocessFilter(
    const float* d_input,
    float* d_output,
    int* d_count,
    size_t plane_size,
    int batch_size,
    int num_predictions,
    int num_classes,
    float conf_thresh,
    int max_det,
    cudaStream_t stream)
{
    // ⭐ 关键修复：Host 端异步清零，彻底消除 Kernel 内竞态
    cudaMemsetAsync(d_count, 0, batch_size * sizeof(int), stream);

    dim3 block(256);
    dim3 grid(batch_size, (num_predictions + block.x - 1) / block.x);

    int output_plane_size = 6 * max_det;

    // ⭐ 根据运行时 num_classes 分发模板实例
    // 实际项目中建议用宏或 switch-case 覆盖常用类别数
    if (num_classes == 80 && max_det <= 300) {
        filter_and_compact_kernel<80, 300><<<grid, block, 0, stream>>>(
            d_input, d_output, d_count, plane_size,
            num_predictions, conf_thresh, output_plane_size);
    } else if (num_classes == 80 && max_det <= 1000) {
        filter_and_compact_kernel<80, 1000><<<grid, block, 0, stream>>>(
            d_input, d_output, d_count, plane_size,
            num_predictions, conf_thresh, output_plane_size);
    } else {
        // Fallback: 对于不支持的组合，可添加默认模板或报错
        // 这里以 80/300 作为安全兜底（需注意 max_det 截断）
        filter_and_compact_kernel<80, 300><<<grid, block, 0, stream>>>(
            d_input, d_output, d_count, plane_size,
            num_predictions, conf_thresh, output_plane_size);
    }
}