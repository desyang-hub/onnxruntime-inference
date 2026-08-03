#include "kernels/yolo/preprocess.cuh"

#include <stdint.h>
#include <stdexcept>


// 方案A：__device__ 辅助函数（推荐，编译器保证内联）
__device__ __forceinline__ float3 read_bgr(
    const uint8_t* src, size_t step, int px, int py) 
{
    size_t off = static_cast<size_t>(py) * step + px * 3;
    return make_float3(
        static_cast<float>(src[off + 2]),
        static_cast<float>(src[off + 1]),
        static_cast<float>(src[off + 0])
    );
}

/// @brief 用于处理yolo预处理的 kernel
/// @return 
__global__ void yolo_preprocess(
    const uint8_t* __restrict__  src_bgr,
    float* __restrict__  dst_nchw,
    const int src_h, const int src_w, const int src_step,
    const int dst_h, const int dst_w, const size_t dst_hw,
    const float scale, const int pad_left, const int pad_top,
    const int right_bt_x, const int right_bt_y,
    const float norm_scale, const float pad_normal_value
) {
    // 选定范围
    int x = threadIdx.x + blockDim.x * blockIdx.x; // block 内线程id + block大小 * block id
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= dst_w || y >= dst_h) return; // 超出范围区域不计算

    const size_t dst_idx = static_cast<size_t>(y) * dst_w + x;

    // 最终需要计算的结果r,g,g
    float r, g, b;

    // 判断是否在图像区域还是pad区域
    if (x >= pad_left && x < right_bt_x && y >= pad_top && y < right_bt_y) {
        // ⭐ 反向映射到源图像坐标（浮点）
        float src_xf = (static_cast<float>(x - pad_left) + 0.5f) / scale - 0.5f;
        float src_yf = (static_cast<float>(y - pad_top) + 0.5f) / scale - 0.5f;

        // 双线性插值
        int x0 = __float2int_rd(src_xf);
        int y0 = __float2int_rd(src_yf);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float fx = src_xf - x0;
        float fy = src_yf - y0;

        // clamp to source bounds
        x0 = max(0, min(x0, src_w - 1));
        y0 = max(0, min(y0, src_h - 1));
        x1 = max(0, min(x1, src_w - 1));
        y1 = max(0, min(y1, src_h - 1));

        float3 p00 = read_bgr(src_bgr, src_step, x0, y0);
        float3 p10 = read_bgr(src_bgr, src_step, x1, y0);
        float3 p01 = read_bgr(src_bgr, src_step, x0, y1);
        float3 p11 = read_bgr(src_bgr, src_step, x1, y1);

        // 双线性插值 + BGR→RGB 同时在寄存器内完成
        float inv_fx = 1.0f - fx;
        float inv_fy = 1.0f - fy;
        r = (p00.x * inv_fx + p10.x * fx) * inv_fy + (p01.x * inv_fx + p11.x * fx) * fy;
        g = (p00.y * inv_fx + p10.y * fx) * inv_fy + (p01.y * inv_fx + p11.y * fx) * fy;
        b = (p00.z * inv_fx + p10.z * fx) * inv_fy + (p01.z * inv_fx + p11.z * fx) * fy;

        // ⭐ Normalize + 写入 CHW 布局
        dst_nchw[dst_idx] = r * norm_scale;
        dst_nchw[dst_idx + dst_hw] = g * norm_scale;
        dst_nchw[dst_idx + 2 * dst_hw] = b * norm_scale;
    }
    else {
        dst_nchw[dst_idx] = pad_normal_value;
        dst_nchw[dst_idx + dst_hw] = pad_normal_value;
        dst_nchw[dst_idx + 2 * dst_hw] = pad_normal_value;
    }
}


void lunchPreprocess(
    void* src_bgr,
    float* dest_nchw,
    int src_h, int src_w, int src_step,
    int new_h, int new_w,
    int dst_h, int dst_w,
    float scale, int pad_left, int pad_top,
    float norm_scale, int pad_value,
    cudaStream_t stream
) 
{
    dim3 block_dim(16, 16); // x, y col, row
    dim3 grid_dim((dst_w + block_dim.x - 1)  / block_dim.x, (dst_h + block_dim.y - 1) / block_dim.y);
    size_t dst_hw = static_cast<size_t>(dst_h) * dst_w;
    float pad_normal_value = norm_scale * pad_value;
    int right_bt_x = pad_left + new_w;
    int right_bt_y = pad_top + new_h;

    yolo_preprocess<<<grid_dim, block_dim, 0, stream>>>(
        static_cast<uint8_t*>(src_bgr),
        dest_nchw, 
        src_h, src_w, src_step, 
        dst_h, dst_w, dst_hw,
        scale, pad_left, pad_top, 
        right_bt_x, right_bt_y,
        norm_scale, pad_normal_value
    );
}

// __device__ __forceinline__ float3 read_bgr(
//     const uint8_t* src, size_t step, int px, int py) 
// {
//     size_t off = static_cast<size_t>(py) * step + px * 3;
//     return make_float3(
//         static_cast<float>(src[off + 2]),
//         static_cast<float>(src[off + 1]),
//         static_cast<float>(src[off + 0]));
// }

template <int BATCH_SIZE>
__global__ void yolo_preprocess_batch(
    const uint8_t* __restrict__ packed_src_bgr,
    const size_t* __restrict__ img_offsets,
    const ImageMeta* __restrict__ metas,
    float* __restrict__ dst_nchw,
    const int dst_h, const int dst_w, 
    const size_t dst_hw,
    const float norm_scale, const float pad_normal_value)
{
    // ⭐ 编译器知道 BATCH_SIZE，此检查在编译期求值
    const int n = blockIdx.z;
    if constexpr (BATCH_SIZE > 0) {
        // 当 gridDim.z == BATCH_SIZE 时，此分支被编译器完全消除
        if (n >= BATCH_SIZE) return;
    }

    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    // ⭐ metas[n] 的索引对编译器而言是"有界常量范围"
    //    有利于向量化和 cache line 预取
    const ImageMeta meta = metas[n];
    const uint8_t* src_bgr = packed_src_bgr + img_offsets[n];
    
    const size_t batch_offset = static_cast<size_t>(n) * 3ULL * dst_hw;
    const size_t dst_idx = static_cast<size_t>(y) * dst_w + x;

    float r, g, b;

    if (x >= meta.pad_left && x < meta.right_bt_x && 
        y >= meta.pad_top  && y < meta.right_bt_y) 
    {
        float src_xf = (static_cast<float>(x - meta.pad_left) + 0.5f) / meta.scale - 0.5f;
        float src_yf = (static_cast<float>(y - meta.pad_top)  + 0.5f) / meta.scale - 0.5f;

        int x0 = __float2int_rd(src_xf);
        int y0 = __float2int_rd(src_yf);
        int x1 = x0 + 1; int y1 = y0 + 1;
        float fx = src_xf - x0;
        float fy = src_yf - y0;

        x0 = max(0, min(x0, meta.src_w - 1));
        y0 = max(0, min(y0, meta.src_h - 1));
        x1 = max(0, min(x1, meta.src_w - 1));
        y1 = max(0, min(y1, meta.src_h - 1));

        float3 p00 = read_bgr(src_bgr, meta.src_step, x0, y0);
        float3 p10 = read_bgr(src_bgr, meta.src_step, x1, y0);
        float3 p01 = read_bgr(src_bgr, meta.src_step, x0, y1);
        float3 p11 = read_bgr(src_bgr, meta.src_step, x1, y1);

        float inv_fx = 1.0f - fx;
        float inv_fy = 1.0f - fy;
        r = (p00.x * inv_fx + p10.x * fx) * inv_fy + (p01.x * inv_fx + p11.x * fx) * fy;
        g = (p00.y * inv_fx + p10.y * fx) * inv_fy + (p01.y * inv_fx + p11.y * fx) * fy;
        b = (p00.z * inv_fx + p10.z * fx) * inv_fy + (p01.z * inv_fx + p11.z * fx) * fy;

        dst_nchw[batch_offset + dst_idx]            = r * norm_scale;
        dst_nchw[batch_offset + dst_idx + dst_hw]   = g * norm_scale;
        dst_nchw[batch_offset + dst_idx + 2*dst_hw] = b * norm_scale;
    } else {
        dst_nchw[batch_offset + dst_idx]            = pad_normal_value;
        dst_nchw[batch_offset + dst_idx + dst_hw]   = pad_normal_value;
        dst_nchw[batch_offset + dst_idx + 2*dst_hw] = pad_normal_value;
    }
}


/// @brief 
/// @tparam BATCH_SZE batch 上限
/// @param packed_src_bgr 
/// @param img_offsets 
/// @param metas 
/// @param batch_size 实际处理的batch大小
/// @param dst_nchw 
/// @param dst_h 
/// @param dst_w 
/// @param norm_scale 
/// @param pad_value 
/// @param stream 
template<int BATCH_SZE>
void launchBatchPreprocess(
    const uint8_t* packed_src_bgr,
    const size_t* img_offsets,
    const ImageMeta* metas,
    int batch_size,
    float* dst_nchw,
    int dst_h, int dst_w,
    float norm_scale, int8_t pad_value,
    cudaStream_t stream
)
{
    dim3 block_dim(16, 16);
    dim3 grid_dim(
        (dst_w  + block_dim.x - 1) / block_dim.x,
        (dst_h + block_dim.y - 1) / block_dim.y,
        batch_size   // ⭐ z = batch，与模板参数一致
    );
    float pad_normal_value = pad_value * norm_scale;
    size_t dst_hw = static_cast<size_t>(dst_h) * dst_w;
    
    yolo_preprocess_batch<BATCH_SZE>
    <<<grid_dim, block_dim, 0, stream>>>
    (packed_src_bgr, img_offsets, metas, dst_nchw, dst_h, dst_w, dst_hw, norm_scale, pad_normal_value);
}

BATCH_TEMPLATE_DEFINE(1);
BATCH_TEMPLATE_DEFINE(2);
BATCH_TEMPLATE_DEFINE(4);
BATCH_TEMPLATE_DEFINE(8);
BATCH_TEMPLATE_DEFINE(16);