// gpu_preprocess.cu
#include "device/gpu_preprocess.h"


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

// ⭐ 双线性插值 + Letterbox + BGR2RGB + Normalize + CHW Transpose
//    全部在一个 kernel 中完成，单次遍历，零中间缓冲
__global__ void yolo_preprocess_kernel(
    const uint8_t* __restrict__ src_bgr,  // HWC BGR (已拷贝到GPU)
    float* __restrict__ dst_nchw,         // CHW RGB normalized
    int src_w, int src_h, int src_step,
    int dst_w, int dst_h, size_t dest_hw,
    float scale, int pad_left, int pad_top,
    float norm_scale, float pad_norm_value)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x; // x是col
    const int y = blockIdx.y * blockDim.y + threadIdx.y; // y是row
    if (x >= dst_w || y >= dst_h) return; // 超出范围的区域不进行计算

    // 对于单个通道的相对位 row(y) * cols(dst_w) + col(x);
    const size_t dst_idx = static_cast<size_t>(y) * dst_w + x; 

    float r, g, b;

    // 判断当前像素是否在 letterbox 有效区域内

    // 将图像平移到左上角来判断是否在区间内
    int lx = x - pad_left;
    int ly = y - pad_top;
    int valid_w = dst_w - 2 * pad_left;  // = round(src_w * scale) aligned
    int valid_h = dst_h - 2 * pad_top;

    if (lx >= 0 && lx < valid_w && ly >= 0 && ly < valid_h) {
        // ⭐ 反向映射到源图像坐标（浮点）
        float src_xf = (static_cast<float>(lx) + 0.5f) / scale - 0.5f;
        float src_yf = (static_cast<float>(ly) + 0.5f) / scale - 0.5f;

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

        // 读取4个邻居像素（BGR）
        // auto read_bgr = [&](int px, int py) -> float3 {
        //     size_t off = static_cast<size_t>(py) * src_step + px * 3;
        //     return make_float3(
        //         static_cast<float>(src_bgr[off + 2]),  // R
        //         static_cast<float>(src_bgr[off + 1]),  // G
        //         static_cast<float>(src_bgr[off + 0])   // B
        //     );
        // };

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
        dst_nchw[0 * dest_hw + dst_idx] = r * norm_scale;
        dst_nchw[1 * dest_hw + dst_idx] = g * norm_scale;
        dst_nchw[2 * dest_hw + dst_idx] = b * norm_scale;
    } else {
        // padding 区域
        dst_nchw[0 * dest_hw + dst_idx] = pad_norm_value;
        dst_nchw[1 * dest_hw + dst_idx] = pad_norm_value;
        dst_nchw[2 * dest_hw + dst_idx] = pad_norm_value;
    }
}

// Host 端调用接口
extern "C" void gpu_yolo_preprocess(
    const uint8_t* d_src_bgr, float* d_dst_nchw,
    int src_w, int src_h, int src_step,
    int dst_w, int dst_h,
    float scale, int pad_left, int pad_top,
    float norm_scale, uint8_t pad_value,
    cudaStream_t stream)
{
    float pad_norm_value = norm_scale * pad_value;
    size_t hw = static_cast<size_t>(dst_h) * dst_w;
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    yolo_preprocess_kernel<<<grid, block, 0, stream>>>(
        d_src_bgr, d_dst_nchw,
        hw,
        src_w, src_h, src_step,
        dst_w, dst_h,
        scale, pad_left, pad_top,
        norm_scale, pad_norm_value);
}