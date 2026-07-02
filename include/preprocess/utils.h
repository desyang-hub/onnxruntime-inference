/**
 * @FilePath     : /onnxruntime-infer/include/preprocess/utils.h
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-07-01 16:14:07
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-01 21:46:35
**/
#pragma once

#include <opencv2/opencv.hpp>

struct LetterboxParams {
    float scale;
    int pad_left;
    int pad_top;
    int orig_w;
    int orig_h;
};

/// @return scale_factor: 实际缩放比例（后处理坐标还原必需）
inline LetterboxParams letterbox_resize(
    const cv::Mat& img, 
    cv::Mat& resized,
    cv::Size target_size, 
    bool auto_aspect_ratio = false,
    cv::Scalar pad_color = cv::Scalar(114, 114, 114))
{
    // ⭐ 1. 计算缩放因子：取 min 保证图像完整放入目标尺寸
    float x_factor = static_cast<float>(target_size.width)  / img.cols;
    float y_factor = static_cast<float>(target_size.height) / img.rows;
    float scale = std::min(x_factor, y_factor);

    // ⭐ 2. auto_aspect_ratio: 将 scale 对齐到 stride（通常为32）
    //    避免后续 padding 出现奇数像素，提升 GPU 推理效率
    if (auto_aspect_ratio) {
        constexpr int stride = 32;
        int new_w = static_cast<int>(std::round(img.cols * scale));
        int new_h = static_cast<int>(std::round(img.rows * scale));
        new_w = new_w - (new_w % stride);  // 向下对齐
        new_h = new_h - (new_h % stride);
        // 重新计算精确 scale，避免累积误差
        scale = std::min(
            static_cast<float>(new_w) / img.cols,
            static_cast<float>(new_h) / img.rows
        );
    }

    // ⭐ 3. 等比缩放
    int new_w = static_cast<int>(std::round(img.cols * scale));
    int new_h = static_cast<int>(std::round(img.rows * scale));
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // ⭐ 4. 居中填充（Letterbox 的 "box" 部分）
    int pad_left   = (target_size.width  - new_w) / 2;
    int pad_top    = (target_size.height - new_h) / 2;
    int pad_right  = target_size.width  - new_w - pad_left;
    int pad_bottom = target_size.height - new_h - pad_top;

    if (pad_left > 0 || pad_top > 0 || pad_right > 0 || pad_bottom > 0) {
        cv::copyMakeBorder(resized, resized, 
                           pad_top, pad_bottom, 
                           pad_left, pad_right,
                           cv::BORDER_CONSTANT, pad_color);
    }

    return {scale, pad_left, pad_top, img.cols, img.rows};  // ⭐ 后处理用这个值还原 bbox 坐标
}


/// @brief HWC(u8,BGR) → NCHW(f32,RGB) + Normalize，单次遍历零临时分配
/// @param src       letterbox 后的图像 (HWC, CV_8UC3)
/// @param dst       目标缓冲区指针 (已分配, 大小 = C*H*W*sizeof(float))
/// @param height    图像高度
/// @param width     图像宽度
/// @param norm_scale 归一化系数 (如 1.0f/255.0f)
/// @param bgr2rgb   是否需要通道交换
inline void convert_and_normalize(
    const cv::Mat& src,
    float* dst,
    cv::Size size,
    float norm_scale,
    bool bgr2rgb)
{
    const int channels = 3;
    const size_t hw = static_cast<size_t>(size.height) * size.width;
    
    // NCHW 布局下各通道的起始偏移
    float* dst_r = dst;           // R 通道
    float* dst_g = dst + hw;      // G 通道  
    float* dst_b = dst + 2 * hw;  // B 通道
    
    // ⭐ 根据是否需要 swapRB 决定通道映射，避免运行时分支
    const uchar* src_data = src.data;
    
    if (bgr2rgb) {
        // BGR → RGB: src[0]=B→dst_b, src[1]=G→dst_g, src[2]=R→dst_r
        for (size_t i = 0; i < hw; ++i) {
            dst_r[i] = src_data[i * 3 + 2] * norm_scale;
            dst_g[i] = src_data[i * 3 + 1] * norm_scale;
            dst_b[i] = src_data[i * 3 + 0] * norm_scale;
        }
    } else {
        // BGR → BGR: 保持原序
        for (size_t i = 0; i < hw; ++i) {
            dst_r[i] = src_data[i * 3 + 0] * norm_scale;
            dst_g[i] = src_data[i * 3 + 1] * norm_scale;
            dst_b[i] = src_data[i * 3 + 2] * norm_scale;
        }
    }
}