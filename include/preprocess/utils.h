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
LetterboxParams letterbox_resize(
    const cv::Mat& img, 
    cv::Mat& resized,
    cv::Size target_size, 
    bool auto_aspect_ratio = false,
    cv::Scalar pad_color = cv::Scalar(114, 114, 114));


/// @brief HWC(u8,BGR) → NCHW(f32,RGB) + Normalize，单次遍历零临时分配
/// @param src       letterbox 后的图像 (HWC, CV_8UC3)
/// @param dst       目标缓冲区指针 (已分配, 大小 = C*H*W*sizeof(float))
/// @param height    图像高度
/// @param width     图像宽度
/// @param norm_scale 归一化系数 (如 1.0f/255.0f)
/// @param bgr2rgb   是否需要通道交换
void convert_and_normalize(
    const cv::Mat& src,
    float* dst,
    cv::Size size,
    float norm_scale,
    bool bgr2rgb);


/// @brief 将图像填充到指定尺寸（居中填充）
/// @param img 输入图像
/// @param padded_img 输出图像
/// @param size 目标尺寸 (width, height)
/// @param pad_color 填充的颜色
/// @return 原图信息
LetterboxParams pad_to_size(const cv::Mat& img, cv::Mat& padded_img, const cv::Size size, cv::Scalar pad_color = cv::Scalar(114, 114, 114));