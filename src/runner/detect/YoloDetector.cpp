/**
 * @FilePath     : /onnxruntime-infer/src/runner/detect/YoloDetector.cpp
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-07-01 15:41:33
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-03 10:44:13
**/

#include <iostream>

#include "preprocess/utils.h"
#include "runner/detect/YoloDetector.h"
#include "backend/OrtSessionWrapper.h"

YoloDetector::YoloDetector(const std::string& config_path) : 
    Detector([&config_path]() {
        // 在 lambda 中独立加载，不依赖任何成员变量
        auto cfg = YAML::LoadFile(config_path);
        return std::make_unique<OrtSessionWrapper>(
            cfg["model"].as<YAML::Node>());
    }()),
    config_(YAML::LoadFile(config_path)) {  // 成员变量后构造，安全 

    warm_up_ = config_["model"]["warm_up"].as<size_t>(0);
        
    auto_aspect_ratio_  = config_["preprocess"]["auto_aspect_ratio"].as<bool>(true);
    std::vector<int> pc = config_["preprocess"]["pad_color"].as<std::vector<int>>(std::vector<int>{114, 114, 114});
    pad_color_          = cv::Scalar(pc[0], pc[1], pc[2]);
    norm_scale_         = 1.0f / 255.0f;
    bgr2rgb_            = config_["preprocess"]["bgr_to_rgb"].as<bool>(true);
    labels_ = config_["postprocess"]["classes"].as<std::vector<std::string>>();

    conf_threshold_     = config_["postprocess"]["conf_threshold"].as<float>(0.25);
    nms_threshold_      = config_["postprocess"]["nms_threshold"].as<float>(0.45);
    max_detections_     = config_["postprocess"]["max_detections"].as<size_t>(300);

    if (warm_up_) backend_->warm_up(warm_up_);
}

// 默认使用Ort后端推理
TensorBuffer YoloDetector::preprocess(const cv::Mat& img) {
    cv::Mat resized;

    const auto& shapes = backend_->shapes();

    TensorBuffer buf = TensorBuffer::create({1, shapes[1], shapes[2], shapes[3]});

    cv::Size size(shapes[3], shapes[2]);
    // b, c, h, w
    buf.letterbox_params = letterbox_resize(img, resized, size, auto_aspect_ratio_, pad_color_);

    convert_and_normalize(resized, buf.data, size, norm_scale_, bgr2rgb_);

    return buf;
}

const std::string& YoloDetector::class_label(size_t id) const {
    return labels_[id];
}

std::vector<Detection> YoloDetector::postprocess(const TensorBuffer& tensor_buf) 
{
    std::vector<Detection> results;
    
    // ========== 1. 基础校验 ==========
    if (!tensor_buf.valid() || tensor_buf.shape.size() != 3) {
        return results;
    }
    
    // YOLOv8 输出形状: [1, numAttributes(4+num_classes), numPredictions]
    const int num_attributes = static_cast<int>(tensor_buf.shape[1]); // 例如 84
    const int num_predictions = static_cast<int>(tensor_buf.shape[2]); // 例如 8400
    const int num_classes = num_attributes - 4;
    
    const float* pdata = tensor_buf.data;
    
    // ========== 2. 解码候选框 (替代原始的 transpose + row遍历) ==========
    // 💡 优化：直接在原始 [84, 8400] 布局上按列访问，避免 cv::transpose 的内存拷贝开销
    std::vector<cv::Rect2i> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    boxes.reserve(max_detections_);
    confidences.reserve(max_detections_);
    class_ids.reserve(max_detections_);
    
    for (int i = 0; i < num_predictions; ++i) {
        // 在 [84, 8400] 布局中，第 i 个预测框的数据起始位置
        // cx=pdata[i], cy=pdata[N+i], w=pdata[2N+i], h=pdata[3N+i]
        const float* box_ptr = pdata + i;
        const float* cls_ptr = pdata + 4 * num_predictions + i;
        
        // Argmax 寻找最佳类别 (替代 minMaxLoc)
        float best_score = 0.f;
        int best_cls = 0;
        for (int c = 0; c < num_classes; ++c) {
            float s = cls_ptr[c * num_predictions];
            if (s > best_score) {
                best_score = s;
                best_cls = c;
            }
        }
        
        if (best_score < conf_threshold_) continue;
        
        // 提取 cx, cy, w, h
        float cx = box_ptr[0];
        float cy = box_ptr[num_predictions];
        float ow = box_ptr[2 * num_predictions];
        float oh = box_ptr[3 * num_predictions];
        
        // ⭐ 坐标还原到原图 (替代原始的 x_factor 乘法)
        float x1 = std::clamp((cx - 0.5f * ow - tensor_buf.letterbox_params.pad_left) / tensor_buf.letterbox_params.scale, 0.f, (float)tensor_buf.letterbox_params.orig_w);
        float y1 = std::clamp((cy - 0.5f * oh - tensor_buf.letterbox_params.pad_top)  / tensor_buf.letterbox_params.scale, 0.f, (float)tensor_buf.letterbox_params.orig_h);
        float x2 = std::clamp((cx + 0.5f * ow - tensor_buf.letterbox_params.pad_left) / tensor_buf.letterbox_params.scale, 0.f, (float)tensor_buf.letterbox_params.orig_w);
        float y2 = std::clamp((cy + 0.5f * oh - tensor_buf.letterbox_params.pad_top)  / tensor_buf.letterbox_params.scale, 0.f, (float)tensor_buf.letterbox_params.orig_h);
        
        // 过滤无效框
        if (x2 <= x1 || y2 <= y1) {
            continue;
        } 
        
        // 转为 cv::Rect 用于 NMS (OpenCV NMSBoxes 要求 Rect)
        boxes.emplace_back(
            static_cast<int>(x1), static_cast<int>(y1),
            static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)
        );
        confidences.push_back(best_score);
        class_ids.push_back(best_cls);
    }
    
    // ========== 3. NMS ==========
    std::vector<int> indexes;
    if (!boxes.empty()) {
        cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indexes);
    }
    
    // ========== 4. 组装最终结果 ==========
    results.reserve(indexes.size());
    for (int idx : indexes) {
        results.push_back({
            boxes[idx],
            confidences[idx],
            class_ids[idx]
        });
    }
    
    return results;
}


std::vector<Detection> YoloDetector::detect(const cv::Mat& img) {
    TensorBuffer tensor = preprocess(img);
    ModelOutput infer_out = infer(tensor);
    infer_out.primary().letterbox_params = tensor.letterbox_params;
    
    return postprocess(infer_out.primary());
}