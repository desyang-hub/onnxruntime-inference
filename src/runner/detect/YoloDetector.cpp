/**
 * @FilePath     : /onnxruntime-infer/src/runner/detect/YoloDetector.cpp
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-07-01 15:41:33
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-03 10:44:13
**/

#include <iostream>
#include <omp.h>
#include <assert.h>

#include "Timer.h"
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


#ifdef ENABLE_CUDA
    streams_.reserve(backend_->shapes()[0]);
    for (int i = 0; i < backend_->shapes()[0]; ++i) {
        cudaStream_t stream{};
        cudaStreamCreate(&stream);
        streams_.emplace_back(stream);
    }
#endif 
        
    auto_aspect_ratio_  = config_["preprocess"]["auto_aspect_ratio"].as<bool>(true);
    std::vector<int> pc = config_["preprocess"]["pad_color"].as<std::vector<int>>(std::vector<int>{114, 114, 114});
    pad_color_          = cv::Scalar(pc[0], pc[1], pc[2]);
    norm_scale_         = 1.0f / 255.0f;
    bgr2rgb_            = config_["preprocess"]["bgr_to_rgb"].as<bool>(true);
    labels_ = config_["postprocess"]["classes"].as<std::vector<std::string>>();

    conf_threshold_     = config_["postprocess"]["conf_threshold"].as<float>(0.25);
    nms_threshold_      = config_["postprocess"]["nms_threshold"].as<float>(0.45);
    max_detections_     = config_["postprocess"]["max_detections"].as<size_t>(300);
}

// 默认使用Ort后端推理
TensorBuffer YoloDetector::preprocess(const cv::Mat& img) {
    cv::Mat resized;

    const auto& shapes = backend_->shapes();

    TensorBuffer buf = TensorBuffer::create({1, shapes[1], shapes[2], shapes[3]});

    cv::Size size(shapes[3], shapes[2]);
    // b, c, h, w
    buf.letterbox_params[0] = letterbox_resize(img, resized, size, auto_aspect_ratio_, pad_color_);

    convert_and_normalize(resized, buf.data, size, norm_scale_, bgr2rgb_);

    return buf;
}

/*
思考？
我们的想法是零拷贝，自己创建tensor_buffer来存储需要推理的张量，减少拷贝所带来的性能损耗
不过，带来的问题确实，必须要先预定batch的大小，也就是一次需要处理的所有图像到齐，才开始构建张量，这样的话，每个线程完成预处理，并进行推理，推理时长= batch等待时长+最长的预处理时长(cpu疯转)+推理时长

如果不进行零拷贝，只要图像一来，就可以进行预处理，将预处理结果进行拷贝，当凑齐了batch时，batch时长++预处理时长（cpu压力小）+创建张量(拷贝结果)


*/

// 批量预处理后处理
void YoloDetector::preprocess(const std::vector<cv::Mat>& imgs) {
    const auto& shapes = backend_->shapes();
    // 这是实际的批量大小，也是图像的数量，如果少于实际推理batch后续进行padding（其实无需进行，因为缓存的张量大小就是按照预定的batch_size来初始化的，只需要在后处理阶段处理有效批量即可）
    const int64_t batch = static_cast<int64_t>(imgs.size());

    // 当前批次超过
    if (batch > shapes[0]) {
        throw std::runtime_error("inference size exceed batch size");
    }

    TensorBuffer& buf = backend_->tensorBuffer();

    cv::Size size(shapes[3], shapes[2]);
    const size_t plane_size = buf.plane_size();

    // 1. CPU 预处理保持 OMP 并行（纯 CPU 操作，安全高效）
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < batch; ++i) {
        cv::Mat resized;
        buf.letterbox_params[i] = 
            letterbox_resize(imgs[i], resized, size, auto_aspect_ratio_, pad_color_);
        convert_and_normalize(resized, buf.data + i * plane_size, size, norm_scale_, bgr2rgb_);
    }

    // 1. 同步搬运数据
    // cudaMemcpy(backend_->data(), buf.data, buf.byte_size(), cudaMemcpyHostToDevice);
    // 2. H2D 全部走异步多流（唯一的数据搬运路径）
#ifdef ENABLE_CUDA
    if (backend_->isGPUActivate()) {
        for (int i = 0; i < batch; ++i) {
            cudaMemcpyAsync(
                backend_->data() + i * plane_size,
                buf.data + i * plane_size,
                plane_size * sizeof(float),
                cudaMemcpyHostToDevice,
                streams_[i].get()
            );
        }
    }
#endif
}

/// @brief 根据 class_id 返回实际的类名称
/// @param id 类别id
/// @return 实际类别名称
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
        float x1 = std::clamp((cx - 0.5f * ow - tensor_buf.letterbox_params[0].pad_left) / tensor_buf.letterbox_params[0].scale, 0.f, (float)tensor_buf.letterbox_params[0].orig_w);
        float y1 = std::clamp((cy - 0.5f * oh - tensor_buf.letterbox_params[0].pad_top)  / tensor_buf.letterbox_params[0].scale, 0.f, (float)tensor_buf.letterbox_params[0].orig_h);
        float x2 = std::clamp((cx + 0.5f * ow - tensor_buf.letterbox_params[0].pad_left) / tensor_buf.letterbox_params[0].scale, 0.f, (float)tensor_buf.letterbox_params[0].orig_w);
        float y2 = std::clamp((cy + 0.5f * oh - tensor_buf.letterbox_params[0].pad_top)  / tensor_buf.letterbox_params[0].scale, 0.f, (float)tensor_buf.letterbox_params[0].orig_h);
        
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


std::vector<std::vector<Detection>> YoloDetector::postprocess(const TensorBuffer& tensor_buf, size_t batch) {   
    // ========== 1. 基础校验 ==========
    if (!tensor_buf.valid() || tensor_buf.shape.size() != 3) {
        return {};
    }
    assert(batch <= tensor_buf.shape[0] && "error found real batch size exceed batch size");
    
    // YOLOv8 输出形状: [1, numAttributes(4+num_classes), numPredictions]
    const int num_attributes = static_cast<int>(tensor_buf.shape[1]); // 例如 84
    const int num_predictions = static_cast<int>(tensor_buf.shape[2]); // 例如 8400
    const int num_classes = num_attributes - 4;

    std::vector<std::vector<Detection>> results(batch);
    const size_t plane_size = tensor_buf.plane_size();

    assert(tensor_buf.letterbox_params.size() >= batch);

#pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < batch; ++b) { // 只需要处理实际的批量大小即可
        const float* pdata = tensor_buf.data + b * plane_size;
        std::vector<Detection> result;
    
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
            float x1 = std::clamp((cx - 0.5f * ow - tensor_buf.letterbox_params[b].pad_left) / tensor_buf.letterbox_params[b].scale, 0.f, (float)tensor_buf.letterbox_params[b].orig_w);
            float y1 = std::clamp((cy - 0.5f * oh - tensor_buf.letterbox_params[b].pad_top)  / tensor_buf.letterbox_params[b].scale, 0.f, (float)tensor_buf.letterbox_params[b].orig_h);
            float x2 = std::clamp((cx + 0.5f * ow - tensor_buf.letterbox_params[b].pad_left) / tensor_buf.letterbox_params[b].scale, 0.f, (float)tensor_buf.letterbox_params[b].orig_w);
            float y2 = std::clamp((cy + 0.5f * oh - tensor_buf.letterbox_params[b].pad_top)  / tensor_buf.letterbox_params[b].scale, 0.f, (float)tensor_buf.letterbox_params[b].orig_h);
            
            // 过滤无效框
            if (x2 <= x1 || y2 <= y1) {
                continue;
            } 
            
            // 转为 cv::Rect 用于 NMS (OpenCV NMSBoxes 要求 Rect)
            boxes.emplace_back(
                static_cast<int>(std::round(x1)), static_cast<int>(std::round(y1)),
                static_cast<int>(std::round(x2 - x1)), static_cast<int>(std::round(y2 - y1))
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
        result.reserve(indexes.size());
        for (int idx : indexes) {
            result.push_back({
                boxes[idx],
                confidences[idx],
                class_ids[idx]
            });
        }

        // 将并行结果移动到结果中
        results[b] = std::move(result);
    }
    
    return results;
}


std::vector<Detection> YoloDetector::detect(const cv::Mat& img) {
    TIMER_START_TAG(preprocess);
    TensorBuffer tensor = preprocess(img);
    TIMER_FINISH_TAG(preprocess)
    TIMER_START_TAG(infer);
    ModelOutput infer_out = infer();

    infer_out.primary().letterbox_params = tensor.letterbox_params;
    TIMER_FINISH_TAG(infer);

    TIMER_START_TAG(postprocess);
    auto res = postprocess(infer_out.primary());
    TIMER_FINISH_TAG(postprocess);


    return res;
}

std::vector<std::vector<Detection>> YoloDetector::detect(const std::vector<cv::Mat>& imgs) {
    TIMER_START();

    preprocess(imgs);
    TIMER_ASPECT(preprocess);

    ModelOutput infer_out = infer();
    TIMER_ASPECT(infer);

    auto res = postprocess(infer_out.primary(), imgs.size());
    TIMER_ASPECT(postprocess);

    return res;
}