#pragma once

#include "Bench.h"

// kernel下
class BenchBase : public Bench
{

public:
    BenchBase(const YAML::Node& config) : Bench(config) {
    }

    LetterboxParams preprocess(const cv::Mat& img) override {
        // 所需要的参数应当通过上下文传入
        cv::Mat resized;
        // b, c, h, w
        LetterboxParams params = 
            letterbox_resize(img, resized, context_.img_size, context_.auto_aspect_ratio);
        
        convert_and_normalize(
            resized, h_input_.data(), 
            context_.img_size, 
            context_.norm_scale, 
            context_.bgr2rgb
        );
    
        return params;
    }

    void infer() override {
        // h2d 并设置状态
        cudaMemcpyAsync(
            d_buffer_.g_input.get(), 
            h_input_.data(),
            context_.num_input_bytes_size, 
            cudaMemcpyHostToDevice
        );

        context_.session->Run(
            Ort::RunOptions{}, 
            context_.input_names_c_str.data(), 
            &d_buffer_.input_tensor, context_.input_names_c_str.size(),
            context_.output_names_c_str.data(), 
            &d_buffer_.output_tensor, context_.output_names_c_str.size()
        );

        cudaMemcpyAsync(
            h_output_.data(), 
            d_buffer_.g_output.get(), 
            context_.num_output_bytes_size, 
            cudaMemcpyDeviceToHost
        );
    }

    // yolo后处理
    std::vector<std::vector<Detection>> postprocess(const std::vector<LetterboxParams>& params) override {

        int patch = params.size();
        // YOLOv8 输出形状: [1, numAttributes(4+num_classes), numPredictions]
        std::vector<std::vector<Detection>> results(patch);
    
    // #pragma omp parallel for schedule(dynamic)
        for (int b = 0; b < patch; ++b) { // 只需要处理实际的批量大小即可
            const float* pdata = h_output_.data() + b * context_.output_plane_size;
            std::vector<Detection> result;
        
            // ========== 2. 解码候选框 (替代原始的 transpose + row遍历) ==========
            // 💡 优化：直接在原始 [84, 8400] 布局上按列访问，避免 cv::transpose 的内存拷贝开销
            std::vector<cv::Rect2i> boxes;
            std::vector<float> confidences;
            std::vector<int> class_ids;
            boxes.reserve(context_.max_detections);
            confidences.reserve(context_.max_detections);
            class_ids.reserve(context_.max_detections);
            
            for (int i = 0; i < context_.num_predictions; ++i) {
                // 在 [84, 8400] 布局中，第 i 个预测框的数据起始位置
                // cx=pdata[i], cy=pdata[N+i], w=pdata[2N+i], h=pdata[3N+i]
                const float* box_ptr = pdata + i;
                const float* cls_ptr = pdata + 4 * context_.num_predictions + i;
                
                // Argmax 寻找最佳类别 (替代 minMaxLoc)
                float best_score = 0.f;
                int best_cls = 0;
                for (int c = 0; c < context_.num_classes; ++c) {
                    size_t id = c * context_.num_predictions;
                    float s = cls_ptr[c * context_.num_predictions];
                    
                    // LOG_INFO("IDX: {}", c * context_.num_predictions);
                    if (s > best_score) {
                        best_score = s;
                        best_cls = c;
                    }
                }
                
                if (best_score < context_.conf_threshold) continue;
                
                // 提取 cx, cy, w, h
                float cx = box_ptr[0];
                float cy = box_ptr[context_.num_predictions];
                float ow = box_ptr[2 * context_.num_predictions];
                float oh = box_ptr[3 * context_.num_predictions];
                
                // ⭐ 坐标还原到原图 (替代原始的 x_factor 乘法)
                float x1 = std::clamp((cx - 0.5f * ow - params[b].pad_left) / params[b].scale, 0.f, (float)params[b].orig_w);
                float y1 = std::clamp((cy - 0.5f * oh - params[b].pad_top)  / params[b].scale, 0.f, (float)params[b].orig_h);
                float x2 = std::clamp((cx + 0.5f * ow - params[b].pad_left) / params[b].scale, 0.f, (float)params[b].orig_w);
                float y2 = std::clamp((cy + 0.5f * oh - params[b].pad_top)  / params[b].scale, 0.f, (float)params[b].orig_h);
                
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
                cv::dnn::NMSBoxes(boxes, confidences, context_.conf_threshold, context_.nms_threshold, indexes);
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
};