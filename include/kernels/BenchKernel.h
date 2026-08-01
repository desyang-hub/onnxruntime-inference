#pragma once

#include "Bench.h"
#include "kernels/yolo/preprocess.cuh"
#include "kernels/yolo/postprocess.cuh"

// kernel下
class BenchKernel : public Bench
{
    const int kAttributeSize{6}; // x, y, w, h, class_id, score,
    int count_bytes_size_;

    CudaMallocGuard d_output_;
    CudaMallocGuard d_counts_;
    std::unique_ptr<int[]> h_counts_;
    

public:
    BenchKernel(const YAML::Node& config) : 
        Bench(config),
        count_bytes_size_(context_.batch_size * sizeof(int)),
        // batch * max_detections * 6 // batch 最大目标数量 * 类别数量
        d_output_(context_.batch_size * context_.max_detections * kAttributeSize * sizeof(float)),
        d_counts_(count_bytes_size_),
        h_counts_(std::make_unique<int[]>(context_.batch_size)) {
    }

    LetterboxParams preprocess(const cv::Mat& img) override {
        // CPU上计算参数，并将数据移动到GPU上
        float scale_h = static_cast<float>(context_.img_size.height) / img.rows;
        float scale_w = static_cast<float>(context_.img_size.width) / img.cols;

        float scale = std::min(scale_h, scale_w);
        int new_h = static_cast<int>(std::round(scale * img.rows));
        int new_w = static_cast<int>(std::round(scale * img.cols));

        new_h -= (new_h % context_.stride);
        new_w -= (new_w % context_.stride);

        scale = std::min(static_cast<float>(new_w) / img.cols,
                 static_cast<float>(new_h) / img.rows);

        int pad_left = (context_.img_size.width - new_w) / 2;
        int pad_top = (context_.img_size.height - new_h) / 2;

        // CPU计算好参数将数据传入GPU
        cudaMemcpyAsync(gpu_store.get(), img.data, img.rows * img.step, cudaMemcpyHostToDevice);

        // 在GPU上运行预处理，并将结果输出到GPU Buffer
        lunchPreprocess(
            gpu_store.get(),
            d_buffer_.g_input.get(),
            img.rows, img.cols, img.step,
            new_h, new_w,
            context_.img_size.height, context_.img_size.width,
            scale, pad_left, pad_top,
            context_.norm_scale, context_.pad_value
        );

        CUDA_CHECK(cudaStreamSynchronize(0));

        return LetterboxParams{scale, pad_left, pad_top, img.cols, img.rows};
    }


    void infer() override {
        // kernel 过程
        context_.session->Run(
            Ort::RunOptions{}, 
            context_.input_names_c_str.data(), &d_buffer_.input_tensor, context_.input_names_c_str.size(),
            context_.output_names_c_str.data(), &d_buffer_.output_tensor, context_.output_names_c_str.size()
        );
    }

    std::vector<std::vector<Detection>> postprocess(const std::vector<LetterboxParams>& params) override {
        // 只需要处理实际需要处理的batch数量即可
        int batch = params.size();  
        
        // postprocess filter kernel
        lunchPostprocessFilter(
            d_buffer_.g_output.get(), 
            d_output_.get(),
            d_counts_.get<int>(),
            context_.output_plane_size,
            batch,
            context_.num_predictions,
            context_.num_classes,
            context_.conf_threshold,
            context_.max_detections,
            0
        );

        // ========== Step 2: D2H 拷贝计数 ==========
        cudaMemcpyAsync(h_counts_.get(), d_counts_.get(), batch * sizeof(int), cudaMemcpyDeviceToHost, 0);

        // 同步
        CUDA_CHECK(cudaStreamSynchronize(0));

        // 钳制并计算总检测数
        int total_dets = 0;
        for (int i = 0; i < batch; ++i) {
            if (h_counts_[i] <= 0 || h_counts_[i] > context_.max_detections) {
                h_counts_[i] = std::clamp(h_counts_[i], 0, static_cast<int> (context_.max_detections));
            }
            total_dets += h_counts_[i];
        }

        if (total_dets == 0) {
            return {};
        }

        size_t oupt_plane_size = context_.max_detections * 6;

        // ========== Step 3: D2H 拷贝紧凑数据 ==========
        // ⚠️ 关键：stride 必须是 MAX_GPU_DETECTIONS，不是 host_counts[b]
        cudaMemcpyAsync(h_output_.data(), d_output_.get(),
                        batch * oupt_plane_size * sizeof(float),
                        cudaMemcpyDeviceToHost, 0);
        CUDA_CHECK(cudaStreamSynchronize(0));

        // ========== Step 4: CPU 解码 + NMS ==========
        std::vector<std::vector<Detection>> results(batch);

        for (size_t b = 0; b < batch; ++b) {
            const int cnt = h_counts_[b];
            if (cnt == 0) continue;

            // ⚠️ 关键：按 MAX_GPU_DETECTIONS 寻址，不是按 host_counts
            const float* pdata = h_output_.data() + b * oupt_plane_size;

            std::vector<cv::Rect2i> boxes;
            std::vector<float> confidences;
            std::vector<int> class_ids;
            boxes.reserve(cnt);
            confidences.reserve(cnt);
            class_ids.reserve(cnt);

            const auto& lp = params[b];

            for (int i = 0; i < cnt; ++i) {
                const float* det = pdata + i * 6;
                float cx = det[0], cy = det[1], w = det[2], h = det[3];
                float score = det[4];
                int cls = static_cast<int>(det[5]);

                float x1 = std::clamp((cx - 0.5f*w - lp.pad_left) / lp.scale, 0.f, (float)lp.orig_w);
                float y1 = std::clamp((cy - 0.5f*h - lp.pad_top)  / lp.scale, 0.f, (float)lp.orig_h);
                float x2 = std::clamp((cx + 0.5f*w - lp.pad_left) / lp.scale, 0.f, (float)lp.orig_w);
                float y2 = std::clamp((cy + 0.5f*h - lp.pad_top)  / lp.scale, 0.f, (float)lp.orig_h);

                boxes.emplace_back(cv::Rect2i(
                    static_cast<int>(x1), static_cast<int>(y1),
                    static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)));
                confidences.push_back(score);
                class_ids.push_back(cls);
            }

            // NMS
            std::vector<int> nms_indices;
            cv::dnn::NMSBoxes(boxes, confidences, context_.conf_threshold, context_.nms_threshold,nms_indices);

            std::vector<Detection>& batch_result = results[b];
            batch_result.reserve(nms_indices.size());
            for (int idx : nms_indices) {
                Detection d;
                d.box = boxes[idx];
                d.score = confidences[idx];
                d.class_id = class_ids[idx];
                batch_result.push_back(d);
            }
        }

        return results;
    }
};